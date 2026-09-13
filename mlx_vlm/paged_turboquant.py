"""Host-side page ownership primitives for paged TurboQuant KV caches.

This module deliberately has no MLX dependency.  It defines the allocator and
logical block-table lifecycle that the page-backed tensor storage and Metal
attention path build on.  Keeping ownership here makes allocation failures,
row filtering, and request cleanup testable without a GPU.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass


class PageAllocationError(RuntimeError):
    """Raised when a page request cannot fit without changing allocator state."""


@dataclass(frozen=True)
class PagePoolStats:
    capacity_pages: int
    used_pages: int
    free_pages: int
    high_water_pages: int
    page_size: int

    @property
    def capacity_tokens(self) -> int:
        return self.capacity_pages * self.page_size


@dataclass(frozen=True)
class PageAppend:
    """Metadata needed to undo an append when its payload write fails."""

    start: int
    stop: int
    new_page_ids: tuple[int, ...]
    sequence_identity: int | None = None


@dataclass(frozen=True)
class PagedWorkSchedule:
    """Compact page work consumed by a future batched Metal attention path."""

    physical_page_ids: tuple[int, ...]
    page_owner: tuple[int, ...]
    row_page_offsets: tuple[int, ...]


@dataclass(frozen=True)
class PagePoolSpec:
    """Fixed host-side capacity for one page-backed model layer/cache leaf."""

    capacity_pages: int
    page_size: int

    def __post_init__(self):
        if self.capacity_pages <= 0:
            raise ValueError("capacity_pages must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")


@dataclass(frozen=True)
class CrossLayerPageAppend:
    """Rollback handle for one append reserved across every paged layer."""

    start: int
    stop: int
    layer_appends: tuple[tuple[Hashable, PageAppend], ...]
    sequence_identity: int


class PageAllocator:
    """Deterministic fixed-capacity page allocator with reference counts."""

    def __init__(self, capacity_pages: int, page_size: int):
        if capacity_pages <= 0:
            raise ValueError("capacity_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.capacity_pages = int(capacity_pages)
        self.page_size = int(page_size)
        self._free = list(range(self.capacity_pages))
        heapq.heapify(self._free)
        self._refcounts = [0] * self.capacity_pages
        self._used_pages = 0
        self._high_water_pages = 0
        self._lock = threading.RLock()

    def _validated_unique_ids(self, page_ids: Iterable[int]) -> tuple[int, ...]:
        ids = tuple(int(page_id) for page_id in page_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("page IDs must be unique within one operation")
        if any(page_id < 0 or page_id >= self.capacity_pages for page_id in ids):
            raise IndexError("page ID is outside allocator capacity")
        return ids

    def allocate(self, count: int) -> tuple[int, ...]:
        count = int(count)
        if count < 0:
            raise ValueError("page count must be non-negative")
        if count == 0:
            return ()
        with self._lock:
            if count > len(self._free):
                raise PageAllocationError(
                    f"requested {count} pages with only {len(self._free)} free"
                )
            page_ids = tuple(heapq.heappop(self._free) for _ in range(count))
            for page_id in page_ids:
                if self._refcounts[page_id] != 0:
                    raise AssertionError("free-list page still has a live reference")
                self._refcounts[page_id] = 1
            self._used_pages += count
            self._high_water_pages = max(self._high_water_pages, self._used_pages)
            return page_ids

    def retain(self, page_ids: Iterable[int]) -> None:
        ids = self._validated_unique_ids(page_ids)
        with self._lock:
            if any(self._refcounts[page_id] <= 0 for page_id in ids):
                raise RuntimeError("cannot retain a free page")
            for page_id in ids:
                self._refcounts[page_id] += 1

    def release(self, page_ids: Iterable[int]) -> None:
        ids = self._validated_unique_ids(page_ids)
        with self._lock:
            if any(self._refcounts[page_id] <= 0 for page_id in ids):
                raise RuntimeError("cannot release an unallocated page")
            for page_id in ids:
                self._refcounts[page_id] -= 1
                if self._refcounts[page_id] == 0:
                    heapq.heappush(self._free, page_id)
                    self._used_pages -= 1

    def refcount(self, page_id: int) -> int:
        ids = self._validated_unique_ids((page_id,))
        with self._lock:
            return self._refcounts[ids[0]]

    def stats(self) -> PagePoolStats:
        with self._lock:
            return PagePoolStats(
                capacity_pages=self.capacity_pages,
                used_pages=self._used_pages,
                free_pages=len(self._free),
                high_water_pages=self._high_water_pages,
                page_size=self.page_size,
            )


class PagedSequence:
    """One request's logical token sequence and physical page mapping."""

    def __init__(
        self,
        allocator: PageAllocator,
        *,
        page_ids: Iterable[int] = (),
        length: int = 0,
        retain_pages: bool = False,
    ):
        self.allocator = allocator
        self._page_ids = [int(page_id) for page_id in page_ids]
        self.length = int(length)
        self.released = False
        self._reserved_append: PageAppend | None = None
        if self.length < 0:
            raise ValueError("sequence length must be non-negative")
        required = self._required_pages(self.length)
        if required != len(self._page_ids):
            raise ValueError("page count does not match sequence length")
        if len(self._page_ids) != len(set(self._page_ids)):
            raise ValueError("a sequence cannot reference the same page twice")
        if retain_pages and self._page_ids:
            self.allocator.retain(self._page_ids)
        elif self._page_ids:
            for page_id in self._page_ids:
                if self.allocator.refcount(page_id) <= 0:
                    raise ValueError("sequence references an unallocated page")

    def _required_pages(self, length: int) -> int:
        return (length + self.allocator.page_size - 1) // self.allocator.page_size

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError("paged sequence has been released")

    @property
    def page_ids(self) -> tuple[int, ...]:
        return tuple(self._page_ids)

    @property
    def capacity_tokens(self) -> int:
        return len(self._page_ids) * self.allocator.page_size

    @property
    def tail_waste_tokens(self) -> int:
        return self.capacity_tokens - self.length

    def reserve_append(self, token_count: int) -> PageAppend:
        """Reserve physical tail pages without advancing logical length."""

        self._ensure_live()
        if self._reserved_append is not None:
            raise RuntimeError("paged sequence already has a reserved append")
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        start = self.length
        stop = start + token_count
        missing = self._required_pages(stop) - len(self._page_ids)
        new_page_ids = self.allocator.allocate(missing)
        append = PageAppend(
            start=start,
            stop=stop,
            new_page_ids=new_page_ids,
            sequence_identity=id(self),
        )
        self._reserved_append = append
        return append

    def commit_append(self, append: PageAppend) -> None:
        """Publish a physical reservation to cache/mask consumers."""

        self._ensure_live()
        if append.sequence_identity != id(self):
            raise ValueError("append handle belongs to another sequence")
        if self._reserved_append != append:
            raise RuntimeError("append is not the active physical reservation")
        if self.length != append.start:
            raise RuntimeError("sequence changed after append reservation")
        self._page_ids.extend(append.new_page_ids)
        self.length = append.stop
        self._reserved_append = None

    def cancel_append(self, append: PageAppend) -> None:
        """Release a reservation that has not advanced logical length."""

        self._ensure_live()
        if append.sequence_identity != id(self):
            raise ValueError("append handle belongs to another sequence")
        if self._reserved_append != append or self.length != append.start:
            raise RuntimeError("append reservation is no longer cancellable")
        self._reserved_append = None
        if append.new_page_ids:
            self.allocator.release(append.new_page_ids)

    def append(self, token_count: int) -> PageAppend:
        append = self.reserve_append(token_count)
        self.commit_append(append)
        return append

    def rollback(self, append: PageAppend) -> None:
        self._ensure_live()
        if append.sequence_identity != id(self):
            raise ValueError("append handle belongs to another sequence")
        if self.length != append.stop:
            raise RuntimeError("append rollback is no longer at the sequence tail")
        if append.start > append.stop:
            raise ValueError("invalid append span")
        new_page_ids = tuple(append.new_page_ids)
        if new_page_ids:
            if tuple(self._page_ids[-len(new_page_ids) :]) != new_page_ids:
                raise RuntimeError("append pages are no longer at the sequence tail")
            del self._page_ids[-len(new_page_ids) :]
            self.allocator.release(new_page_ids)
        self.length = append.start

    def truncate(self, new_length: int) -> None:
        self._ensure_live()
        if self._reserved_append is not None:
            raise RuntimeError("cannot truncate during an append reservation")
        new_length = int(new_length)
        if new_length < 0 or new_length > self.length:
            raise ValueError("new length must be between zero and current length")
        keep_pages = self._required_pages(new_length)
        released = tuple(self._page_ids[keep_pages:])
        del self._page_ids[keep_pages:]
        if released:
            self.allocator.release(released)
        self.length = new_length

    def release(self) -> None:
        if self.released:
            return
        if self._reserved_append is not None:
            self.cancel_append(self._reserved_append)
        page_ids = tuple(self._page_ids)
        self._page_ids.clear()
        self.length = 0
        self.released = True
        if page_ids:
            self.allocator.release(page_ids)


class PagedPoolSet:
    """Generator-scoped allocators shared by independently built prompt batches.

    A pool set contains one allocator per compatible model layer/cache leaf.
    Prompt batches created at different times must receive the same pool set so
    their per-layer rows can later be joined by metadata-only ``extend()``.
    """

    def __init__(self, layer_specs: Mapping[Hashable, PagePoolSpec]):
        if not layer_specs:
            raise ValueError("paged pool set requires at least one layer")
        specs = dict(layer_specs)
        if any(not isinstance(spec, PagePoolSpec) for spec in specs.values()):
            raise TypeError("every layer specification must be a PagePoolSpec")
        page_sizes = {spec.page_size for spec in specs.values()}
        if len(page_sizes) != 1:
            raise ValueError("all paged layers must use the same page_size")
        self._specs = specs
        self._allocators = {
            layer: PageAllocator(spec.capacity_pages, spec.page_size)
            for layer, spec in specs.items()
        }

    @classmethod
    def uniform(
        cls,
        layer_keys: Iterable[Hashable],
        *,
        capacity_pages: int,
        page_size: int,
    ) -> PagedPoolSet:
        keys = tuple(layer_keys)
        if len(keys) != len(set(keys)):
            raise ValueError("paged layer keys must be unique")
        spec = PagePoolSpec(capacity_pages=capacity_pages, page_size=page_size)
        return cls({key: spec for key in keys})

    @property
    def layer_keys(self) -> tuple[Hashable, ...]:
        return tuple(self._specs)

    @property
    def page_size(self) -> int:
        return next(iter(self._specs.values())).page_size

    def allocator_for(self, layer_key: Hashable) -> PageAllocator:
        try:
            return self._allocators[layer_key]
        except KeyError:
            raise KeyError(f"unknown paged layer {layer_key!r}") from None

    def spec_for(self, layer_key: Hashable) -> PagePoolSpec:
        try:
            return self._specs[layer_key]
        except KeyError:
            raise KeyError(f"unknown paged layer {layer_key!r}") from None

    def new_sequence(self, *, length: int = 0) -> CrossLayerPagedSequence:
        sequence = CrossLayerPagedSequence(self)
        if length:
            try:
                sequence.reserve_append(length)
            except Exception:
                sequence.release()
                raise
        return sequence

    def stats(self) -> dict[Hashable, PagePoolStats]:
        return {
            layer: allocator.stats() for layer, allocator in self._allocators.items()
        }


class CrossLayerPagedSequence:
    """A request's lockstep page reservations across all paged model layers.

    Reservation happens before the model forward.  If any layer lacks pages,
    every already-successful layer is rolled back before the error escapes.
    The returned handle can likewise undo the reservation when a later payload
    write or forward fails.
    """

    def __init__(self, pools: PagedPoolSet):
        if not isinstance(pools, PagedPoolSet):
            raise TypeError("pools must be a PagedPoolSet")
        self.pools = pools
        self._sequences = {
            layer: PagedSequence(pools.allocator_for(layer))
            for layer in pools.layer_keys
        }
        self.released = False
        self._lock = threading.RLock()

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError("cross-layer paged sequence has been released")

    def _synchronized_length(self) -> int:
        lengths = {sequence.length for sequence in self._sequences.values()}
        if len(lengths) != 1:
            raise RuntimeError("paged layer sequence lengths are out of sync")
        return next(iter(lengths))

    @property
    def length(self) -> int:
        with self._lock:
            return self._synchronized_length()

    def sequence_for(self, layer_key: Hashable) -> PagedSequence:
        try:
            return self._sequences[layer_key]
        except KeyError:
            raise KeyError(f"unknown paged layer {layer_key!r}") from None

    def reserve_append(self, token_count: int) -> CrossLayerPageAppend:
        """Advance every layer atomically and return a payload rollback handle."""
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        with self._lock:
            self._ensure_live()
            start = self._synchronized_length()
            completed: list[tuple[Hashable, PageAppend]] = []
            try:
                for layer, sequence in self._sequences.items():
                    completed.append((layer, sequence.append(token_count)))
            except Exception:
                for layer, append in reversed(completed):
                    self._sequences[layer].rollback(append)
                raise
            return CrossLayerPageAppend(
                start=start,
                stop=start + token_count,
                layer_appends=tuple(completed),
                sequence_identity=id(self),
            )

    def append(self, token_count: int) -> CrossLayerPageAppend:
        """Backward-friendly alias for ``reserve_append``."""
        return self.reserve_append(token_count)

    def rollback(self, append: CrossLayerPageAppend) -> None:
        with self._lock:
            self._ensure_live()
            if append.sequence_identity != id(self):
                raise ValueError("append handle belongs to another sequence")
            if self._synchronized_length() != append.stop:
                raise RuntimeError("append rollback is no longer at the sequence tail")
            expected_layers = tuple(self._sequences)
            handle_layers = tuple(layer for layer, _ in append.layer_appends)
            if handle_layers != expected_layers:
                raise ValueError("append handle does not cover this pool set")
            for layer, layer_append in reversed(append.layer_appends):
                self._sequences[layer].rollback(layer_append)

    def truncate(self, new_length: int) -> None:
        new_length = int(new_length)
        with self._lock:
            self._ensure_live()
            current = self._synchronized_length()
            if new_length < 0 or new_length > current:
                raise ValueError("new length must be between zero and current length")
            for sequence in self._sequences.values():
                sequence.truncate(new_length)

    def release(self) -> None:
        with self._lock:
            if self.released:
                return
            for sequence in self._sequences.values():
                sequence.release()
            self.released = True


class PagedBatchRows:
    """Movable ownership of ordered request rows sharing one page allocator."""

    def __init__(self, rows: Sequence[PagedSequence] = ()):
        self._rows = list(rows)
        allocators = {id(row.allocator): row.allocator for row in self._rows}
        if len(allocators) > 1:
            raise ValueError("all rows must use the same page allocator")
        if len({id(row) for row in self._rows}) != len(self._rows):
            raise ValueError("batch rows must be unique")
        if any(row.released for row in self._rows):
            raise ValueError("batch cannot contain a released sequence")
        self._allocator = next(iter(allocators.values()), None)

    @property
    def rows(self) -> tuple[PagedSequence, ...]:
        return tuple(self._rows)

    @property
    def allocator(self) -> PageAllocator | None:
        return self._allocator

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return tuple(row.length for row in self._rows)

    def extend(self, other: PagedBatchRows) -> None:
        if not isinstance(other, PagedBatchRows):
            raise TypeError("can only extend with PagedBatchRows")
        if other is self:
            raise ValueError("cannot extend a batch with itself")
        if not other._rows:
            return
        if self._allocator is None:
            self._allocator = other._allocator
        elif self._allocator is not other._allocator:
            raise ValueError("batches must use the same page allocator")
        existing = {id(row) for row in self._rows}
        if any(id(row) in existing for row in other._rows):
            raise ValueError("batch rows must be unique")
        self._rows.extend(other._rows)
        other._rows = []

    def filter(self, keep: Sequence[int]) -> None:
        indices = tuple(int(index) for index in keep)
        if len(indices) != len(set(indices)):
            raise ValueError("kept row indices must be unique")
        if any(index < 0 or index >= len(self._rows) for index in indices):
            raise IndexError("kept row index is out of range")
        kept_ids = set(indices)
        removed = [row for index, row in enumerate(self._rows) if index not in kept_ids]
        selected = [self._rows[index] for index in indices]
        for row in removed:
            row.release()
        self._rows = selected

    def release(self) -> None:
        for row in self._rows:
            row.release()
        self._rows = []

    def compact_schedule(self) -> PagedWorkSchedule:
        physical_page_ids = []
        page_owner = []
        row_page_offsets = [0]
        for row_index, row in enumerate(self._rows):
            physical_page_ids.extend(row.page_ids)
            page_owner.extend([row_index] * len(row.page_ids))
            row_page_offsets.append(len(physical_page_ids))
        return PagedWorkSchedule(
            physical_page_ids=tuple(physical_page_ids),
            page_owner=tuple(page_owner),
            row_page_offsets=tuple(row_page_offsets),
        )

    def block_tables(self, sentinel: int = -1) -> tuple[tuple[int, ...], ...]:
        width = max((len(row.page_ids) for row in self._rows), default=0)
        return tuple(
            row.page_ids + (int(sentinel),) * (width - len(row.page_ids))
            for row in self._rows
        )
