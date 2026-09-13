"""Fixed-capacity page-backed storage for TurboQuant MSE K/V states.

The host allocator in :mod:`mlx_vlm.paged_turboquant` owns page identity and
request lifetime.  This module owns the corresponding MLX tensors for one
model layer.  Appends copy already-quantized chunks directly into their
physical pages; they never rebuild the request's existing prefix.

The initial production target is integer Q4, whose key and value payloads are
both :class:`TurboQuantMSEState`.  The implementation intentionally keeps the
packed widths configurable so key and value codecs need not have identical
layouts.
"""

from __future__ import annotations

import mlx.core as mx

from .paged_turboquant import PageAllocator, PageAppend, PagedSequence
from .turboquant import TurboQuantMSEState

DEFAULT_PAGE_SIZE = 256


class PagedTurboQuantMSEStorage:
    """A fixed set of physical pages for one layer's quantized K/V payloads."""

    def __init__(
        self,
        allocator: PageAllocator,
        *,
        kv_heads: int,
        key_packed_width: int,
        value_packed_width: int | None = None,
        norm_dtype=mx.float16,
        index_dtype=mx.uint32,
    ):
        if not isinstance(allocator, PageAllocator):
            raise TypeError("allocator must be a PageAllocator")
        kv_heads = int(kv_heads)
        key_packed_width = int(key_packed_width)
        value_packed_width = int(
            key_packed_width if value_packed_width is None else value_packed_width
        )
        if kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if key_packed_width <= 0 or value_packed_width <= 0:
            raise ValueError("packed widths must be positive")

        self.allocator = allocator
        self.kv_heads = kv_heads
        self.key_packed_width = key_packed_width
        self.value_packed_width = value_packed_width
        page_shape = (
            allocator.capacity_pages,
            kv_heads,
            allocator.page_size,
        )
        self.keys = TurboQuantMSEState(
            mx.zeros(page_shape, dtype=norm_dtype),
            mx.zeros((*page_shape, key_packed_width), dtype=index_dtype),
        )
        self.values = TurboQuantMSEState(
            mx.zeros(page_shape, dtype=norm_dtype),
            mx.zeros((*page_shape, value_packed_width), dtype=index_dtype),
        )

    @classmethod
    def create(
        cls,
        capacity_pages: int,
        *,
        kv_heads: int,
        key_packed_width: int,
        value_packed_width: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        norm_dtype=mx.float16,
        index_dtype=mx.uint32,
    ) -> PagedTurboQuantMSEStorage:
        """Create storage and its allocator with a default 256-token page."""

        return cls(
            PageAllocator(capacity_pages=capacity_pages, page_size=page_size),
            kv_heads=kv_heads,
            key_packed_width=key_packed_width,
            value_packed_width=value_packed_width,
            norm_dtype=norm_dtype,
            index_dtype=index_dtype,
        )

    @property
    def page_size(self) -> int:
        return self.allocator.page_size

    @property
    def capacity_pages(self) -> int:
        return self.allocator.capacity_pages

    @staticmethod
    def _state_length(state: TurboQuantMSEState) -> int:
        return int(state.norms.shape[2])

    def _validate_source(
        self,
        state: TurboQuantMSEState,
        *,
        name: str,
        packed_width: int,
        token_count: int,
        norm_dtype,
        index_dtype,
    ) -> None:
        if not isinstance(state, TurboQuantMSEState):
            raise TypeError(f"{name} must be a TurboQuantMSEState")
        if state.norms.ndim != 3 or state.indices.ndim != 4:
            raise ValueError(f"{name} must use [1, Hkv, T, ...] layout")
        if (
            int(state.norms.shape[0]) != 1
            or int(state.indices.shape[0]) != 1
            or int(state.norms.shape[1]) != self.kv_heads
            or int(state.indices.shape[1]) != self.kv_heads
            or int(state.indices.shape[3]) != packed_width
        ):
            raise ValueError(f"{name} geometry does not match the page pool")
        if (
            int(state.norms.shape[2]) != token_count
            or int(state.indices.shape[2]) != token_count
        ):
            raise ValueError(f"{name} token count does not match the append span")
        if state.norms.dtype != norm_dtype or state.indices.dtype != index_dtype:
            raise ValueError(f"{name} dtypes do not match the page pool")

    def _validate_append(
        self,
        sequence: PagedSequence,
        append: PageAppend,
        keys: TurboQuantMSEState,
        values: TurboQuantMSEState,
    ) -> int:
        if not isinstance(sequence, PagedSequence):
            raise TypeError("sequence must be a PagedSequence")
        if not isinstance(append, PageAppend):
            raise TypeError("append must be a PageAppend")
        if sequence.allocator is not self.allocator:
            raise ValueError("sequence and storage must use the same allocator")
        if sequence.released:
            raise RuntimeError("paged sequence has been released")
        if append.sequence_identity != id(sequence):
            raise ValueError("append handle belongs to another sequence")
        if append.start < 0 or append.stop < append.start:
            raise ValueError("invalid append span")
        if sequence.length != append.stop:
            raise RuntimeError("append is no longer at the sequence tail")
        token_count = append.stop - append.start
        self._validate_source(
            keys,
            name="keys",
            packed_width=self.key_packed_width,
            token_count=token_count,
            norm_dtype=self.keys.norms.dtype,
            index_dtype=self.keys.indices.dtype,
        )
        self._validate_source(
            values,
            name="values",
            packed_width=self.value_packed_width,
            token_count=token_count,
            norm_dtype=self.values.norms.dtype,
            index_dtype=self.values.indices.dtype,
        )

        if token_count:
            first_page = append.start // self.page_size
            last_page = (append.stop - 1) // self.page_size
            if last_page >= len(sequence.page_ids):
                raise RuntimeError("append span is not covered by the block table")
            touched = sequence.page_ids[first_page : last_page + 1]
            if any(self.allocator.refcount(page_id) != 1 for page_id in touched):
                raise RuntimeError(
                    "cannot write shared pages before copy-on-write is implemented"
                )
        return token_count

    @staticmethod
    def _write_slice(
        destination: TurboQuantMSEState,
        source: TurboQuantMSEState,
        *,
        physical_page: int,
        page_start: int,
        source_start: int,
        count: int,
    ) -> None:
        page_stop = page_start + count
        source_stop = source_start + count
        destination.norms[physical_page, :, page_start:page_stop] = source.norms[
            0, :, source_start:source_stop
        ]
        destination.indices[physical_page, :, page_start:page_stop, :] = source.indices[
            0, :, source_start:source_stop, :
        ]

    def write_append(
        self,
        sequence: PagedSequence,
        append: PageAppend,
        keys: TurboQuantMSEState,
        values: TurboQuantMSEState,
    ) -> None:
        """Write an already-reserved append without copying its existing prefix.

        All shape, dtype, ownership, and span checks happen before either K or V
        is mutated.  The caller still owns metadata rollback if MLX evaluation
        itself fails.
        """

        token_count = self._validate_append(sequence, append, keys, values)
        if token_count == 0:
            return

        logical_position = append.start
        source_position = 0
        while logical_position < append.stop:
            logical_page, page_start = divmod(logical_position, self.page_size)
            physical_page = sequence.page_ids[logical_page]
            count = min(
                append.stop - logical_position,
                self.page_size - page_start,
            )
            self._write_slice(
                self.keys,
                keys,
                physical_page=physical_page,
                page_start=page_start,
                source_start=source_position,
                count=count,
            )
            self._write_slice(
                self.values,
                values,
                physical_page=physical_page,
                page_start=page_start,
                source_start=source_position,
                count=count,
            )
            logical_position += count
            source_position += count

        # Deliberately do not mx.eval() here: this method runs once per model
        # layer and evaluation at every layer would serialize a lazy forward.
        # The generation boundary materializes model output before scheduler
        # lifecycle code may release and recycle these physical pages.

    def write_page_run(
        self,
        sequence: PagedSequence,
        logical_page_start: int,
        keys: TurboQuantMSEState,
        values: TurboQuantMSEState,
    ) -> int:
        """Copy page-major packed APC payload into an existing block table."""

        if (
            not isinstance(sequence, PagedSequence)
            or sequence.allocator is not self.allocator
        ):
            raise ValueError("sequence and storage must use the same allocator")
        logical_page_start = int(logical_page_start)
        if logical_page_start < 0:
            raise ValueError("logical_page_start must be non-negative")

        def validate(state, name, packed_width, destination):
            if not isinstance(state, TurboQuantMSEState):
                raise TypeError(f"{name} must be a TurboQuantMSEState")
            if state.norms.ndim != 3 or state.indices.ndim != 4:
                raise ValueError(f"{name} must use [pages, Hkv, P, ...] layout")
            page_count = int(state.norms.shape[0])
            if (
                int(state.indices.shape[0]) != page_count
                or int(state.norms.shape[1]) != self.kv_heads
                or int(state.indices.shape[1]) != self.kv_heads
                or int(state.norms.shape[2]) != self.page_size
                or int(state.indices.shape[2]) != self.page_size
                or int(state.indices.shape[3]) != packed_width
            ):
                raise ValueError(f"{name} geometry does not match the page pool")
            if (
                state.norms.dtype != destination.norms.dtype
                or state.indices.dtype != destination.indices.dtype
            ):
                raise ValueError(f"{name} dtypes do not match the page pool")
            return page_count

        key_pages = validate(keys, "keys", self.key_packed_width, self.keys)
        value_pages = validate(values, "values", self.value_packed_width, self.values)
        if key_pages != value_pages:
            raise ValueError("key/value APC page counts do not match")
        logical_page_stop = logical_page_start + key_pages
        if logical_page_stop > len(sequence.page_ids):
            raise ValueError("paged APC run exceeds the reserved block table")
        physical = sequence.page_ids[logical_page_start:logical_page_stop]
        if any(self.allocator.refcount(page_id) != 1 for page_id in physical):
            raise RuntimeError("cannot restore into shared pages")

        source_start = 0
        while source_start < key_pages:
            source_stop = source_start + 1
            while (
                source_stop < key_pages
                and physical[source_stop] == physical[source_stop - 1] + 1
            ):
                source_stop += 1
            target_start = physical[source_start]
            target_stop = physical[source_stop - 1] + 1
            self.keys.norms[target_start:target_stop] = keys.norms[
                source_start:source_stop
            ]
            self.keys.indices[target_start:target_stop] = keys.indices[
                source_start:source_stop
            ]
            self.values.norms[target_start:target_stop] = values.norms[
                source_start:source_stop
            ]
            self.values.indices[target_start:target_stop] = values.indices[
                source_start:source_stop
            ]
            source_start = source_stop
        return key_pages

    @staticmethod
    def _empty_materialized(
        state: TurboQuantMSEState,
        *,
        kv_heads: int,
        packed_width: int,
    ) -> TurboQuantMSEState:
        return TurboQuantMSEState(
            mx.zeros((1, kv_heads, 0), dtype=state.norms.dtype),
            mx.zeros((1, kv_heads, 0, packed_width), dtype=state.indices.dtype),
        )

    def _materialize_state(
        self,
        sequence: PagedSequence,
        state: TurboQuantMSEState,
        *,
        packed_width: int,
    ) -> TurboQuantMSEState:
        if sequence.length == 0:
            return self._empty_materialized(
                state,
                kv_heads=self.kv_heads,
                packed_width=packed_width,
            )
        norm_chunks = []
        index_chunks = []
        remaining = sequence.length
        for physical_page in sequence.page_ids:
            count = min(remaining, self.page_size)
            norm_chunks.append(
                state.norms[physical_page : physical_page + 1, :, :count]
            )
            index_chunks.append(
                state.indices[physical_page : physical_page + 1, :, :count, :]
            )
            remaining -= count
        if remaining != 0:
            raise RuntimeError("sequence block table does not cover its logical length")
        return TurboQuantMSEState(
            mx.concatenate(norm_chunks, axis=2),
            mx.concatenate(index_chunks, axis=2),
        )

    def materialize(
        self, sequence: PagedSequence
    ) -> tuple[TurboQuantMSEState, TurboQuantMSEState]:
        """Reconstruct one logical row for debugging and prefill/APC fallback."""

        if not isinstance(sequence, PagedSequence):
            raise TypeError("sequence must be a PagedSequence")
        if sequence.allocator is not self.allocator:
            raise ValueError("sequence and storage must use the same allocator")
        if sequence.released:
            raise RuntimeError("paged sequence has been released")
        keys = self._materialize_state(
            sequence, self.keys, packed_width=self.key_packed_width
        )
        values = self._materialize_state(
            sequence, self.values, packed_width=self.value_packed_width
        )
        return keys, values
