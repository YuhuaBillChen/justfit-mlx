"""Generator-scoped storage registry for paged TurboQuant caches.

The page-backed cache facade can create another request with ``new_empty()``,
but only after its first append has established the tensor geometry.  A
continuous-batching generator cannot rely on that ordering: two prompt batches
may be constructed independently before either one runs a model forward.

This module moves geometry ownership above request caches.  A registry eagerly
creates one fixed Q4 page pool per model cache leaf and every request facade is
born attached to that stable storage.  Joining independently-created prompt
caches is consequently a block-table operation; it never creates, replaces, or
copies the physical K/V pool.

The registry deliberately does not choose which model leaves are pageable.
Its keys are opaque hashable cache-leaf paths (for example ``12`` or
``(12, "attention")``), allowing model-specific cache trees to retain their
native identity.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Hashable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from weakref import WeakSet

import mlx.core as mx

from .paged_turboquant import PagePoolSpec, PagePoolStats, PagedPoolSet
from .paged_turboquant_cache import PagedBatchTurboQuantKVCache
from .paged_turboquant_kernel import (
    PAGED_TURBOQUANT_BITS,
    PAGED_TURBOQUANT_DIM,
    PAGED_TURBOQUANT_PAGE_SIZE,
)
from .paged_turboquant_storage import PagedTurboQuantMSEStorage
from .turboquant import DEFAULT_TURBOQUANT_SEED, _TurboQuantMSECodec

logger = logging.getLogger("mlx_vlm.paged_turboquant_pool")


@dataclass(frozen=True)
class PagedTurboQuantLayerSpec:
    """Fixed physical geometry for one pageable model cache leaf."""

    capacity_pages: int
    kv_heads: int
    page_size: int = PAGED_TURBOQUANT_PAGE_SIZE
    head_dim: int = PAGED_TURBOQUANT_DIM
    bits: int = PAGED_TURBOQUANT_BITS
    norm_dtype: object = mx.float16
    index_dtype: object = mx.uint32

    def __post_init__(self) -> None:
        if (
            int(self.capacity_pages) != self.capacity_pages
            or self.capacity_pages <= 0
        ):
            raise ValueError("capacity_pages must be positive")
        if int(self.kv_heads) != self.kv_heads or self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.page_size != PAGED_TURBOQUANT_PAGE_SIZE:
            raise ValueError(
                f"paged TurboQuant requires page_size={PAGED_TURBOQUANT_PAGE_SIZE}"
            )
        if self.head_dim != PAGED_TURBOQUANT_DIM:
            raise ValueError(
                f"paged TurboQuant requires head_dim={PAGED_TURBOQUANT_DIM}"
            )
        if self.bits != PAGED_TURBOQUANT_BITS:
            raise ValueError(f"paged TurboQuant requires Q{PAGED_TURBOQUANT_BITS}")
        if self.norm_dtype != mx.float16:
            raise ValueError("paged TurboQuant requires float16 norms")
        if self.index_dtype != mx.uint32:
            raise ValueError("paged TurboQuant requires uint32 packed indices")

    @property
    def packed_width(self) -> int:
        return (self.head_dim * self.bits + 31) // 32


@dataclass(frozen=True)
class PagedTurboQuantRegistryStats:
    """A generator-wide snapshot; page counts are summed across layer pools."""

    per_leaf: Mapping[Hashable, PagePoolStats]
    capacity_layer_pages: int
    used_layer_pages: int
    free_layer_pages: int
    high_water_layer_pages: int
    used_token_slots: int
    capacity_token_slots: int
    pool_nbytes: int
    live_facades: int
    closed: bool


class PagedTurboQuantPoolRegistry:
    """Own stable per-layer Q4 storage for one generator lifetime.

    All layer pools must have identical page counts and page size.  This keeps
    admission capacity consistent across pageable leaves and is a prerequisite
    for a later shared cross-layer reservation table.  Payload tensor geometry
    (notably ``kv_heads``) may still differ per model leaf.
    """

    def __init__(
        self,
        layer_specs: Mapping[Hashable, PagedTurboQuantLayerSpec],
        *,
        seed: int = DEFAULT_TURBOQUANT_SEED,
    ):
        if not layer_specs:
            raise ValueError("paged TurboQuant registry requires at least one leaf")
        specs = dict(layer_specs)
        for leaf_key, spec in specs.items():
            try:
                hash(leaf_key)
            except TypeError:
                raise TypeError("paged cache leaf keys must be hashable") from None
            if not isinstance(spec, PagedTurboQuantLayerSpec):
                raise TypeError(
                    f"specification for leaf {leaf_key!r} must be a "
                    "PagedTurboQuantLayerSpec"
                )

        capacities = {int(spec.capacity_pages) for spec in specs.values()}
        page_sizes = {int(spec.page_size) for spec in specs.values()}
        if len(capacities) != 1:
            raise ValueError("all paged leaves must use the same capacity_pages")
        if len(page_sizes) != 1:
            raise ValueError("all paged leaves must use the same page_size")

        self.seed = int(seed)
        self._specs = specs
        self._pools = PagedPoolSet(
            {
                leaf_key: PagePoolSpec(
                    capacity_pages=spec.capacity_pages,
                    page_size=spec.page_size,
                )
                for leaf_key, spec in specs.items()
            }
        )
        self._storages = {
            leaf_key: PagedTurboQuantMSEStorage(
                self._pools.allocator_for(leaf_key),
                kv_heads=spec.kv_heads,
                key_packed_width=spec.packed_width,
                value_packed_width=spec.packed_width,
                norm_dtype=spec.norm_dtype,
                index_dtype=spec.index_dtype,
            )
            for leaf_key, spec in specs.items()
        }
        # Codecs depend only on Q4/D256 and the generator seed, so sharing them
        # avoids independently-created requests constructing equivalent MLX
        # constants while preserving the existing K seed / V seed+1 contract.
        self._key_codec = _TurboQuantMSECodec(
            PAGED_TURBOQUANT_DIM, PAGED_TURBOQUANT_BITS, self.seed
        )
        self._value_codec = _TurboQuantMSECodec(
            PAGED_TURBOQUANT_DIM, PAGED_TURBOQUANT_BITS, self.seed + 1
        )
        # The registry must not retain every completed request forever.  A
        # weak set still lets shutdown release facades owned by the generator
        # while ordinary request teardown can reclaim its Python objects.
        self._facades: WeakSet[PagedBatchTurboQuantKVCache] = WeakSet()
        self._closed = False
        self._lock = threading.RLock()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("paged TurboQuant registry has been released")

    @property
    def leaf_keys(self) -> tuple[Hashable, ...]:
        return tuple(self._specs)

    def spec_for(self, leaf_key: Hashable) -> PagedTurboQuantLayerSpec:
        try:
            return self._specs[leaf_key]
        except (KeyError, TypeError):
            raise KeyError(f"unknown paged cache leaf {leaf_key!r}") from None

    def storage_for(self, leaf_key: Hashable) -> PagedTurboQuantMSEStorage:
        try:
            return self._storages[leaf_key]
        except (KeyError, TypeError):
            raise KeyError(f"unknown paged cache leaf {leaf_key!r}") from None

    def new_cache(self, leaf_key: Hashable) -> PagedBatchTurboQuantKVCache:
        """Create an empty B1 facade already bound to stable layer storage."""

        with self._lock:
            self._ensure_open()
            storage = self.storage_for(leaf_key)
            facade = PagedBatchTurboQuantKVCache(
                [0],
                bits=PAGED_TURBOQUANT_BITS,
                storage=storage,
                seed=self.seed,
                key_bits=PAGED_TURBOQUANT_BITS,
                value_bits=PAGED_TURBOQUANT_BITS,
                key_codec=self._key_codec,
                value_codec=self._value_codec,
            )
            self._facades.add(facade)
            return facade

    def new_cache_set(
        self, leaf_keys: Iterable[Hashable] | None = None
    ) -> dict[Hashable, PagedBatchTurboQuantKVCache]:
        """Create one independently-owned prompt cache for each requested leaf."""

        keys = self.leaf_keys if leaf_keys is None else tuple(leaf_keys)
        if len(keys) != len(set(keys)):
            raise ValueError("paged cache leaf keys must be unique")
        # Resolve every key before allocating any request facade so a typo does
        # not leave a partially-created prompt cache set in the registry.
        for leaf_key in keys:
            self.spec_for(leaf_key)
        return {leaf_key: self.new_cache(leaf_key) for leaf_key in keys}

    def restore_cache_list(self, caches):
        """Replace restored contiguous Q4 leaves with page-backed facades.

        Exact APC currently reconstructs packed ``TurboQuantKVCache`` objects
        from disk.  Conversion is layer-local and writes the packed payload
        directly into the registry's existing pool; it never dequantizes or
        reallocates the pool.
        """

        from .apc import PagedTurboQuantDiskRestore
        from .turboquant import BatchTurboQuantKVCache, TurboQuantKVCache, _slice_state

        restored = list(caches)
        created = []
        direct_disk_leaves = 0
        direct_disk_tokens = 0
        try:
            for leaf_key in self.leaf_keys:
                if not isinstance(leaf_key, int):
                    raise NotImplementedError(
                        "paged APC restore currently supports top-level cache leaves"
                    )
                source = restored[leaf_key]
                if isinstance(source, PagedTurboQuantDiskRestore):
                    spec = self.spec_for(leaf_key)
                    if (
                        source.bits != spec.bits
                        or source.key_bits != spec.bits
                        or source.value_bits != spec.bits
                        or source.seed != self.seed
                        or source.head_dim != spec.head_dim
                    ):
                        raise ValueError(
                            f"warm cache leaf {leaf_key!r} does not match page pool"
                        )
                    target = self.new_cache(leaf_key)
                    created.append(target)
                    target.restore_packed_page_runs(
                        source.iter_packed_page_runs(), source.offset
                    )
                    direct_disk_leaves += 1
                    direct_disk_tokens = max(direct_disk_tokens, source.offset)
                    restored[leaf_key] = target
                    continue
                if isinstance(source, BatchTurboQuantKVCache):
                    if source.batch_size != 1:
                        raise ValueError("paged APC restore requires a single warm row")
                    source = source.extract(0)
                if not isinstance(source, TurboQuantKVCache):
                    raise TypeError(
                        f"warm cache leaf {leaf_key!r} is not packed TurboQuant"
                    )
                target = self.new_cache(leaf_key)
                created.append(target)
                length = int(source.offset)
                if length:
                    target.restore_packed(
                        _slice_state(source.keys, length),
                        _slice_state(source.values, length),
                        length,
                    )
                restored[leaf_key] = target
        except Exception:
            for target in created:
                target.release()
            raise
        # The paged Q4 leaves already preserve ragged rows. Exact APC also
        # restores quality-policy float leaves (for example the final
        # full-attention layer) as BatchKVCache objects. Mark those leaves with
        # the same segmented-join policy used by a cold paged cache; otherwise
        # a later request with a different prefix length attempts a dense
        # concatenate along B before the sequence axes have been aligned.
        from .models.cache import BatchKVCache, CacheList

        def enable_segmented_join(entry):
            if isinstance(entry, BatchKVCache):
                entry.segment_on_extend = True
            elif isinstance(entry, CacheList):
                for child in entry.caches:
                    enable_segmented_join(child)

        for entry in restored:
            enable_segmented_join(entry)
        if direct_disk_leaves:
            logger.info(
                "Paged TurboQuant APC direct restore: leaves=%d "
                "sequence_tokens=%d contiguous_staging=0",
                direct_disk_leaves,
                direct_disk_tokens,
            )
        return restored

    def release_cache(self, cache: PagedBatchTurboQuantKVCache) -> None:
        """Release one request facade while retaining fixed generator storage."""

        if not isinstance(cache, PagedBatchTurboQuantKVCache):
            raise TypeError("cache must be a PagedBatchTurboQuantKVCache")
        with self._lock:
            if cache not in self._facades:
                raise ValueError("cache was not created by this registry")
            cache.release()

    @contextmanager
    def reserve_append(self, caches, token_count: int):
        """Reserve one append across every paged layer before model execution.

        The reservation is committed only after every layer consumed its
        handle. Allocation failure or a forward exception rolls all rows and
        layers back to their original lengths before pages become reusable.
        """

        token_count = int(token_count)
        if token_count <= 0:
            raise ValueError("reserved token count must be positive")
        cache_by_leaf = (
            caches
            if isinstance(caches, Mapping)
            else {leaf_key: caches[leaf_key] for leaf_key in self.leaf_keys}
        )
        reservations = []
        with self._lock:
            self._ensure_open()
            try:
                for leaf_key in self.leaf_keys:
                    facade = cache_by_leaf[leaf_key]
                    if not isinstance(facade, PagedBatchTurboQuantKVCache):
                        raise TypeError(f"cache leaf {leaf_key!r} is not paged")
                    if facade not in self._facades:
                        raise ValueError(
                            f"cache leaf {leaf_key!r} belongs to another registry"
                        )
                    if facade.storage is not self.storage_for(leaf_key):
                        raise ValueError(
                            f"cache leaf {leaf_key!r} has unexpected page storage"
                        )
                    appends = []
                    try:
                        for row in facade._rows.rows:
                            appends.append((row, row.reserve_append(token_count)))
                        facade._install_reserved_appends(appends, token_count)
                    except Exception:
                        for row, append in reversed(appends):
                            row.cancel_append(append)
                        raise
                    reservations.append((facade, tuple(appends)))
            except Exception:
                for facade, appends in reversed(reservations):
                    for row, append in reversed(appends):
                        row.cancel_append(append)
                    facade._clear_reserved_appends()
                raise

        try:
            yield
            missing = [
                facade
                for facade, _appends in reservations
                if not facade._reservation_consumed
            ]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} paged cache layers did not consume reservation"
                )
        except Exception:
            for facade, appends in reversed(reservations):
                facade._materialize_pending_writes()
                for row, append in reversed(appends):
                    if row.length == append.stop:
                        row.rollback(append)
                    elif row.length == append.start:
                        row.cancel_append(append)
                    else:
                        raise RuntimeError(
                            "paged sequence moved beyond its reserved append"
                        )
                facade._clear_reserved_appends()
            raise
        else:
            for facade, _appends in reservations:
                facade._clear_reserved_appends()

    def stats(self) -> PagedTurboQuantRegistryStats:
        with self._lock:
            per_leaf = self._pools.stats()
            values = tuple(per_leaf.values())
            pool_nbytes = sum(
                sum(
                    array.nbytes
                    for state in (storage.keys, storage.values)
                    for array in state
                )
                for storage in self._storages.values()
            )
            return PagedTurboQuantRegistryStats(
                per_leaf=MappingProxyType(per_leaf),
                capacity_layer_pages=sum(value.capacity_pages for value in values),
                used_layer_pages=sum(value.used_pages for value in values),
                free_layer_pages=sum(value.free_pages for value in values),
                high_water_layer_pages=sum(
                    value.high_water_pages for value in values
                ),
                used_token_slots=sum(
                    value.used_pages * value.page_size for value in values
                ),
                capacity_token_slots=sum(value.capacity_tokens for value in values),
                pool_nbytes=pool_nbytes,
                live_facades=sum(not cache._released for cache in self._facades),
                closed=self._closed,
            )

    def release(self) -> PagedTurboQuantRegistryStats:
        """Release every live request row and close the registry to admission."""

        with self._lock:
            if not self._closed:
                for cache in list(self._facades):
                    cache.release()
                self._closed = True
            return self.stats()
