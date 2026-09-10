"""Single-layer paged TurboQuant cache facade.

This is an intentionally narrow integration scaffold: integer Q4 MSE K/V,
256-wide heads, 256-token pages, and one request per prefill batch.  It joins
the existing host allocator, fixed page storage, and batched paged Metal
decode kernel without changing the production cache dispatch.

The facade owns ordered :class:`PagedSequence` rows.  Independently prefetched
rows created by :meth:`new_empty` share the same physical storage and can be
joined with metadata only.  ``state``/``materialize`` are debugging and oracle
surfaces; paged decode consumes the page pool directly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import mlx.core as mx

from .models.cache import _BaseCache, create_causal_mask
from .paged_turboquant import PageAllocator, PageAppend, PagedBatchRows, PagedSequence
from .paged_turboquant_kernel import (
    PAGED_TURBOQUANT_BITS,
    PAGED_TURBOQUANT_DIM,
    PAGED_TURBOQUANT_PAGE_SIZE,
    build_compact_page_schedule,
    paged_mse_q4_decode_attention,
    paged_mse_q4_prefill_direct_inverse_attention,
    paged_mse_q4_verify_attention,
)
from .paged_turboquant_storage import PagedTurboQuantMSEStorage
from .turboquant import (
    DEFAULT_TURBOQUANT_SEED,
    TurboQuantKVCache,
    TurboQuantMSEState,
    _TurboQuantMSECodec,
    resolve_kv_bits,
)


class PagedTurboQuantAPCView:
    """Borrowed single-row page view consumed synchronously by exact APC.

    The object deliberately exposes physical page runs instead of a contiguous
    logical tensor.  This keeps direct disk checkpoints from allocating a
    second long-context KV cache while the live request still owns its pages.
    """

    def __init__(self, cache: "PagedBatchTurboQuantKVCache", row: int):
        cache._ensure_live()
        cache._materialize_pending_writes()
        self.cache = cache
        self.row = int(row)
        if self.row < 0 or self.row >= cache.batch_size:
            raise IndexError("paged APC row is out of range")

    def apc_page_runs(self) -> tuple[tuple[int, int], ...]:
        page_ids = self.cache._rows.rows[self.row].page_ids
        if not page_ids:
            return ()
        runs = []
        start = previous = int(page_ids[0])
        for raw_page_id in page_ids[1:]:
            page_id = int(raw_page_id)
            if page_id == previous + 1:
                previous = page_id
                continue
            runs.append((start, previous + 1))
            start = previous = page_id
        runs.append((start, previous + 1))
        return tuple(runs)

    @property
    def sequence_length(self) -> int:
        return self.cache.sequence_lengths[self.row]


class PagedBatchTurboQuantKVCache(_BaseCache):
    """Own one layer's ordered request rows in a shared Q4 page pool."""

    cache_step = PAGED_TURBOQUANT_PAGE_SIZE
    # Physical storage is allocated once by the generator-scoped page pool;
    # decode appends only reserve logical pages from that fixed capacity.
    decode_capacity_is_fixed = True

    def __init__(
        self,
        left_padding: Sequence[int],
        bits: float,
        *,
        capacity_pages: int | None = None,
        storage: PagedTurboQuantMSEStorage | None = None,
        seed: int = DEFAULT_TURBOQUANT_SEED,
        key_bits: float | None = None,
        value_bits: float | None = None,
        key_codec=None,
        value_codec=None,
    ):
        self.bits, self.key_bits, self.value_bits = resolve_kv_bits(
            bits, key_bits, value_bits
        )
        if (
            self.bits != PAGED_TURBOQUANT_BITS
            or self.key_bits != PAGED_TURBOQUANT_BITS
            or self.value_bits != PAGED_TURBOQUANT_BITS
        ):
            raise ValueError("paged TurboQuant facade supports Q4 K/V only")
        padding = tuple(int(value) for value in left_padding)
        if padding != (0,):
            raise ValueError(
                "paged TurboQuant prefill currently requires one unpadded row"
            )
        if storage is None:
            if capacity_pages is None:
                raise ValueError("capacity_pages is required without shared storage")
            allocator = PageAllocator(
                capacity_pages=int(capacity_pages),
                page_size=PAGED_TURBOQUANT_PAGE_SIZE,
            )
        else:
            if not isinstance(storage, PagedTurboQuantMSEStorage):
                raise TypeError("storage must be PagedTurboQuantMSEStorage")
            if storage.page_size != PAGED_TURBOQUANT_PAGE_SIZE:
                raise ValueError("paged decode requires 256-token pages")
            if (
                capacity_pages is not None
                and int(capacity_pages) != storage.capacity_pages
            ):
                raise ValueError("capacity_pages does not match shared storage")
            allocator = storage.allocator

        self.seed = int(seed)
        self.storage = storage
        self.key_codec = key_codec
        self.value_codec = value_codec
        self._rows = PagedBatchRows([PagedSequence(allocator)])
        self._schedule = None
        self._released = False
        self._has_pending_writes = False
        self._reserved_appends: tuple[tuple[PagedSequence, PageAppend], ...] | None = (
            None
        )
        self._reserved_token_count: int | None = None
        self._reservation_consumed = False
        self._validate_existing_codecs()

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("paged cache has been released or moved")

    def _validate_existing_codecs(self) -> None:
        codecs = (self.key_codec, self.value_codec)
        if codecs == (None, None):
            return
        if any(not isinstance(codec, _TurboQuantMSECodec) for codec in codecs):
            raise TypeError("paged TurboQuant facade requires MSE codecs")
        if any(
            codec.bits != PAGED_TURBOQUANT_BITS or codec.dim != PAGED_TURBOQUANT_DIM
            for codec in codecs
        ):
            raise ValueError("paged TurboQuant facade requires Q4 D=256 codecs")

    def _ensure_codecs(self, keys: mx.array, values: mx.array) -> None:
        if self.key_codec is None:
            helper = TurboQuantKVCache(
                bits=self.bits,
                seed=self.seed,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
            )
            helper._ensure_codecs(keys, values)
            self.key_codec = helper.key_codec
            self.value_codec = helper.value_codec
        self._validate_existing_codecs()

    def _validate_input(self, keys: mx.array, values: mx.array) -> None:
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
            raise ValueError("keys and values must have matching [B,H,T,D] shapes")
        if int(keys.shape[0]) != self.batch_size:
            raise ValueError("input batch does not match paged cache rows")
        if int(keys.shape[-1]) != PAGED_TURBOQUANT_DIM:
            raise ValueError("paged TurboQuant facade requires head_dim=256")
        if int(keys.shape[2]) > 1 and self.batch_size != 1:
            raise ValueError("paged prefill currently supports batch_size=1 only")
        if int(keys.shape[2]) <= 0:
            raise ValueError("append must contain at least one token")

    def _ensure_storage(
        self, keys: TurboQuantMSEState, values: TurboQuantMSEState
    ) -> None:
        if self.storage is None:
            allocator = self._rows.allocator
            if allocator is None:
                raise RuntimeError("paged cache has no allocator")
            self.storage = PagedTurboQuantMSEStorage(
                allocator,
                kv_heads=int(keys.norms.shape[1]),
                key_packed_width=int(keys.indices.shape[-1]),
                value_packed_width=int(values.indices.shape[-1]),
                norm_dtype=keys.norms.dtype,
                index_dtype=keys.indices.dtype,
            )
        if (
            self.storage.kv_heads != int(keys.norms.shape[1])
            or self.storage.key_packed_width != int(keys.indices.shape[-1])
            or self.storage.value_packed_width != int(values.indices.shape[-1])
            or self.storage.keys.norms.dtype != keys.norms.dtype
            or self.storage.values.norms.dtype != values.norms.dtype
            or self.storage.keys.indices.dtype != keys.indices.dtype
            or self.storage.values.indices.dtype != values.indices.dtype
        ):
            raise ValueError("quantized append geometry does not match page storage")

    @staticmethod
    def _state_row(state: TurboQuantMSEState, row: int) -> TurboQuantMSEState:
        return TurboQuantMSEState(
            state.norms[row : row + 1], state.indices[row : row + 1]
        )

    def _invalidate_schedule(self) -> None:
        self._schedule = None

    def _materialize_pending_writes(self) -> None:
        if self._has_pending_writes and self.storage is not None:
            mx.eval(self.storage.keys, self.storage.values)
            self._has_pending_writes = False

    def _install_reserved_appends(
        self,
        appends: Sequence[tuple[PagedSequence, PageAppend]],
        token_count: int,
    ) -> None:
        """Bind allocator reservations made atomically above model layers."""

        self._ensure_live()
        if self._reserved_appends is not None:
            raise RuntimeError("paged cache already has an active reservation")
        appends = tuple(appends)
        if len(appends) != self.batch_size:
            raise ValueError("reservation row count does not match paged cache")
        for expected_row, (row, append) in zip(self._rows.rows, appends):
            if row is not expected_row or append.sequence_identity != id(row):
                raise ValueError("reservation does not belong to this paged cache")
            if append.stop - append.start != int(token_count):
                raise ValueError("reservation token count does not match append")
        self._reserved_appends = appends
        self._reserved_token_count = int(token_count)
        self._reservation_consumed = False
        self._invalidate_schedule()

    def _clear_reserved_appends(self) -> None:
        self._reserved_appends = None
        self._reserved_token_count = None
        self._reservation_consumed = False
        self._invalidate_schedule()

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize one append and scatter it into each row's physical pages."""

        self._ensure_live()
        self._validate_input(keys, values)
        self._ensure_codecs(keys, values)
        helper = TurboQuantKVCache(
            bits=self.bits,
            seed=self.seed,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
        )
        helper.key_codec = self.key_codec
        helper.value_codec = self.value_codec
        quantized_keys, quantized_values = helper._try_fused_kv_quantize(
            keys, values
        )
        if quantized_keys is None:
            quantized_keys = self.key_codec.quantize(keys)
            quantized_values = self.value_codec.quantize(values)
        self._ensure_storage(quantized_keys, quantized_values)

        token_count = int(keys.shape[2])
        owns_appends = self._reserved_appends is None
        if owns_appends:
            appends = []
            try:
                for row in self._rows.rows:
                    appends.append((row, row.append(token_count)))
            except Exception:
                for row, append in reversed(appends):
                    row.rollback(append)
                raise
        else:
            if self._reservation_consumed:
                raise RuntimeError("paged cache reservation was already consumed")
            if token_count != self._reserved_token_count:
                raise ValueError("model append does not match reserved token count")
            appends = list(self._reserved_appends)
            for row, append in appends:
                row.commit_append(append)

        try:
            # Validate every row before scheduling the first page write.
            for index, (row, append) in enumerate(appends):
                self.storage._validate_append(
                    row,
                    append,
                    self._state_row(quantized_keys, index),
                    self._state_row(quantized_values, index),
                )
            for index, (row, append) in enumerate(appends):
                self.storage.write_append(
                    row,
                    append,
                    self._state_row(quantized_keys, index),
                    self._state_row(quantized_values, index),
                )
        except Exception:
            # Resolve any already-scheduled writes before recycling their pages.
            mx.eval(self.storage.keys, self.storage.values)
            if owns_appends:
                for row, append in reversed(appends):
                    row.rollback(append)
            raise

        self._has_pending_writes = True
        if not owns_appends:
            self._reservation_consumed = True
        self._invalidate_schedule()
        # Prefill chunks are lifecycle boundaries and should not retain an
        # unbounded lazy write graph. Decode appends stay lazy until attention.
        if token_count > 1:
            self._materialize_pending_writes()
        return self.storage.keys, self.storage.values

    def _decode_schedule(self):
        if self._schedule is None:
            self._schedule = build_compact_page_schedule(
                [row.page_ids for row in self._rows.rows],
                self.sequence_lengths,
                page_size=PAGED_TURBOQUANT_PAGE_SIZE,
            )
        return self._schedule

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: mx.array | None = None,
        **kwargs,
    ) -> mx.array:
        """Run one batched paged Metal attention path over all live rows."""

        del keys_state, values_state, kwargs
        self._ensure_live()
        if mask is not None:
            raise ValueError("paged decode currently supports mask=None only")
        if self.storage is None or self.empty():
            raise ValueError("cannot decode an empty paged cache")
        if int(queries.shape[0]) != self.batch_size:
            raise ValueError("query batch does not match paged cache rows")
        return paged_mse_q4_decode_attention(
            queries,
            self.storage.keys,
            self.storage.values,
            self._decode_schedule(),
            key_codec=self.key_codec,
            value_codec=self.value_codec,
            scale=scale,
            page_size=PAGED_TURBOQUANT_PAGE_SIZE,
        )

    def packed_decode_attention(self, *args, **kwargs) -> mx.array:
        return self.decode_attention(*args, **kwargs)

    @property
    def packed_verify_eligible(self) -> bool:
        return self.batch_size == 1 and self.storage is not None and not self.empty()

    def packed_verify_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: mx.array | str | None = None,
        **kwargs,
    ) -> mx.array:
        """Qwen exact-verifier hook that keeps singleton MTP page-native."""

        del keys_state, values_state, kwargs
        query_length = int(queries.shape[-2])
        causal_array = (
            isinstance(mask, mx.array)
            and mask.ndim >= 2
            and int(mask.shape[-2]) == query_length
            and int(mask.shape[-1]) == self.sequence_lengths[0]
        )
        if not (
            mask is None
            or isinstance(mask, str)
            and mask == "causal"
            or causal_array
        ):
            raise ValueError("paged MTP verification supports causal masking only")
        return paged_mse_q4_verify_attention(
            queries,
            self.storage.keys,
            self.storage.values,
            self._decode_schedule(),
            key_codec=self.key_codec,
            value_codec=self.value_codec,
            scale=scale,
            page_size=PAGED_TURBOQUANT_PAGE_SIZE,
        )

    def paged_attention(
        self,
        queries: mx.array,
        *,
        scale: float = 1.0,
        mask: mx.array | str | None = None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        """Dispatch attention while keeping the owning cache page-backed.

        Decode consumes the physical pool directly.  The first prefill path
        reconstructs only this request's packed Q4 logical view and tries the
        existing TurboQuant prefill kernel.  Its compatibility fallback
        dequantizes that one row, matching the existing MSE-Q4 behavior; it
        never creates a padded multi-row cache.  A native paged prefill kernel
        can replace this path without changing the runtime protocol.
        """

        if sinks is not None:
            raise ValueError("paged TurboQuant attention does not support sinks")
        if int(queries.shape[-2]) == 1:
            return self.decode_attention(queries, scale=scale, mask=mask)
        if self.batch_size != 1:
            raise ValueError("paged prefill currently supports batch_size=1 only")

        query_length = int(queries.shape[-2])
        causal_array = (
            isinstance(mask, mx.array)
            and mask.ndim >= 2
            and int(mask.shape[-2]) == query_length
            and int(mask.shape[-1]) == self.sequence_lengths[0]
        )
        if (
            os.environ.get("MLX_VLM_PAGED_PREFILL_IMPL") == "direct_inverse"
            and (isinstance(mask, str) and mask == "causal" or causal_array)
        ):
            if self.storage is None or self.empty():
                raise ValueError("cannot prefill against an empty paged cache")
            result = paged_mse_q4_prefill_direct_inverse_attention(
                queries,
                self.storage.keys,
                self.storage.values,
                self._decode_schedule(),
                key_codec=self.key_codec,
                value_codec=self.value_codec,
                scale=scale,
                mask=mask,
                page_size=PAGED_TURBOQUANT_PAGE_SIZE,
            )
            if os.environ.get("MLX_VLM_PAGED_PREFILL_EAGER_RELEASE") == "1":
                mx.eval(result)
            return result

        if (
            os.environ.get("MLX_VLM_TQ_MTP_QTILE") == "1"
            and isinstance(mask, str)
            and mask == "causal"
            and 2 <= query_length <= 4
        ):
            if self.storage is None or self.empty():
                raise ValueError("cannot verify against an empty paged cache")
            return self.packed_verify_attention(
                queries,
                scale=scale,
                mask=mask,
            )

        keys, values = self.materialize(0)
        helper = TurboQuantKVCache(
            bits=self.bits,
            seed=self.seed,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
        )
        helper.key_codec = self.key_codec
        helper.value_codec = self.value_codec
        result = helper.prefill_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=scale,
            mask=mask,
        )
        if result is None:
            float_keys, float_values = helper.dequantize_for_attention(keys, values)
            result = mx.fast.scaled_dot_product_attention(
                queries,
                float_keys.astype(queries.dtype),
                float_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )
        if os.environ.get("MLX_VLM_PAGED_PREFILL_EAGER_RELEASE") == "1":
            # Bound the lifetime of this layer's temporary contiguous and
            # dequantized compatibility views in an otherwise lazy forward.
            mx.eval(result)
        return result

    def new_empty(self) -> PagedBatchTurboQuantKVCache:
        """Create a B=1 prefill cache sharing this facade's page pool."""

        self._ensure_live()
        if self.storage is None:
            raise RuntimeError("shared storage is available after the first append")
        return type(self)(
            [0],
            bits=self.bits,
            storage=self.storage,
            seed=self.seed,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            key_codec=self.key_codec,
            value_codec=self.value_codec,
        )

    def extend(self, other: PagedBatchTurboQuantKVCache) -> None:
        """Move rows from another facade without copying any K/V payload."""

        self._ensure_live()
        if not isinstance(other, type(self)):
            raise TypeError("can only extend with a paged TurboQuant facade")
        other._ensure_live()
        if self._reserved_appends is not None or other._reserved_appends is not None:
            raise RuntimeError("cannot extend caches during an active reservation")
        if self.storage is None or self.storage is not other.storage:
            raise ValueError("paged cache extend requires identical shared storage")
        if (
            self.seed != other.seed
            or self.key_bits != other.key_bits
            or self.value_bits != other.value_bits
        ):
            raise ValueError("paged cache codecs are incompatible")
        self._rows.extend(other._rows)
        self._has_pending_writes = self._has_pending_writes or other._has_pending_writes
        other._released = True
        other._has_pending_writes = False
        other._schedule = None
        self._invalidate_schedule()

    def filter(self, batch_indices) -> None:
        """Keep/reorder selected rows and release every omitted row's pages."""

        self._ensure_live()
        if self._reserved_appends is not None:
            raise RuntimeError("cannot filter a cache during an active reservation")
        self._materialize_pending_writes()
        if isinstance(batch_indices, mx.array):
            batch_indices = batch_indices.tolist()
        self._rows.filter(batch_indices)
        self._invalidate_schedule()

    def trim(self, n: int) -> int:
        self._ensure_live()
        if self._reserved_appends is not None:
            raise RuntimeError("cannot trim a cache during an active reservation")
        self._materialize_pending_writes()
        n = max(0, int(n))
        trimmed = []
        for row in self._rows.rows:
            amount = min(row.length, n)
            row.truncate(row.length - amount)
            trimmed.append(amount)
        self._invalidate_schedule()
        return min(trimmed, default=0)

    def release(self) -> None:
        if self._released:
            return
        if self._reserved_appends is not None:
            raise RuntimeError("cannot release a cache during an active reservation")
        self._materialize_pending_writes()
        self._rows.release()
        self._released = True
        self._invalidate_schedule()

    def materialize(self, row: int) -> tuple[TurboQuantMSEState, TurboQuantMSEState]:
        """Reconstruct one contiguous logical row for debugging/oracles."""

        self._ensure_live()
        if self.storage is None:
            return None, None
        self._materialize_pending_writes()
        return self.storage.materialize(self._rows.rows[int(row)])

    def restore_packed(
        self,
        keys: TurboQuantMSEState,
        values: TurboQuantMSEState,
        token_count: int,
    ) -> None:
        """Restore one packed exact-APC row directly into owned pages."""

        self._ensure_live()
        if self.batch_size != 1 or not self.empty():
            raise ValueError("paged APC restore requires one empty row")
        if self.storage is None:
            raise RuntimeError("paged APC restore requires registry-backed storage")
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("paged APC restore length must be non-negative")
        if token_count == 0:
            return
        append = self._rows.rows[0].append(token_count)
        try:
            self.storage.write_append(
                self._rows.rows[0], append, keys, values
            )
            mx.eval(self.storage.keys, self.storage.values)
        except Exception:
            self._rows.rows[0].rollback(append)
            raise
        self._has_pending_writes = False
        self._invalidate_schedule()

    def restore_packed_page_runs(self, runs, token_count: int) -> None:
        """Stream page-major exact-APC runs into this row without staging KV."""

        self._ensure_live()
        if self.batch_size != 1 or not self.empty():
            raise ValueError("paged APC restore requires one empty row")
        if self.storage is None:
            raise RuntimeError("paged APC restore requires registry-backed storage")
        token_count = int(token_count)
        if token_count < 0:
            raise ValueError("paged APC restore length must be non-negative")
        if token_count == 0:
            return
        append = self._rows.rows[0].append(token_count)
        expected_pages = (token_count + self.cache_step - 1) // self.cache_step
        restored_pages = 0
        try:
            for kn, ki, vn, vi in runs:
                keys = TurboQuantMSEState(kn, ki)
                values = TurboQuantMSEState(vn, vi)
                restored_pages += self.storage.write_page_run(
                    self._rows.rows[0], restored_pages, keys, values
                )
                # Bound restore peak to one disk run plus the fixed page pool.
                mx.eval(self.storage.keys, self.storage.values)
                del keys, values, kn, ki, vn, vi
            if restored_pages != expected_pages:
                raise ValueError(
                    f"paged APC restored {restored_pages} pages, "
                    f"expected {expected_pages}"
                )
        except Exception:
            self._rows.rows[0].rollback(append)
            raise
        self._has_pending_writes = False
        self._invalidate_schedule()

    @property
    def reference_state(self):
        """Materialized per-row state; never used by the paged decode path."""

        return [self.materialize(index) for index in range(self.batch_size)]

    @property
    def eval_state(self):
        """Cheap page-backed arrays suitable for an outer ``mx.eval`` boundary."""

        if self.storage is None:
            return None, None
        return self.storage.keys, self.storage.values

    def synchronize(self) -> None:
        """Finish scheduled page writes without reconstructing logical rows."""

        self._ensure_live()
        self._materialize_pending_writes()

    @property
    def state(self):
        # Cache evaluation must not accidentally reconstruct every logical row.
        # ``reference_state`` is the explicit contiguous/oracle interface.
        return self.eval_state

    @state.setter
    def state(self, value):
        raise NotImplementedError(
            "restoring page storage and ownership metadata is not implemented"
        )

    def extract(self, idx: int) -> TurboQuantKVCache:
        """Return an owning contiguous TurboQuant cache for reference use."""

        cache = TurboQuantKVCache(
            bits=self.bits,
            seed=self.seed,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
        )
        cache.key_codec = self.key_codec
        cache.value_codec = self.value_codec
        keys, values = self.materialize(idx)
        if keys is not None:
            cache.state = (keys, values)
        return cache

    def extract_view(self, idx: int) -> PagedTurboQuantAPCView:
        """Borrow physical page runs for a synchronous exact APC write."""

        return PagedTurboQuantAPCView(self, idx)

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return self._rows.sequence_lengths

    @property
    def offset(self) -> mx.array:
        return mx.array(self.sequence_lengths, dtype=mx.int32)

    @property
    def left_padding(self) -> mx.array:
        maximum = max(self.sequence_lengths, default=0)
        return mx.array(
            [maximum - length for length in self.sequence_lengths], dtype=mx.int32
        )

    @property
    def _idx(self) -> int:
        return max(self.sequence_lengths, default=0)

    @property
    def batch_size(self) -> int:
        return len(self._rows.rows)

    def size(self) -> int:
        return self._idx

    def empty(self) -> bool:
        return not self.sequence_lengths or all(
            length == 0 for length in self.sequence_lengths
        )

    def is_trimmable(self) -> bool:
        return True

    def make_mask(self, n: int, **kwargs):
        kwargs.pop("return_array", None)
        return create_causal_mask(
            n, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    @property
    def keys(self):
        return None if self.storage is None else self.storage.keys

    @property
    def values(self):
        return None if self.storage is None else self.storage.values

    @property
    def physical_token_capacity(self) -> int:
        return sum(row.capacity_tokens for row in self._rows.rows)

    @property
    def pool_token_capacity(self) -> int:
        if self.storage is None:
            allocator = self._rows.allocator
            return 0 if allocator is None else allocator.stats().capacity_tokens
        return self.storage.capacity_pages * self.storage.page_size

    def _bytes_per_page(self) -> int:
        if self.storage is None:
            return 0
        return (
            sum(
                array.nbytes
                for state in (self.storage.keys, self.storage.values)
                for array in state
            )
            // self.storage.capacity_pages
        )

    @property
    def nbytes(self) -> int:
        """Bytes attributable to live physical pages, including tail waste."""

        if self.storage is None:
            return 0
        live_pages = sum(len(row.page_ids) for row in self._rows.rows)
        return live_pages * self._bytes_per_page()

    @property
    def pool_nbytes(self) -> int:
        """Bytes in the fixed physical tensor pool shared by all facades."""

        if self.storage is None:
            return 0
        return self.storage.capacity_pages * self._bytes_per_page()

    @property
    def group_size(self) -> int:
        return 64
