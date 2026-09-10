"""Kernel-only Paged TurboQuant Q4 decode proof of concept.

This module deliberately owns neither page allocation nor request lifecycle.
It proves the narrow part needed to replace per-row segmented decode: a single
batched Metal attention interface can consume MSE-quantized K/V pages in an
arbitrary physical order while scanning only live tokens.

The initial specialization matches Qwen3.8-27B's full-attention geometry:
4-bit MSE K/V, head dimension 256, and 256-token pages.  Query/KV head counts
remain runtime shapes so the same code can exercise B=1 and B=2 tests.
"""

from collections.abc import Sequence
from functools import cache, lru_cache
from typing import NamedTuple

import mlx.core as mx

from .turboquant import (
    TurboQuantMSEState,
    _gen_unrolled_extract,
    _gen_unrolled_score,
    _gen_unrolled_value,
    _metal_available,
)

PAGED_TURBOQUANT_BITS = 4
PAGED_TURBOQUANT_DIM = 256
PAGED_TURBOQUANT_PAGE_SIZE = 256


class CompactPageSchedule(NamedTuple):
    """Device metadata derived from canonical per-request block tables.

    ``physical_page_ids`` contains only live logical pages.  Pages belonging
    to each request are consecutive and remain in logical-token order even
    when their physical IDs are arbitrarily permuted.  This avoids launching
    workgroups for the unused tail of a rectangular ``[B, max_pages]`` table.
    """

    physical_page_ids: mx.array
    page_owners: mx.array
    row_page_offsets: mx.array
    seq_lens: mx.array

    @property
    def num_logical_pages(self) -> int:
        return int(self.physical_page_ids.shape[0])


def _host_int_list(values) -> list[int]:
    if isinstance(values, mx.array):
        values = values.tolist()
    return [int(value) for value in values]


def build_compact_page_schedule(
    block_tables: Sequence[Sequence[int]] | mx.array,
    seq_lens: Sequence[int] | mx.array,
    *,
    page_size: int = PAGED_TURBOQUANT_PAGE_SIZE,
) -> CompactPageSchedule:
    """Compact canonical block tables into one entry per live logical page.

    This is a lifecycle operation, not a decode-step operation.  A real cache
    should rebuild the schedule only after admission, allocation, filtering,
    or release, then reuse the device arrays for subsequent decode steps.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    lengths = _host_int_list(seq_lens)
    if isinstance(block_tables, mx.array):
        tables = block_tables.tolist()
    else:
        tables = block_tables
    tables = [[int(page) for page in row] for row in tables]
    if len(tables) != len(lengths):
        raise ValueError("block table row count must equal seq_lens length")

    physical_ids: list[int] = []
    owners: list[int] = []
    offsets = [0]
    for row, (table, length) in enumerate(zip(tables, lengths)):
        if length <= 0:
            raise ValueError("paged decode requires every sequence to be non-empty")
        pages = (length + page_size - 1) // page_size
        if len(table) < pages:
            raise ValueError(f"row {row} needs {pages} pages but has {len(table)}")
        live = table[:pages]
        if any(page < 0 for page in live):
            raise ValueError(f"row {row} contains an invalid live physical page")
        physical_ids.extend(live)
        owners.extend([row] * pages)
        offsets.append(len(physical_ids))

    return CompactPageSchedule(
        mx.array(physical_ids, dtype=mx.int32),
        mx.array(owners, dtype=mx.int32),
        mx.array(offsets, dtype=mx.int32),
        mx.array(lengths, dtype=mx.int32),
    )


def page_mse_states(
    states: Sequence[TurboQuantMSEState],
    block_tables: Sequence[Sequence[int]] | mx.array,
    seq_lens: Sequence[int] | mx.array,
    *,
    page_size: int = PAGED_TURBOQUANT_PAGE_SIZE,
    num_physical_pages: int | None = None,
) -> tuple[TurboQuantMSEState, CompactPageSchedule]:
    """Copy already-quantized contiguous states into a synthetic page pool.

    This helper is intentionally an oracle/PoC bridge.  Production append will
    write newly quantized tokens directly to allocator-owned pages instead of
    repacking existing request states.
    """

    schedule = build_compact_page_schedule(block_tables, seq_lens, page_size=page_size)
    lengths = _host_int_list(seq_lens)
    if len(states) != len(lengths):
        raise ValueError("one contiguous MSE state is required per request")
    if not states:
        raise ValueError("at least one state is required")

    first = states[0]
    if not isinstance(first, TurboQuantMSEState):
        raise TypeError("paged Q4 PoC supports TurboQuantMSEState only")
    if first.norms.ndim != 3 or first.norms.shape[0] != 1:
        raise ValueError("each source state must have shape [1, Hkv, T, ...]")
    kv_heads = int(first.norms.shape[1])
    packed_width = int(first.indices.shape[-1])
    page_ids = _host_int_list(schedule.physical_page_ids)
    required_pages = max(page_ids, default=-1) + 1
    pool_pages = required_pages if num_physical_pages is None else num_physical_pages
    if pool_pages < required_pages:
        raise ValueError("num_physical_pages does not cover the block tables")

    norms = mx.zeros((pool_pages, kv_heads, page_size), dtype=first.norms.dtype)
    indices = mx.zeros(
        (pool_pages, kv_heads, page_size, packed_width),
        dtype=first.indices.dtype,
    )
    page_slot = 0
    for row, (state, length) in enumerate(zip(states, lengths)):
        if not isinstance(state, TurboQuantMSEState):
            raise TypeError("all source states must be TurboQuantMSEState")
        if (
            state.norms.shape[0] != 1
            or state.norms.shape[1] != kv_heads
            or state.indices.shape[-1] != packed_width
            or state.norms.shape[2] < length
        ):
            raise ValueError(f"incompatible source state for row {row}")
        row_start = int(schedule.row_page_offsets[row].item())
        row_end = int(schedule.row_page_offsets[row + 1].item())
        for logical_page, slot in enumerate(range(row_start, row_end)):
            physical_page = page_ids[slot]
            token_start = logical_page * page_size
            token_end = min(length, token_start + page_size)
            count = token_end - token_start
            norms[physical_page, :, :count] = state.norms[0, :, token_start:token_end]
            indices[physical_page, :, :count, :] = state.indices[
                0, :, token_start:token_end, :
            ]
            page_slot += 1
    if page_slot != schedule.num_logical_pages:
        raise AssertionError("internal compact schedule mismatch")
    mx.eval(norms, indices)
    return TurboQuantMSEState(norms, indices), schedule


@cache
def _paged_mse_decode_pass1_kernel(
    key_bits: int,
    val_bits: int,
    dim: int,
    page_size: int,
):
    if not _metal_available() or key_bits != 4 or val_bits != 4:
        return None
    if dim != PAGED_TURBOQUANT_DIM or page_size != PAGED_TURBOQUANT_PAGE_SIZE:
        return None

    elems_per_lane = dim // 32
    score = _gen_unrolled_score(key_bits, elems_per_lane)
    value_update = _gen_unrolled_value(val_bits, elems_per_lane)
    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        auto work_head = threadgroup_position_in_grid.z;
        auto kv_heads = key_norms_shape[1];
        auto page_slot = work_head / kv_heads;
        auto kv_head_idx = work_head % kv_heads;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;

        int row = page_owners[page_slot];
        int logical_page = int(page_slot) - row_page_offsets[row];
        int physical_page = physical_page_ids[page_slot];
        int remaining = seq_lens[row] - logical_page * PageSize;
        int valid_tokens = min(PageSize, max(remaining, 0));
        int q_head_idx = int(kv_head_idx) * RepeatCount + int(gqa_idx);
        int bqh = row * NumQueryHeads + q_head_idx;

        auto k_nm = key_norms +
            (physical_page * kv_heads + kv_head_idx) * PageSize;
        auto k_pk = key_packed +
            (physical_page * kv_heads + kv_head_idx) * PageSize * KPackedWidth;
        auto v_nm = val_norms +
            (physical_page * kv_heads + kv_head_idx) * PageSize;
        auto v_pk = val_packed +
            (physical_page * kv_heads + kv_head_idx) * PageSize * VPackedWidth;

        thread U q[qk_per_thread];
        auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        thread U o[v_per_thread] = {{}};
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        int k_byte_base = simd_lid * qk_per_thread * 4 / 8;
        int v_byte_base = simd_lid * v_per_thread * 4 / 8;

        for (int t = 0; t < valid_tokens; t++) {{
            U kn = static_cast<U>(k_nm[t]);
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) +
                k_byte_base;
            U score = {score};
            score = simd_sum(score) * kn;

            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) +
                v_byte_base;
            U vn = static_cast<U>(v_nm[t]);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            {value_update}
        }}

        int partial_head = int(page_slot) * NumQueryHeads + q_head_idx;
        if (simd_lid == 0) {{
            out_sums[partial_head] = sum_exp_score;
            out_maxs[partial_head] = max_score;
        }}
        for (int i = 0; i < v_per_thread; i++)
            out_acc[partial_head * Dim + simd_lid * v_per_thread + i] = o[i];
    """
    return mx.fast.metal_kernel(
        name="turboquant_paged_mse_q4_decode_pass1_d256_p256",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            "physical_page_ids",
            "page_owners",
            "row_page_offsets",
            "seq_lens",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )


@lru_cache(maxsize=1)
def _paged_mse_decode_pass2_kernel():
    if not _metal_available():
        return None
    source = r"""
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = Dim / BD;
        typedef float U;

        auto bqh = threadgroup_position_in_grid.x;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;
        int row = int(bqh) / NumQueryHeads;
        int q_head = int(bqh) - row * NumQueryHeads;
        int page_start = row_page_offsets[row];
        int page_end = row_page_offsets[row + 1];

        thread U o[elem_per_thread] = {};
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        for (int slot = page_start + int(simd_gid); slot < page_end; slot += BN) {
            int partial_head = slot * NumQueryHeads + q_head;
            U page_max = maxs[partial_head];
            U page_sum = sums[partial_head];
            if (!(page_sum > 0) || !isfinite(page_max))
                continue;
            U new_max = max(max_score, page_max);
            U old_factor = fast::exp(max_score - new_max);
            U page_factor = fast::exp(page_max - new_max);
            sum_exp_score = sum_exp_score * old_factor + page_sum * page_factor;
            for (int i = 0; i < elem_per_thread; i++)
                o[i] = o[i] * old_factor +
                    partials[partial_head * Dim + simd_lid * elem_per_thread + i] *
                    page_factor;
            max_score = new_max;
        }

        threadgroup U sg_maxs[BN];
        threadgroup U sg_sums[BN];
        threadgroup U outputs[BN * BD];
        if (simd_lid == 0) {
            sg_maxs[simd_gid] = max_score;
            sg_sums[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        U sg_max = sg_maxs[simd_lid];
        U global_max = simd_max(sg_max);
        U sg_factor = isfinite(global_max) ? fast::exp(sg_max - global_max) : 0;
        U total_sum = simd_sum(sg_sums[simd_lid] * sg_factor);
        U my_factor = isfinite(global_max) ? fast::exp(max_score - global_max) : 0;

        for (int i = 0; i < elem_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i] * my_factor;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = total_sum > 0 ? o[i] / total_sum : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; i++)
                out[int(bqh) * Dim + int(simd_gid) * elem_per_thread + i] = o[i];
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_paged_mse_q4_decode_pass2_d256",
        input_names=["partials", "sums", "maxs", "row_page_offsets"],
        output_names=["out"],
        source=source,
    )


@cache
def _singleton_paged_mse_decode_pass1_kernel(
    key_bits: int,
    val_bits: int,
    dim: int,
    page_size: int,
):
    """Split-K decode for one page-backed sequence.

    The generic paged kernel emits one partial per page.  That under-fills the
    GPU for singleton long-context decode (16K/P256 is only 64 pages), whereas
    the mature contiguous TurboQuant path uses 128 interleaved blocks.  This
    specialization keeps the exact same physical page pool and block table,
    but decouples execution parallelism from page size.
    """

    if not _metal_available() or key_bits != 4 or val_bits != 4:
        return None
    if dim != PAGED_TURBOQUANT_DIM or page_size != PAGED_TURBOQUANT_PAGE_SIZE:
        return None

    elems_per_lane = dim // 32
    score = _gen_unrolled_score(key_bits, elems_per_lane)
    value_update = _gen_unrolled_value(val_bits, elems_per_lane)
    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        auto kv_head_idx = threadgroup_position_in_grid.x;
        auto block_idx = threadgroup_position_in_grid.z;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;

        auto kv_heads = key_norms_shape[1];
        int token_count = seq_lens[0];
        int q_head_idx = int(kv_head_idx) * RepeatCount + int(gqa_idx);

        thread U q[qk_per_thread];
        auto qr = queries + q_head_idx * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        thread U o[v_per_thread] = {{}};
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        int k_byte_base = simd_lid * qk_per_thread * 4 / 8;
        int v_byte_base = simd_lid * v_per_thread * 4 / 8;

        // Hoist page-table lookup and integer division out of the token hot
        // loop. Blocks <= PageSize consume several lanes within each page;
        // larger fan-outs stripe whole logical pages between block phases.
        int block = int(block_idx);
        int page_lane = block % PageSize;
        int page_phase = block / PageSize;
        int page_stride = (Blocks + PageSize - 1) / PageSize;
        int logical_pages = (token_count + PageSize - 1) / PageSize;
        for (int logical_page = page_phase; logical_page < logical_pages;
             logical_page += page_stride) {{
            int physical_page = physical_page_ids[logical_page];
            auto k_nm = key_norms +
                (physical_page * kv_heads + kv_head_idx) * PageSize;
            auto k_pk = key_packed +
                (physical_page * kv_heads + kv_head_idx) * PageSize * KPackedWidth;
            auto v_nm = val_norms +
                (physical_page * kv_heads + kv_head_idx) * PageSize;
            auto v_pk = val_packed +
                (physical_page * kv_heads + kv_head_idx) * PageSize * VPackedWidth;
            int token_base = logical_page * PageSize;
            int token_stride = min(Blocks, PageSize);
            for (int page_token = page_lane;
                 page_token < PageSize && token_base + page_token < token_count;
                 page_token += token_stride) {{
                U kn = static_cast<U>(k_nm[page_token]);
                auto kb = (const device uint8_t*)(
                    k_pk + page_token * KPackedWidth) + k_byte_base;
                U score = {score};
                score = simd_sum(score) * kn;
                auto vb = (const device uint8_t*)(
                    v_pk + page_token * VPackedWidth) + v_byte_base;
                U vn = static_cast<U>(v_nm[page_token]);
                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                {value_update}
            }}
        }}

        int partial_head = q_head_idx * Blocks + int(block_idx);
        if (simd_lid == 0) {{
            out_sums[partial_head] = sum_exp_score;
            out_maxs[partial_head] = max_score;
        }}
        for (int i = 0; i < v_per_thread; i++)
            out_acc[partial_head * Dim + simd_lid * v_per_thread + i] = o[i];
    """
    return mx.fast.metal_kernel(
        name="turboquant_singleton_paged_mse_q4_decode_pass1_d256_p256",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            "physical_page_ids",
            "seq_lens",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )


@cache
def _batched_split_paged_mse_decode_pass1_kernel(
    key_bits: int,
    val_bits: int,
    dim: int,
    page_size: int,
):
    """Split-K decode whose execution blocks are independent of KV pages."""

    if not _metal_available() or key_bits != 4 or val_bits != 4:
        return None
    if dim != PAGED_TURBOQUANT_DIM or page_size != PAGED_TURBOQUANT_PAGE_SIZE:
        return None

    elems_per_lane = dim // 32
    score = _gen_unrolled_score(key_bits, elems_per_lane)
    value_update = _gen_unrolled_value(val_bits, elems_per_lane)
    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        int work = int(threadgroup_position_in_grid.z);
        int block_idx = work % Blocks;
        work /= Blocks;
        int kv_head_idx = work % NumKVHeads;
        int row = work / NumKVHeads;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;
        int q_head_idx = kv_head_idx * RepeatCount + int(gqa_idx);
        int bqh = row * NumQueryHeads + q_head_idx;
        int token_count = seq_lens[row];
        int page_table_start = row_page_offsets[row];

        thread U q[qk_per_thread];
        auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        thread U o[v_per_thread] = {{}};
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        int k_byte_base = simd_lid * qk_per_thread * 4 / 8;
        int v_byte_base = simd_lid * v_per_thread * 4 / 8;

        int page_lane = block_idx % PageSize;
        int page_phase = block_idx / PageSize;
        int page_stride = (Blocks + PageSize - 1) / PageSize;
        int logical_pages = (token_count + PageSize - 1) / PageSize;
        for (int logical_page = page_phase; logical_page < logical_pages;
             logical_page += page_stride) {{
            int physical_page = physical_page_ids[page_table_start + logical_page];
            auto k_nm = key_norms +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize;
            auto k_pk = key_packed +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize * KPackedWidth;
            auto v_nm = val_norms +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize;
            auto v_pk = val_packed +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize * VPackedWidth;

            int token_base = logical_page * PageSize;
            int token_stride = min(Blocks, PageSize);
            for (int page_token = page_lane;
                 page_token < PageSize && token_base + page_token < token_count;
                 page_token += token_stride) {{
                U kn = static_cast<U>(k_nm[page_token]);
                auto kb = (const device uint8_t*)(
                    k_pk + page_token * KPackedWidth) + k_byte_base;
                U score = {score};
                score = simd_sum(score) * kn;
                auto vb = (const device uint8_t*)(
                    v_pk + page_token * VPackedWidth) + v_byte_base;
                U vn = static_cast<U>(v_nm[page_token]);
                U new_max = max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                {value_update}
            }}
        }}

        int partial_head = bqh * Blocks + block_idx;
        if (simd_lid == 0) {{
            out_sums[partial_head] = sum_exp_score;
            out_maxs[partial_head] = max_score;
        }}
        for (int i = 0; i < v_per_thread; i++)
            out_acc[partial_head * Dim + simd_lid * v_per_thread + i] = o[i];
    """
    return mx.fast.metal_kernel(
        name="turboquant_batched_split_paged_mse_q4_decode_pass1_d256_p256",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            "physical_page_ids",
            "row_page_offsets",
            "seq_lens",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )


@lru_cache(maxsize=1)
def _singleton_paged_mse_decode_pass2_kernel():
    if not _metal_available():
        return None
    source = r"""
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = Dim / BD;
        typedef float U;

        auto head_idx = threadgroup_position_in_grid.x;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        thread U o[elem_per_thread] = {};
        U max_score = -INFINITY;
        U sum_exp_score = 0;
        for (int b = int(simd_gid); b < Blocks; b += BN) {
            int partial_head = int(head_idx) * Blocks + b;
            U block_max = maxs[partial_head];
            U block_sum = sums[partial_head];
            if (!(block_sum > 0) || !isfinite(block_max))
                continue;
            U new_max = max(max_score, block_max);
            U old_factor = fast::exp(max_score - new_max);
            U block_factor = fast::exp(block_max - new_max);
            sum_exp_score = sum_exp_score * old_factor + block_sum * block_factor;
            for (int i = 0; i < elem_per_thread; i++)
                o[i] = o[i] * old_factor +
                    partials[partial_head * Dim + simd_lid * elem_per_thread + i] *
                    block_factor;
            max_score = new_max;
        }

        threadgroup U sg_maxs[BN];
        threadgroup U sg_sums[BN];
        threadgroup U outputs[BN * BD];
        if (simd_lid == 0) {
            sg_maxs[simd_gid] = max_score;
            sg_sums[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        U sg_max = sg_maxs[simd_lid];
        U global_max = simd_max(sg_max);
        U sg_factor = isfinite(global_max) ? fast::exp(sg_max - global_max) : 0;
        U total_sum = simd_sum(sg_sums[simd_lid] * sg_factor);
        U my_factor = isfinite(global_max) ? fast::exp(max_score - global_max) : 0;

        for (int i = 0; i < elem_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i] * my_factor;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = total_sum > 0 ? o[i] / total_sum : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; i++)
                out[int(head_idx) * Dim + int(simd_gid) * elem_per_thread + i] = o[i];
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_singleton_paged_mse_q4_decode_pass2_d256",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=source,
    )


@cache
def _singleton_paged_mse_verify_qtile_kernel(
    key_bits: int,
    val_bits: int,
    dim: int,
    page_size: int,
    query_tile: int = 4,
):
    """Page-native MTP verification sharing one KV scan across Q=2..4."""

    if not _metal_available() or key_bits != 4 or val_bits != 4:
        return None
    if (
        dim != PAGED_TURBOQUANT_DIM
        or page_size != PAGED_TURBOQUANT_PAGE_SIZE
        or query_tile != 4
    ):
        return None

    elems_per_lane = dim // 32
    # Reuse the existing unrolled extraction generator while naming one
    # register per lane element. Q4/D256 is byte aligned.
    key_exprs = _gen_unrolled_extract(key_bits, elems_per_lane, "key_codebook")
    value_exprs = [
        expression.replace("kb[", "vb[")
        for expression in _gen_unrolled_extract(
            val_bits, elems_per_lane, "val_codebook"
        )
    ]

    query_decls = []
    accum_decls = []
    query_updates = []
    query_reductions = []
    for query_idx in range(query_tile):
        query_decls.extend(
            [
                f"        thread U query_{query_idx}[qk_per_thread];",
                f"        if ({query_idx} < QueryLength) {{",
                f"            auto query_ptr_{query_idx} = queries + (q_head_idx * QueryLength + {query_idx}) * Dim + simd_lid * qk_per_thread;",
                f"            for (int i = 0; i < qk_per_thread; i++) query_{query_idx}[i] = static_cast<U>(query_ptr_{query_idx}[i]);",
                "        }",
            ]
        )
        accum_decls.extend(
            [
                f"        thread U output_{query_idx}[v_per_thread] = {{}};",
                f"        U max_score_{query_idx} = -INFINITY;",
                f"        U sum_exp_score_{query_idx} = 0;",
            ]
        )
        score = "\n                    + ".join(
            f"query_{query_idx}[{i}] * key_value_{i}"
            for i in range(elems_per_lane)
        )
        value_updates = "\n".join(
            f"                output_{query_idx}[{i}] = output_{query_idx}[{i}] * factor_{query_idx} + exp_score_{query_idx} * value_{i} * value_norm;"
            for i in range(elems_per_lane)
        )
        query_updates.extend(
            [
                f"            if ({query_idx} < QueryLength && token_idx <= token_count - QueryLength + {query_idx}) {{",
                f"                U score_{query_idx} = {score};",
                f"                score_{query_idx} = simd_sum(score_{query_idx}) * key_norm;",
                f"                U new_max_{query_idx} = max(max_score_{query_idx}, score_{query_idx});",
                f"                U factor_{query_idx} = fast::exp(max_score_{query_idx} - new_max_{query_idx});",
                f"                U exp_score_{query_idx} = fast::exp(score_{query_idx} - new_max_{query_idx});",
                f"                max_score_{query_idx} = new_max_{query_idx};",
                f"                sum_exp_score_{query_idx} = sum_exp_score_{query_idx} * factor_{query_idx} + exp_score_{query_idx};",
                value_updates,
                "            }",
            ]
        )
        output_index = f"(q_head_idx * QueryLength + {query_idx}) * Dim"
        query_reductions.extend(
            [
                f"        if ({query_idx} < QueryLength) {{",
                "            if (simd_lid == 0) {",
                f"                max_scores[simd_gid] = max_score_{query_idx};",
                f"                sum_exp_scores[simd_gid] = sum_exp_score_{query_idx};",
                "            }",
                "            threadgroup_barrier(mem_flags::mem_threadgroup);",
                "            U simdgroup_max = simd_lid < BN ? max_scores[simd_lid] : -INFINITY;",
                "            U global_max = simd_max(simdgroup_max);",
                "            U simdgroup_factor = isfinite(global_max) ? fast::exp(simdgroup_max - global_max) : 0;",
                "            U simdgroup_sum = simd_lid < BN ? sum_exp_scores[simd_lid] : 0;",
                "            U total_sum = simd_sum(simdgroup_sum * simdgroup_factor);",
                f"            U output_factor = isfinite(global_max) ? fast::exp(max_score_{query_idx} - global_max) : 0;",
                "            for (int i = 0; i < v_per_thread; i++) {",
                f"                shared[simd_lid * BN + simd_gid] = output_{query_idx}[i] * output_factor;",
                "                threadgroup_barrier(mem_flags::mem_threadgroup);",
                "                U lane_value = simd_lid < BN ? shared[simd_gid * BN + simd_lid] : 0;",
                "                U reduced = simd_sum(lane_value);",
                "                if (simd_lid == 0)",
                f"                    out[{output_index} + simd_gid * v_per_thread + i] = static_cast<U>(total_sum > 0 ? reduced / total_sum : 0);",
                "                threadgroup_barrier(mem_flags::mem_threadgroup);",
                "            }",
                "        }",
            ]
        )

    key_decls = "\n".join(
        f"            U key_value_{i} = {expression};"
        for i, expression in enumerate(key_exprs)
    )
    value_decls = "\n".join(
        f"            U value_{i} = {expression};"
        for i, expression in enumerate(value_exprs)
    )
    source = f"""
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        int q_head_idx = int(threadgroup_position_in_grid.x);
        int kv_head_idx = q_head_idx / RepeatCount;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;
        int token_count = seq_lens[0];
        int logical_pages = (token_count + PageSize - 1) / PageSize;

{chr(10).join(query_decls)}
{chr(10).join(accum_decls)}

        int key_byte_base = simd_lid * qk_per_thread * 4 / 8;
        int value_byte_base = simd_lid * v_per_thread * 4 / 8;
        for (int logical_page = 0; logical_page < logical_pages; logical_page++) {{
            int physical_page = physical_page_ids[logical_page];
            auto key_norms_ptr = key_norms +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize;
            auto keys_ptr = key_packed +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize * KPackedWidth;
            auto value_norms_ptr = val_norms +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize;
            auto values_ptr = val_packed +
                (physical_page * NumKVHeads + kv_head_idx) * PageSize * VPackedWidth;
            int token_base = logical_page * PageSize;
            for (int page_token = int(simd_gid);
                 page_token < PageSize && token_base + page_token < token_count;
                 page_token += BN) {{
                int token_idx = token_base + page_token;
                U key_norm = static_cast<U>(key_norms_ptr[page_token]);
                U value_norm = static_cast<U>(value_norms_ptr[page_token]);
                auto kb = (const device uint8_t*)(
                    keys_ptr + page_token * KPackedWidth) + key_byte_base;
                auto vb = (const device uint8_t*)(
                    values_ptr + page_token * VPackedWidth) + value_byte_base;
{key_decls}
{value_decls}
{chr(10).join(query_updates)}
            }}
        }}

        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];
        threadgroup U shared[BD * BN];
{chr(10).join(query_reductions)}
    """
    return mx.fast.metal_kernel(
        name="turboquant_singleton_paged_mse_q4_verify_qtile_d256_p256",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            "physical_page_ids",
            "seq_lens",
        ],
        output_names=["out"],
        source=source,
    )


@cache
def _paged_mse_q4_direct_inverse_kernel(dim: int, page_size: int):
    """Direct Q4 page dequant with MLX-compatible inverse RHT per vector."""

    if not _metal_available() or dim != 256 or page_size != 256:
        return None
    header = r"""
#include <metal_simdgroup>
template <short R>
METAL_FUNC void tq_radix(thread float* x) {
    constexpr short logR = __builtin_ctz(R);
    short h = 1;
    #pragma clang loop unroll(full)
    for (short s = 0; s < logR; ++s) {
        #pragma clang loop unroll(full)
        for (short ii = 0; ii < R / 2; ++ii) {
            short k = ii & (h - 1);
            short j = ((ii - k) << 1) + k;
            float a = x[j];
            float b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        h <<= 1;
    }
}
"""
    source = r"""
        constexpr short Radix = 16;
        constexpr short Threads = Dim / Radix;
        short i = short(thread_index_in_threadgroup);
        int token = int(threadgroup_position_in_grid.y);
        int head = int(threadgroup_position_in_grid.z);
        int token_count = seq_lens[0];
        if (token >= token_count || head >= NumKVHeads) return;

        int logical_page = token / PageSize;
        int page_token = token - logical_page * PageSize;
        int physical_page = physical_page_ids[logical_page];
        int norm_index =
            (physical_page * NumKVHeads + head) * PageSize + page_token;
        int key_word_base = norm_index * KPackedWidth;
        int val_word_base = norm_index * VPackedWidth;
        threadgroup float buf[Dim];
        float x[Radix];

        #pragma clang loop unroll(full)
        for (short j = 0; j < Radix / 4; ++j) {
            short index = j * 4 * Threads + i * 4;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; ++r) {
                int d = int(index + r);
                int byte_in_vector = d >> 1;
                int word_in_vector = byte_in_vector >> 2;
                int byte_in_word = byte_in_vector & 3;
                uint packed_byte = (key_packed[key_word_base + word_in_vector]
                    >> (byte_in_word * 8)) & 255u;
                uint code = (packed_byte >> ((d & 1) * 4)) & 15u;
                buf[d] = static_cast<float>(key_codebook[code]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        short h = 1;
        #pragma clang loop unroll(full)
        for (short stage = 0; stage < 2; ++stage) {
            short k = i & (h - 1);
            short j = ((i - k) << 4) + k;
            #pragma clang loop unroll(full)
            for (short r = 0; r < Radix; ++r) x[r] = buf[j + h * r];
            tq_radix<Radix>(x);
            #pragma clang loop unroll(full)
            for (short r = 0; r < Radix; ++r) buf[j + h * r] = x[r];
            h <<= 4;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float key_norm = static_cast<float>(key_norms[norm_index]);
        #pragma clang loop unroll(full)
        for (short j = 0; j < Radix / 4; ++j) {
            short index = j * 4 * Threads + i * 4;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; ++r) {
                int d = int(index + r);
                int output_index = (head * token_count + token) * Dim + d;
                keys_out[output_index] = static_cast<T>(
                    buf[d] * (1.0f / 16.0f) * key_norm
                    * static_cast<float>(key_signs[d]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short j = 0; j < Radix / 4; ++j) {
            short index = j * 4 * Threads + i * 4;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; ++r) {
                int d = int(index + r);
                int byte_in_vector = d >> 1;
                int word_in_vector = byte_in_vector >> 2;
                int byte_in_word = byte_in_vector & 3;
                uint packed_byte = (val_packed[val_word_base + word_in_vector]
                    >> (byte_in_word * 8)) & 255u;
                uint code = (packed_byte >> ((d & 1) * 4)) & 15u;
                buf[d] = static_cast<float>(val_codebook[code]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        h = 1;
        #pragma clang loop unroll(full)
        for (short stage = 0; stage < 2; ++stage) {
            short k = i & (h - 1);
            short j = ((i - k) << 4) + k;
            #pragma clang loop unroll(full)
            for (short r = 0; r < Radix; ++r) x[r] = buf[j + h * r];
            tq_radix<Radix>(x);
            #pragma clang loop unroll(full)
            for (short r = 0; r < Radix; ++r) buf[j + h * r] = x[r];
            h <<= 4;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float val_norm = static_cast<float>(val_norms[norm_index]);
        #pragma clang loop unroll(full)
        for (short j = 0; j < Radix / 4; ++j) {
            short index = j * 4 * Threads + i * 4;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; ++r) {
                int d = int(index + r);
                int output_index = (head * token_count + token) * Dim + d;
                vals_out[output_index] = static_cast<T>(
                    buf[d] * (1.0f / 16.0f) * val_norm
                    * static_cast<float>(val_signs[d]));
            }
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_paged_q4_direct_inverse_d256_p256",
        input_names=[
            "key_norms",
            "key_packed",
            "key_codebook",
            "key_signs",
            "val_norms",
            "val_packed",
            "val_codebook",
            "val_signs",
            "physical_page_ids",
            "seq_lens",
        ],
        output_names=["keys_out", "vals_out"],
        header=header,
        source=source,
    )


def paged_mse_q4_prefill_direct_inverse_attention(
    queries: mx.array,
    key_pages: TurboQuantMSEState,
    value_pages: TurboQuantMSEState,
    schedule: CompactPageSchedule,
    *,
    key_codec,
    value_codec,
    scale: float,
    mask,
    page_size: int = PAGED_TURBOQUANT_PAGE_SIZE,
) -> mx.array:
    """Directly dequantize paged Q4 K/V to BF16, then use stock SDPA."""

    if queries.ndim != 4 or int(queries.shape[0]) != 1:
        raise ValueError("paged direct-inverse prefill requires [1, Hq, Q, D]")
    if int(schedule.seq_lens.shape[0]) != 1:
        raise ValueError("paged direct-inverse prefill requires one sequence")
    token_count = int(schedule.seq_lens[0].item())
    _, query_heads, _, dim = queries.shape
    kv_heads = int(key_pages.norms.shape[1])
    if dim != 256 or query_heads % kv_heads:
        raise ValueError("unsupported paged direct-inverse prefill geometry")
    kernel = _paged_mse_q4_direct_inverse_kernel(dim, page_size)
    if kernel is None:
        raise RuntimeError("paged direct-inverse dequant kernel is unavailable")
    keys, values = kernel(
        inputs=[
            key_pages.norms,
            key_pages.indices,
            key_codec.codebook,
            key_codec.signs,
            value_pages.norms,
            value_pages.indices,
            value_codec.codebook,
            value_codec.signs,
            schedule.physical_page_ids,
            schedule.seq_lens,
        ],
        template=[
            ("T", queries.dtype),
            ("Dim", dim),
            ("PageSize", page_size),
            ("NumKVHeads", kv_heads),
            ("KPackedWidth", key_pages.indices.shape[-1]),
            ("VPackedWidth", value_pages.indices.shape[-1]),
        ],
        grid=(16, token_count, kv_heads),
        threadgroup=(16, 1, 1),
        output_shapes=[
            (1, kv_heads, token_count, dim),
            (1, kv_heads, token_count, dim),
        ],
        output_dtypes=[queries.dtype, queries.dtype],
    )
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask
    )


def paged_mse_q4_verify_attention(
    queries: mx.array,
    key_pages: TurboQuantMSEState,
    value_pages: TurboQuantMSEState,
    schedule: CompactPageSchedule,
    *,
    key_codec,
    value_codec,
    scale: float,
    page_size: int = PAGED_TURBOQUANT_PAGE_SIZE,
) -> mx.array:
    """Causal singleton Q=2..4 attention without materializing paged K/V."""

    if not _metal_available():
        raise RuntimeError("paged TurboQuant verification requires MLX Metal")
    if queries.ndim != 4 or queries.shape[0] != 1:
        raise ValueError("verification queries must have shape [1, Hq, Q, D]")
    _, query_heads, query_length, dim = queries.shape
    kv_heads = int(key_pages.norms.shape[1])
    if (
        query_length < 2
        or query_length > 4
        or dim != PAGED_TURBOQUANT_DIM
        or page_size != PAGED_TURBOQUANT_PAGE_SIZE
        or key_codec.bits != PAGED_TURBOQUANT_BITS
        or value_codec.bits != PAGED_TURBOQUANT_BITS
        or query_heads % kv_heads != 0
        or schedule.seq_lens.shape[0] != 1
        or key_pages.norms.shape[2] != page_size
        or value_pages.norms.shape[:3] != key_pages.norms.shape[:3]
        or key_pages.indices.shape[-1] != 32
        or value_pages.indices.shape[-1] != 32
    ):
        raise ValueError("unsupported paged Q4 verification geometry or metadata")
    if int(schedule.seq_lens[0].item()) < query_length:
        raise ValueError("verification cache is shorter than the query tile")

    repeats = query_heads // kv_heads
    grouped = (queries * scale).reshape(
        1, kv_heads, repeats, query_length, dim
    )
    q_rot = key_codec.prepare_queries(grouped).reshape(
        query_heads * query_length, dim
    )
    kernel = _singleton_paged_mse_verify_qtile_kernel(4, 4, dim, page_size)
    if kernel is None:
        raise RuntimeError("paged Q4 verification kernel is unavailable")
    out = kernel(
        inputs=[
            q_rot,
            key_pages.norms,
            key_pages.indices,
            key_codec.codebook,
            value_pages.norms,
            value_pages.indices,
            value_codec.codebook,
            schedule.physical_page_ids,
            schedule.seq_lens,
        ],
        template=[
            ("Dim", dim),
            ("PageSize", page_size),
            ("RepeatCount", repeats),
            ("NumKVHeads", kv_heads),
            ("QueryLength", query_length),
            ("KPackedWidth", key_pages.indices.shape[-1]),
            ("VPackedWidth", value_pages.indices.shape[-1]),
        ],
        grid=(query_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(query_heads * query_length, dim)],
        output_dtypes=[mx.float32],
    )[0]
    rotated = out.reshape(1, kv_heads, repeats, query_length, dim)
    output = value_codec._rotate_inverse(rotated)
    return output.reshape(1, query_heads, query_length, dim).astype(queries.dtype)


def paged_mse_q4_decode_attention(
    queries: mx.array,
    key_pages: TurboQuantMSEState,
    value_pages: TurboQuantMSEState,
    schedule: CompactPageSchedule,
    *,
    key_codec,
    value_codec,
    scale: float,
    page_size: int = PAGED_TURBOQUANT_PAGE_SIZE,
) -> mx.array:
    """Decode one query token per row directly from arbitrarily ordered pages."""

    if not _metal_available():
        raise RuntimeError("paged TurboQuant decode requires MLX Metal")
    if queries.ndim != 4 or queries.shape[2] != 1:
        raise ValueError("queries must have shape [B, Hq, 1, D]")
    if not isinstance(key_pages, TurboQuantMSEState) or not isinstance(
        value_pages, TurboQuantMSEState
    ):
        raise TypeError("paged Q4 decode supports MSE page states only")
    batch, query_heads, _, dim = queries.shape
    kv_heads = int(key_pages.norms.shape[1])
    if (
        dim != PAGED_TURBOQUANT_DIM
        or page_size != PAGED_TURBOQUANT_PAGE_SIZE
        or key_codec.bits != PAGED_TURBOQUANT_BITS
        or value_codec.bits != PAGED_TURBOQUANT_BITS
        or query_heads % kv_heads != 0
        or schedule.seq_lens.shape[0] != batch
        or key_pages.norms.shape[2] != page_size
        or value_pages.norms.shape[:3] != key_pages.norms.shape[:3]
        or key_pages.indices.shape[-1] != 32
        or value_pages.indices.shape[-1] != 32
    ):
        raise ValueError("unsupported paged Q4 geometry or metadata")
    logical_pages = schedule.num_logical_pages
    if logical_pages == 0:
        raise ValueError("paged decode needs at least one live page")

    repeats = query_heads // kv_heads
    grouped = (queries * scale).reshape(batch, kv_heads, repeats, 1, dim)
    q_rot = key_codec.prepare_queries(grouped).reshape(batch * query_heads, dim)
    if batch == 1:
        token_count = int(schedule.seq_lens[0].item())
        if token_count <= 8192:
            num_blocks = 64
        elif token_count <= 32768:
            num_blocks = 128
        elif token_count <= 65536:
            num_blocks = 256
        else:
            num_blocks = 512
        singleton_pass1 = _singleton_paged_mse_decode_pass1_kernel(
            4, 4, dim, page_size
        )
        singleton_pass2 = _singleton_paged_mse_decode_pass2_kernel()
        if singleton_pass1 is not None and singleton_pass2 is not None:
            acc_shape = (query_heads * num_blocks, dim)
            sm_shape = (query_heads * num_blocks,)
            partials, sums, maxs = singleton_pass1(
                inputs=[
                    q_rot,
                    key_pages.norms,
                    key_pages.indices,
                    key_codec.codebook,
                    value_pages.norms,
                    value_pages.indices,
                    value_codec.codebook,
                    schedule.physical_page_ids,
                    schedule.seq_lens,
                ],
                template=[
                    ("Dim", dim),
                    ("PageSize", page_size),
                    ("RepeatCount", repeats),
                    ("Blocks", num_blocks),
                    ("KPackedWidth", key_pages.indices.shape[-1]),
                    ("VPackedWidth", value_pages.indices.shape[-1]),
                ],
                grid=(kv_heads * 32, repeats, num_blocks),
                threadgroup=(32, repeats, 1),
                output_shapes=[acc_shape, sm_shape, sm_shape],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )
            out = singleton_pass2(
                inputs=[partials, sums, maxs],
                template=[("Dim", dim), ("Blocks", num_blocks)],
                grid=(query_heads * 1024, 1, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(query_heads, dim)],
                output_dtypes=[mx.float32],
            )[0]
            rotated = out.reshape(1, kv_heads, repeats, dim)
            output = value_codec._rotate_inverse(rotated)
            return output.reshape(1, query_heads, 1, dim).astype(queries.dtype)

    # Page count is a storage concern, not a useful measure of GPU
    # parallelism.  Keep a fixed split-K fan-out for multi-row decode so short
    # and ragged batches do not under-fill the device merely because P=256.
    num_blocks = 128
    batched_pass1 = _batched_split_paged_mse_decode_pass1_kernel(
        4, 4, dim, page_size
    )
    split_pass2 = _singleton_paged_mse_decode_pass2_kernel()
    if batched_pass1 is not None and split_pass2 is not None:
        total_heads = batch * query_heads
        acc_shape = (total_heads * num_blocks, dim)
        sm_shape = (total_heads * num_blocks,)
        partials, sums, maxs = batched_pass1(
            inputs=[
                q_rot,
                key_pages.norms,
                key_pages.indices,
                key_codec.codebook,
                value_pages.norms,
                value_pages.indices,
                value_codec.codebook,
                schedule.physical_page_ids,
                schedule.row_page_offsets,
                schedule.seq_lens,
            ],
            template=[
                ("Dim", dim),
                ("PageSize", page_size),
                ("RepeatCount", repeats),
                ("NumQueryHeads", query_heads),
                ("NumKVHeads", kv_heads),
                ("Blocks", num_blocks),
                ("KPackedWidth", key_pages.indices.shape[-1]),
                ("VPackedWidth", value_pages.indices.shape[-1]),
            ],
            grid=(32, repeats, batch * kv_heads * num_blocks),
            threadgroup=(32, repeats, 1),
            output_shapes=[acc_shape, sm_shape, sm_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        out = split_pass2(
            inputs=[partials, sums, maxs],
            template=[("Dim", dim), ("Blocks", num_blocks)],
            grid=(total_heads * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(total_heads, dim)],
            output_dtypes=[mx.float32],
        )[0]
        rotated = out.reshape(batch, kv_heads, repeats, dim)
        output = value_codec._rotate_inverse(rotated)
        return output.reshape(batch, query_heads, 1, dim).astype(queries.dtype)

    pass1 = _paged_mse_decode_pass1_kernel(4, 4, dim, page_size)
    pass2 = _paged_mse_decode_pass2_kernel()
    if pass1 is None or pass2 is None:
        raise RuntimeError("paged Q4 Metal kernels are unavailable")

    partials, sums, maxs = pass1(
        inputs=[
            q_rot,
            key_pages.norms,
            key_pages.indices,
            key_codec.codebook,
            value_pages.norms,
            value_pages.indices,
            value_codec.codebook,
            schedule.physical_page_ids,
            schedule.page_owners,
            schedule.row_page_offsets,
            schedule.seq_lens,
        ],
        template=[
            ("Dim", dim),
            ("PageSize", page_size),
            ("RepeatCount", repeats),
            ("NumQueryHeads", query_heads),
            ("KPackedWidth", key_pages.indices.shape[-1]),
            ("VPackedWidth", value_pages.indices.shape[-1]),
        ],
        grid=(32, repeats, logical_pages * kv_heads),
        threadgroup=(32, repeats, 1),
        output_shapes=[
            (logical_pages, query_heads, dim),
            (logical_pages, query_heads),
            (logical_pages, query_heads),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = pass2(
        inputs=[partials, sums, maxs, schedule.row_page_offsets],
        template=[("Dim", dim), ("NumQueryHeads", query_heads)],
        grid=(batch * query_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch * query_heads, dim)],
        output_dtypes=[mx.float32],
    )[0]
    rotated = out.reshape(batch, kv_heads, repeats, dim)
    output = value_codec._rotate_inverse(rotated)
    return output.reshape(batch, query_heads, 1, dim).astype(queries.dtype)
