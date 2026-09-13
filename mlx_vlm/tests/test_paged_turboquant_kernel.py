"""Metal correctness tests for the kernel-only paged TurboQuant Q4 PoC."""

import mlx.core as mx
import pytest

from mlx_vlm.paged_turboquant import PagedSequence
from mlx_vlm.paged_turboquant_kernel import (
    PAGED_TURBOQUANT_PAGE_SIZE,
    build_compact_page_schedule,
    page_mse_states,
    paged_mse_q4_decode_attention,
    paged_mse_q4_verify_attention,
)
from mlx_vlm.paged_turboquant_storage import PagedTurboQuantMSEStorage
from mlx_vlm.turboquant import TurboQuantKVCache

H_Q = 24
H_KV = 4
D = 256
BITS = 4
PAGE = PAGED_TURBOQUANT_PAGE_SIZE
SCALE = D**-0.5


def _make_rows(lengths):
    rows = []
    for length in lengths:
        cache = TurboQuantKVCache(bits=BITS)
        keys = mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16)
        values = mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)
        rows.append(cache)
    return rows


def _paged_output(rows, tables, queries, *, extra_pages=0):
    lengths = [row.offset for row in rows]
    max_page = max(page for table in tables for page in table if page >= 0)
    key_pages, schedule = page_mse_states(
        [row.state[0] for row in rows],
        tables,
        lengths,
        num_physical_pages=max_page + 1 + extra_pages,
    )
    value_pages, value_schedule = page_mse_states(
        [row.state[1] for row in rows],
        tables,
        lengths,
        num_physical_pages=max_page + 1 + extra_pages,
    )
    assert mx.array_equal(
        schedule.physical_page_ids, value_schedule.physical_page_ids
    ).item()
    output = paged_mse_q4_decode_attention(
        queries,
        key_pages,
        value_pages,
        schedule,
        key_codec=rows[0].key_codec,
        value_codec=rows[0].value_codec,
        scale=SCALE,
    )
    return output, key_pages, value_pages, schedule


def _segmented_reference(rows, queries):
    outputs = [
        row.decode_attention(queries[i : i + 1], scale=SCALE, mask=None)
        for i, row in enumerate(rows)
    ]
    return mx.concatenate(outputs, axis=0)


def _dequantized_reference(rows, queries):
    outputs = []
    for i, row in enumerate(rows):
        keys, values = row.dequantize()
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                queries[i : i + 1],
                keys.astype(queries.dtype),
                values.astype(queries.dtype),
                scale=SCALE,
                mask=None,
            )
        )
    return mx.concatenate(outputs, axis=0)


def _contiguous_verify_reference(row, queries):
    return row.prefill_attention(
        queries,
        scale=SCALE,
        mask="causal",
    )


def test_compact_schedule_drops_rectangular_padding():
    schedule = build_compact_page_schedule(
        [[7, 2, 9, 99], [4, 1, -1, -1]], [PAGE * 3, PAGE + 1]
    )
    assert schedule.physical_page_ids.tolist() == [7, 2, 9, 4, 1]
    assert schedule.page_owners.tolist() == [0, 0, 0, 1, 1]
    assert schedule.row_page_offsets.tolist() == [0, 3, 5]
    assert schedule.seq_lens.tolist() == [PAGE * 3, PAGE + 1]


def test_compact_schedule_rejects_missing_or_invalid_live_pages():
    with pytest.raises(ValueError, match="needs 2 pages"):
        build_compact_page_schedule([[0]], [PAGE + 1])
    with pytest.raises(ValueError, match="invalid live physical page"):
        build_compact_page_schedule([[0, -1]], [PAGE + 1])


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
@pytest.mark.parametrize(
    ("lengths", "tables"),
    [
        ([PAGE - 1], [[3]]),
        ([PAGE + 1], [[2, 0]]),
        ([PAGE * 2 + 17, PAGE + 1], [[4, 0, 3], [1, 2, -1]]),
        (
            [PAGE * 16 + 1, PAGE * 4 + 7],
            [list(range(20, 36)) + [3], [11, 9, 7, 5, 1]],
        ),
    ],
)
def test_paged_q4_matches_segmented_and_dequantized(lengths, tables):
    mx.random.seed(7301)
    rows = _make_rows(lengths)
    queries = mx.random.normal((len(rows), H_Q, 1, D)).astype(mx.bfloat16)
    paged, _, _, _ = _paged_output(rows, tables, queries, extra_pages=2)
    segmented = _segmented_reference(rows, queries)
    dequantized = _dequantized_reference(rows, queries)
    mx.eval(paged, segmented, dequantized)

    assert paged.shape == queries.shape
    assert mx.all(mx.isfinite(paged)).item()
    assert mx.allclose(paged, segmented, rtol=2e-2, atol=2e-2).item()
    assert mx.allclose(paged, dequantized, rtol=2e-2, atol=2e-2).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_tail_and_unallocated_pages_are_not_read():
    mx.random.seed(7302)
    rows = _make_rows([PAGE + 1, PAGE - 1])
    tables = [[4, 1], [3, -1]]
    queries = mx.random.normal((2, H_Q, 1, D)).astype(mx.bfloat16)
    expected, key_pages, value_pages, schedule = _paged_output(
        rows, tables, queries, extra_pages=2
    )
    mx.eval(expected)

    # Poison both unused physical pages and the invalid tail of every live
    # final page. Correct seq_len masking makes these values unobservable.
    key_pages.norms[0] = mx.full(key_pages.norms[0].shape, 60000, mx.float16)
    value_pages.norms[0] = mx.full(value_pages.norms[0].shape, 60000, mx.float16)
    key_pages.norms[2] = mx.full(key_pages.norms[2].shape, 60000, mx.float16)
    value_pages.norms[2] = mx.full(value_pages.norms[2].shape, 60000, mx.float16)
    key_pages.norms[1, :, 1:] = 60000
    value_pages.norms[1, :, 1:] = 60000
    key_pages.norms[3, :, PAGE - 1 :] = 60000
    value_pages.norms[3, :, PAGE - 1 :] = 60000
    mx.eval(key_pages.norms, value_pages.norms)

    actual = paged_mse_q4_decode_attention(
        queries,
        key_pages,
        value_pages,
        schedule,
        key_codec=rows[0].key_codec,
        value_codec=rows[0].value_codec,
        scale=SCALE,
    )
    mx.eval(actual)
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_page_backed_storage_feeds_kernel_without_contiguous_repack():
    mx.random.seed(7303)
    lengths = [PAGE * 3 + 11, PAGE + 5]
    rows = _make_rows(lengths)
    storage = PagedTurboQuantMSEStorage.create(
        8,
        kv_heads=H_KV,
        key_packed_width=32,
        value_packed_width=32,
    )
    # Reserve two pages first so both logical rows receive non-zero,
    # non-contiguous physical mappings after the blocker is partially freed.
    blockers = storage.allocator.allocate(2)
    sequences = []
    for index, (row, length) in enumerate(zip(rows, lengths)):
        if index == 1:
            storage.allocator.release(blockers[:1])
        sequence = PagedSequence(storage.allocator)
        append = sequence.append(length)
        storage.write_append(sequence, append, row.state[0], row.state[1])
        sequences.append(sequence)

    tables = [sequence.page_ids for sequence in sequences]
    schedule = build_compact_page_schedule(tables, lengths)
    queries = mx.random.normal((2, H_Q, 1, D)).astype(mx.bfloat16)
    actual = paged_mse_q4_decode_attention(
        queries,
        storage.keys,
        storage.values,
        schedule,
        key_codec=rows[0].key_codec,
        value_codec=rows[0].value_codec,
        scale=SCALE,
    )
    expected = _segmented_reference(rows, queries)
    mx.eval(actual, expected)

    assert sequences[0].page_ids != tuple(range(len(sequences[0].page_ids)))
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
@pytest.mark.parametrize("query_length", [2, 3, 4])
@pytest.mark.parametrize("token_count", [PAGE * 2 + 1, PAGE * 16 + 3])
def test_singleton_paged_verify_matches_contiguous_qtile(
    monkeypatch, query_length, token_count
):
    monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE", "1")
    mx.random.seed(7400 + query_length + token_count)
    row = _make_rows([token_count])[0]
    page_count = (token_count + PAGE - 1) // PAGE
    table = list(reversed(range(2, page_count + 2)))
    key_pages, schedule = page_mse_states(
        [row.state[0]],
        [table],
        [token_count],
        num_physical_pages=page_count + 3,
    )
    value_pages, _ = page_mse_states(
        [row.state[1]],
        [table],
        [token_count],
        num_physical_pages=page_count + 3,
    )
    queries = mx.random.normal((1, H_Q, query_length, D)).astype(mx.bfloat16)

    actual = paged_mse_q4_verify_attention(
        queries,
        key_pages,
        value_pages,
        schedule,
        key_codec=row.key_codec,
        value_codec=row.value_codec,
        scale=SCALE,
    )
    expected = _contiguous_verify_reference(row, queries)
    mx.eval(actual, expected)

    assert actual.shape == queries.shape
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()
