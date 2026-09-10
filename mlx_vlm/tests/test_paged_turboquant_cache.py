"""Lifecycle and Metal correctness tests for the paged cache facade."""

import mlx.core as mx
import pytest

from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.paged_turboquant_cache import PagedBatchTurboQuantKVCache
from mlx_vlm.paged_turboquant_kernel import PAGED_TURBOQUANT_PAGE_SIZE
from mlx_vlm.turboquant import TurboQuantKVCache

H_Q = 24
H_KV = 4
D = 256
PAGE = PAGED_TURBOQUANT_PAGE_SIZE
SCALE = D**-0.5


def _kv(length, *, batch=1):
    return (
        mx.random.normal((batch, H_KV, length, D)).astype(mx.bfloat16),
        mx.random.normal((batch, H_KV, length, D)).astype(mx.bfloat16),
    )


def _oracle(keys, values):
    cache = TurboQuantKVCache(bits=4)
    cache.update_and_fetch(keys, values)
    return cache


def test_facade_rejects_unsupported_quantization_and_prefill_shapes():
    with pytest.raises(ValueError, match="Q4"):
        PagedBatchTurboQuantKVCache([0], bits=3, capacity_pages=4)
    with pytest.raises(ValueError, match="one unpadded row"):
        PagedBatchTurboQuantKVCache([0, 0], bits=4, capacity_pages=4)

    cache = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=4)
    with pytest.raises(ValueError, match="head_dim=256"):
        cache.update_and_fetch(mx.zeros((1, H_KV, 1, 64)), mx.zeros((1, H_KV, 1, 64)))


def test_b1_b2_b1_lifecycle_is_metadata_only_and_releases_pages():
    mx.random.seed(8101)
    active = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=8)
    active.update_and_fetch(*_kv(PAGE + 3))
    pending = active.new_empty()
    pending.update_and_fetch(*_kv(17))
    active_pages = active._rows.rows[0].page_ids
    pending_pages = pending._rows.rows[0].page_ids
    pool_identity = id(active.storage)

    active.extend(pending)

    assert id(active.storage) == pool_identity
    assert active.state[0] is active.storage.keys
    assert active.state[1] is active.storage.values
    assert active.sequence_lengths == (PAGE + 3, 17)
    assert active.offset.tolist() == [PAGE + 3, 17]
    assert active.left_padding.tolist() == [0, PAGE - 14]
    assert [row.page_ids for row in active._rows.rows] == [
        active_pages,
        pending_pages,
    ]
    assert active.physical_token_capacity == 3 * PAGE
    with pytest.raises(RuntimeError, match="released or moved"):
        pending.update_and_fetch(*_kv(1))

    active.update_and_fetch(*_kv(1, batch=2))
    assert active.sequence_lengths == (PAGE + 4, 18)
    first_pages_after_decode = active._rows.rows[0].page_ids
    assert first_pages_after_decode == active_pages

    active.filter(mx.array([1], dtype=mx.int32))
    assert active.sequence_lengths == (18,)
    assert active.left_padding.tolist() == [0]
    assert active._rows.rows[0].page_ids == pending_pages
    assert active.storage.allocator.stats().used_pages == 1

    active.update_and_fetch(*_kv(1))
    assert active.sequence_lengths == (19,)
    assert active.trim(18) == 18
    assert active.sequence_lengths == (1,)
    assert active.physical_token_capacity == PAGE
    assert active.nbytes < active.pool_nbytes

    active.release()
    assert active.storage.allocator.stats().used_pages == 0
    active.release()


def test_extend_rejects_an_independent_pool():
    first = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=2)
    second = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=2)
    first.update_and_fetch(*_kv(1))
    second.update_and_fetch(*_kv(1))

    with pytest.raises(ValueError, match="identical shared storage"):
        first.extend(second)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_paged_facade_decode_matches_segmented_oracle_through_b1_b2_b1():
    mx.random.seed(8102)
    first_kv = _kv(PAGE + 7)
    second_kv = _kv(31)
    decode_kv = _kv(1, batch=2)

    active = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=8)
    active.update_and_fetch(*first_kv)
    pending = active.new_empty()
    pending.update_and_fetch(*second_kv)
    active.extend(pending)
    active.update_and_fetch(*decode_kv)

    first_oracle = _oracle(*first_kv)
    second_oracle = _oracle(*second_kv)
    first_oracle.update_and_fetch(decode_kv[0][:1], decode_kv[1][:1])
    second_oracle.update_and_fetch(decode_kv[0][1:], decode_kv[1][1:])

    queries = mx.random.normal((2, H_Q, 1, D)).astype(mx.bfloat16)
    actual = active.decode_attention(queries, scale=SCALE, mask=None)
    expected = mx.concatenate(
        [
            first_oracle.decode_attention(queries[:1], scale=SCALE, mask=None),
            second_oracle.decode_attention(queries[1:], scale=SCALE, mask=None),
        ],
        axis=0,
    )
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()

    active.filter(mx.array([1], dtype=mx.int32))
    final_kv = _kv(1)
    active.update_and_fetch(*final_kv)
    second_oracle.update_and_fetch(*final_kv)
    final_query = mx.random.normal((1, H_Q, 1, D)).astype(mx.bfloat16)
    actual = active.decode_attention(final_query, scale=SCALE, mask=None)
    expected = second_oracle.decode_attention(final_query, scale=SCALE, mask=None)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_materialized_reference_state_matches_contiguous_oracle():
    mx.random.seed(8103)
    keys, values = _kv(PAGE + 1)
    paged = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=4)
    paged.update_and_fetch(keys, values)
    contiguous = _oracle(keys, values)

    paged_keys, paged_values = paged.materialize(0)
    oracle_keys, oracle_values = contiguous.state
    mx.eval(paged_keys, paged_values, oracle_keys, oracle_values)
    assert mx.array_equal(paged_keys.norms, oracle_keys.norms).item()
    assert mx.array_equal(paged_keys.indices, oracle_keys.indices).item()
    assert mx.array_equal(paged_values.norms, oracle_values.norms).item()
    assert mx.array_equal(paged_values.indices, oracle_values.indices).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_base_attention_dispatches_paged_prefill_without_reading_pool_as_dense():
    mx.random.seed(8104)
    keys, values = _kv(PAGE + 9)
    queries = mx.random.normal((1, H_Q, PAGE + 9, D)).astype(mx.bfloat16)
    paged = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=4)
    pool_keys, pool_values = paged.update_and_fetch(keys, values)

    actual = scaled_dot_product_attention(
        queries,
        pool_keys,
        pool_values,
        cache=paged,
        scale=SCALE,
        mask="causal",
    )
    logical_keys, logical_values = paged.materialize(0)
    oracle = TurboQuantKVCache(bits=4)
    oracle.key_codec = paged.key_codec
    oracle.value_codec = paged.value_codec
    expected = oracle.prefill_attention(
        queries,
        keys_state=logical_keys,
        values_state=logical_values,
        scale=SCALE,
        mask="causal",
    )
    if expected is None:
        float_keys, float_values = oracle.dequantize_for_attention(
            logical_keys, logical_values
        )
        expected = mx.fast.scaled_dot_product_attention(
            queries,
            float_keys.astype(queries.dtype),
            float_values.astype(queries.dtype),
            scale=SCALE,
            mask="causal",
        )

    mx.eval(actual, expected)
    assert actual.shape == queries.shape
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_direct_inverse_prefill_never_materializes_contiguous_kv(monkeypatch):
    mx.random.seed(8110)
    query_length = 8
    keys, values = _kv(PAGE + query_length)
    queries = mx.random.normal((1, H_Q, query_length, D)).astype(mx.bfloat16)
    paged = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=4)
    pool_keys, pool_values = paged.update_and_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        pool_keys,
        pool_values,
        cache=paged,
        scale=SCALE,
        mask="causal",
    )
    mx.eval(expected)
    monkeypatch.setenv("MLX_VLM_PAGED_PREFILL_IMPL", "direct_inverse")

    def fail_materialize(*args, **kwargs):
        raise AssertionError("direct-inverse prefill must not materialize KV")

    monkeypatch.setattr(paged, "materialize", fail_materialize)
    actual = scaled_dot_product_attention(
        queries,
        pool_keys,
        pool_values,
        cache=paged,
        scale=SCALE,
        mask="causal",
    )
    mx.eval(actual, expected)

    assert actual.shape == queries.shape
    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
def test_direct_inverse_prefill_is_bitwise_with_fragmented_physical_pages(
    monkeypatch,
):
    mx.random.seed(8111)
    owner = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=8)
    owner.update_and_fetch(*_kv(PAGE))
    target = owner.new_empty()
    target.update_and_fetch(*_kv(PAGE))
    blocker = owner.new_empty()
    blocker.update_and_fetch(*_kv(PAGE))
    query_length = 37
    target.update_and_fetch(*_kv(PAGE + query_length))
    assert target._rows.rows[0].page_ids[:3] == (1, 3, 4)

    queries = mx.random.normal((1, H_Q, query_length, D)).astype(mx.bfloat16)
    pool_keys, pool_values = target.state
    expected = scaled_dot_product_attention(
        queries,
        pool_keys,
        pool_values,
        cache=target,
        scale=SCALE,
        mask="causal",
    )
    mx.eval(expected)
    monkeypatch.setenv("MLX_VLM_PAGED_PREFILL_IMPL", "direct_inverse")

    def fail_materialize(*args, **kwargs):
        raise AssertionError("direct-inverse prefill must not materialize KV")

    monkeypatch.setattr(target, "materialize", fail_materialize)
    actual = scaled_dot_product_attention(
        queries,
        pool_keys,
        pool_values,
        cache=target,
        scale=SCALE,
        mask="causal",
    )
    mx.eval(actual)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal")
@pytest.mark.parametrize("query_length", [2, 3, 4])
def test_paged_mtp_qtile_dispatch_never_materializes_contiguous_kv(
    monkeypatch, query_length
):
    monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE", "1")
    mx.random.seed(8105 + query_length)
    keys, values = _kv(PAGE * 2 + 9)
    queries = mx.random.normal((1, H_Q, query_length, D)).astype(mx.bfloat16)
    paged = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=4)
    pool_keys, pool_values = paged.update_and_fetch(keys, values)

    oracle = _oracle(keys, values)
    expected = oracle.prefill_attention(
        queries,
        scale=SCALE,
        mask="causal",
    )

    def fail_materialize(*args, **kwargs):
        raise AssertionError("MTP verification must remain page-native")

    monkeypatch.setattr(paged, "materialize", fail_materialize)
    actual = scaled_dot_product_attention(
        queries,
        pool_keys,
        pool_values,
        cache=paged,
        scale=SCALE,
        mask="causal",
    )
    mx.eval(actual, expected)

    assert actual.shape == queries.shape
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()
