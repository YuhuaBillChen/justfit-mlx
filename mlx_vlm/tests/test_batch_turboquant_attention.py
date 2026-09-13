"""Tests for the TurboQuant attention path shared by both KV cache types.

``BatchTurboQuantKVCache`` used to fall back to dequantizing the whole cache
on every step, which made decode memory grow with the context length. It now
reuses the fused kernels through ``_TurboQuantAttentionMixin`` for aligned
batches and a left-padding-aware variant for ragged continuous batches.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.models.base import (
    _turboquant_attention_applies,
    scaled_dot_product_attention,
)
from mlx_vlm.models.cache import BatchKVCache, create_causal_mask
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _contiguous_batch_slice,
    _should_eval_cache_append,
    _TurboQuantAttentionMixin,
)

H, D = 4, 64  # kv heads, head_dim
BITS = 4
SCALE = D**-0.5


def test_small_mtp_append_can_stay_lazy(monkeypatch):
    monkeypatch.setenv("MLX_VLM_TQ_LAZY_VERIFY_APPEND", "1")
    assert not _should_eval_cache_append(3, 40363)
    assert _should_eval_cache_append(3, 40400)
    assert _should_eval_cache_append(16, 40363)


def test_original_eval_policy_is_default(monkeypatch):
    monkeypatch.delenv("MLX_VLM_TQ_LAZY_VERIFY_APPEND", raising=False)
    assert _should_eval_cache_append(3, 40363)


def _rand_kv(batch, seq_len, heads=H):
    k = mx.random.normal((batch, heads, seq_len, D))
    v = mx.random.normal((batch, heads, seq_len, D))
    return k, v


def _filled(left_padding, seq_len, batch=None, bits=BITS):
    batch = len(left_padding) if batch is None else batch
    cache = BatchTurboQuantKVCache(left_padding, bits=bits)
    keys, values = cache.update_and_fetch(*_rand_kv(batch, seq_len))
    return cache, keys, values


class TestSegmentedBatchStorage:
    def test_turboquant_dynamic_join_does_not_pad_short_row(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_SEGMENTED_BATCH_KV", "1")
        active, _, _ = _filled([0], 300)
        pending, _, _ = _filled([0], 20)

        active.extend(pending)

        assert active.is_segmented
        assert active.offset.tolist() == [300, 20]
        assert active.left_padding.tolist() == [0, 280]
        assert active.physical_token_capacity == 300 + 20

    def test_turboquant_segmented_append_filter_and_decode(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_SEGMENTED_BATCH_KV", "1")
        monkeypatch.delenv("MLX_VLM_TQ_BATCH_DECODE", raising=False)
        mx.random.seed(901)
        active, _, _ = _filled([0], 300)
        pending, _, _ = _filled([0], 20)
        active.extend(pending)

        queries = mx.random.normal((2, H, 1, D))
        keys, values = active.update_and_fetch(*_rand_kv(2, 1))
        output = active.packed_decode_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=SCALE,
            mask=None,
        )
        references = []
        for row, segment in enumerate(active._segments):
            dq_k, dq_v = segment.dequantize()
            references.append(
                mx.fast.scaled_dot_product_attention(
                    queries[row : row + 1],
                    dq_k.astype(queries.dtype),
                    dq_v.astype(queries.dtype),
                    scale=SCALE,
                    mask=None,
                )
            )
        reference = mx.concatenate(references, axis=0)
        mx.eval(output, reference)
        assert mx.allclose(output, reference, rtol=2e-2, atol=2e-2).item()

        active.filter(mx.array([1], dtype=mx.int32))
        assert active.offset.tolist() == [21]
        assert active.left_padding.tolist() == [0]
        keys, values = active.update_and_fetch(*_rand_kv(1, 1))
        assert len(keys) == len(values) == 1
        assert active.offset.tolist() == [22]

    def test_float_dynamic_join_does_not_pad_short_row(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_SEGMENTED_BATCH_KV", "1")
        active = BatchKVCache([0])
        pending = BatchKVCache([0])
        active.update_and_fetch(*_rand_kv(1, 300))
        pending.update_and_fetch(*_rand_kv(1, 20))

        active.extend(pending)

        assert active.is_segmented
        assert active.offset.tolist() == [300, 20]
        assert active.left_padding.tolist() == [0, 280]
        assert active.physical_token_capacity == 512 + 256

        queries = mx.random.normal((2, H, 1, D))
        keys, values = active.update_and_fetch(*_rand_kv(2, 1))
        output = active.segmented_attention(queries, scale=SCALE, mask=None)
        reference = mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    queries[row : row + 1],
                    keys[row],
                    values[row],
                    scale=SCALE,
                    mask=None,
                )
                for row in range(2)
            ],
            axis=0,
        )
        mx.eval(output, reference)
        assert mx.allclose(output, reference, atol=1e-5).item()

        active.filter(mx.array([0], dtype=mx.int32))
        keys, values = active.update_and_fetch(*_rand_kv(1, 1))
        output = scaled_dot_product_attention(
            queries[:1], keys, values, cache=active, scale=SCALE, mask=None
        )
        mx.eval(output)
        assert output.shape == (1, H, 1, D)

    def test_float_segmented_attention_slices_causal_mask_per_row(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_SEGMENTED_BATCH_KV", "1")
        mx.random.seed(902)
        active = BatchKVCache([0])
        pending = BatchKVCache([0])
        active.update_and_fetch(*_rand_kv(1, 11))
        pending.update_and_fetch(*_rand_kv(1, 5))
        active.extend(pending)

        queries = mx.random.normal((2, H, 3, D))
        mask = active.make_mask(3, return_array=True)
        keys, values = active.update_and_fetch(*_rand_kv(2, 3))
        output = active.segmented_attention(queries, scale=SCALE, mask=mask)
        references = []
        for row, segment in enumerate(active._segments):
            row_mask = mask[row : row + 1, ..., -segment.offset :]
            references.append(
                mx.fast.scaled_dot_product_attention(
                    queries[row : row + 1],
                    keys[row],
                    values[row],
                    scale=SCALE,
                    mask=row_mask,
                )
            )
        reference = mx.concatenate(references, axis=0)
        mx.eval(output, reference)

        assert output.shape == queries.shape
        assert mx.allclose(output, reference, atol=1e-5).item()

    def test_tiny_qwen_mixed_cache_b1_b2_b1(self, monkeypatch):
        from mlx_vlm.generate.ar import _extend_cache, _make_cache
        from mlx_vlm.models.qwen3_5.language import LanguageModel, TextConfig

        monkeypatch.setenv("MLX_VLM_SEGMENTED_BATCH_KV", "1")
        config = TextConfig(
            model_type="qwen3_5_text",
            hidden_size=16,
            intermediate_size=32,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
            num_hidden_layers=4,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=1,
            max_position_embeddings=128,
            tie_word_embeddings=True,
            head_dim=8,
            full_attention_interval=2,
            rope_parameters={
                "type": "default",
                "mrope_section": [1, 0, 0],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
            },
        )
        model = LanguageModel(config)
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=100,
            video_token_id=101,
            vision_start_token_id=102,
        )

        def make_cache():
            return _make_cache(
                model, [0], kv_bits=4, kv_quant_scheme="turboquant"
            )

        active, pending = make_cache(), make_cache()
        active_out = model(mx.array([[1, 2, 3, 4]]), cache=active)
        pending_out = model(mx.array([[5, 6]]), cache=pending)
        mx.eval(
            active_out.logits,
            pending_out.logits,
            [entry.state for entry in active],
            [entry.state for entry in pending],
        )

        joined = _extend_cache(active, pending)
        b2 = model(mx.array([[7], [8]]), cache=joined)
        mx.eval(b2.logits, [entry.state for entry in joined])
        assert b2.logits.shape == (2, 1, 32)
        assert mx.all(mx.isfinite(b2.logits)).item()

        for entry in joined:
            entry.filter(mx.array([0], dtype=mx.int32))
        b1 = model(mx.array([[9]]), cache=joined)
        mx.eval(b1.logits, [entry.state for entry in joined])
        assert b1.logits.shape == (1, 1, 32)
        assert mx.all(mx.isfinite(b1.logits)).item()


class TestSharedAttentionSurface:
    """Both caches expose the same attention API through the mixin."""

    def test_attention_states_ignores_batch_bookkeeping(self):
        # The batch cache's `state` is a 4-tuple; the mixin must not unpack it.
        cache, _, _ = _filled([0], 4)
        keys_state, values_state = cache._attention_states()
        assert keys_state is not None and values_state is not None


class TestFusedPathGuard:
    def test_single_cache_always_applies(self):
        assert _turboquant_attention_applies(TurboQuantKVCache(bits=BITS))

    def test_single_unpadded_row_applies(self):
        cache, _, _ = _filled([0], 8)
        assert _turboquant_attention_applies(cache)

    def test_aligned_multi_row_applies(self):
        cache, _, _ = _filled([0, 0], 8)
        assert _turboquant_attention_applies(cache)
        assert cache.packed_verify_eligible

    def test_ragged_multi_row_does_not_apply(self):
        cache, _, _ = _filled([1, 0], 8)
        assert not _turboquant_attention_applies(cache)
        assert cache.packed_verify_eligible

    def test_left_padded_row_does_not_apply(self):
        # Padded positions would otherwise be attended to as real tokens.
        cache, _, _ = _filled([3], 8)
        assert not _turboquant_attention_applies(cache)

    def test_cached_eligibility_tracks_batch_lifecycle(self):
        cache = BatchTurboQuantKVCache([0], bits=BITS)
        other = BatchTurboQuantKVCache([0], bits=BITS)
        assert cache.fused_attention_eligible

        cache.extend(other)
        assert cache.fused_attention_eligible
        assert cache.packed_verify_eligible
        assert _turboquant_attention_applies(cache)

        cache.filter(mx.array([0]))
        assert cache.fused_attention_eligible
        assert _turboquant_attention_applies(cache)

        cache.state = BatchTurboQuantKVCache([2], bits=BITS).state
        assert not cache.fused_attention_eligible
        assert cache.packed_verify_eligible
        assert not _turboquant_attention_applies(cache)

    def test_left_padded_rows_remain_packed_verify_eligible(self):
        cache, _, _ = _filled([3, 0], 16)
        assert not cache.fused_attention_eligible
        assert cache.packed_verify_eligible

    def test_contiguous_filter_preserves_packed_batch_view(self):
        cache, _, _ = _filled([7, 3, 0], 16)
        cache.filter(mx.array([0, 1], dtype=mx.int32))

        assert cache.keys.norms.shape[0] == 2
        assert cache.left_padding.tolist() == [4, 0]
        assert _contiguous_batch_slice(mx.array([0, 1])) == slice(0, 2)
        assert _contiguous_batch_slice(mx.array([0, 2])) is None


class TestNumericalEquivalence:
    """The fused path must agree with the dequantizing fallback."""

    def _reference(self, cache, queries, keys, values, mask=None):
        dq_k, dq_v = cache.dequantize(keys, values)
        return mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=SCALE,
            mask=mask,
        )

    @pytest.mark.parametrize(
        ("left_padding", "seq_len"),
        [([3, 0], 16), ([1100, 0], 2050)],
    )
    @pytest.mark.parametrize("simdgroups", [8, 16, 32])
    def test_ragged_packed_verify_matches_per_row_dequantized_attention(
        self, monkeypatch, left_padding, seq_len, simdgroups
    ):
        monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE", "1")
        monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE_SIMDGROUPS", str(simdgroups))
        mx.random.seed(121)
        cache, keys, values = _filled(left_padding, seq_len)
        queries = mx.random.normal((2, H, 4, D))
        mask = create_causal_mask(
            4,
            offset=seq_len - 4,
            left_padding=mx.array(left_padding),
        )

        output = cache.packed_verify_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=SCALE,
            mask=mask,
        )
        references = []
        for row in range(2):
            single = cache.extract(row)
            dq_k, dq_v = single.dequantize()
            references.append(
                mx.fast.scaled_dot_product_attention(
                    queries[row : row + 1],
                    dq_k.astype(queries.dtype),
                    dq_v.astype(queries.dtype),
                    scale=SCALE,
                    mask="causal",
                )
            )
        reference = mx.concatenate(references, axis=0)
        mx.eval(output, reference)

        assert bool(mx.allclose(output, reference, rtol=2e-2, atol=2e-2).item())

    def test_ragged_packed_verify_after_long_dynamic_extend(self, monkeypatch):
        """Long packed rows admitted at different times stay kernel-safe."""
        monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE", "1")
        mx.random.seed(122)
        active, _, _ = _filled([0], 4116)
        pending, _, _ = _filled([0], 1044)
        active.extend(pending)
        assert active._eval_next_append

        queries = mx.random.normal((2, H, 3, D))
        keys, values = active.update_and_fetch(*_rand_kv(2, 3))
        assert not active._eval_next_append
        mask = create_causal_mask(
            3,
            offset=active._idx - 3,
            left_padding=active.left_padding,
        )
        output = active.packed_verify_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=SCALE,
            mask=mask,
        )

        mx.eval(output)
        assert output.shape == queries.shape

    def test_three_row_packed_verify_matches_per_row_attention(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE", "1")
        mx.random.seed(124)
        cache, keys, values = _filled([7, 3, 0], 24)
        queries = mx.random.normal((3, H, 3, D))
        mask = create_causal_mask(
            3,
            offset=21,
            left_padding=cache.left_padding,
        )

        output = cache.packed_verify_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=SCALE,
            mask=mask,
        )
        references = []
        for row in range(3):
            single = cache.extract(row)
            dq_k, dq_v = single.dequantize()
            references.append(
                mx.fast.scaled_dot_product_attention(
                    queries[row : row + 1],
                    dq_k.astype(queries.dtype),
                    dq_v.astype(queries.dtype),
                    scale=SCALE,
                    mask="causal",
                )
            )
        reference = mx.concatenate(references, axis=0)
        mx.eval(output, reference)

        assert output.shape == queries.shape
        assert bool(mx.allclose(output, reference, rtol=2e-2, atol=2e-2).item())

    @pytest.mark.parametrize("seq_len", [16, 300])
    def test_decode_matches_fallback(self, seq_len):
        cache, keys, values = _filled([0], seq_len)
        queries = mx.random.normal((1, H, 1, D))

        fused = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        reference = self._reference(cache, queries, keys, values)
        mx.eval(fused, reference)

        assert fused.shape == reference.shape
        # Both paths read the same quantized state, so they differ only by
        # kernel arithmetic order.
        assert mx.allclose(fused, reference, atol=2e-2).item()

    def test_aligned_multi_row_uses_fused_decode(self, monkeypatch):
        cache, keys, values = _filled([0, 0], 12)
        queries = mx.random.normal((2, H, 1, D))

        def fail_dequantize(*args, **kwargs):
            raise AssertionError("aligned batch decode must not dequantize K/V")

        monkeypatch.setattr(cache, "dequantize_for_attention", fail_dequantize)
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(out)
        assert out.shape == (2, H, 1, D)

    def test_aligned_dynamic_join_and_filter_keep_fused_decode(self, monkeypatch):
        active, _, _ = _filled([0], 12)
        pending, _, _ = _filled([0], 12)
        active.extend(pending)
        assert active._eval_next_append
        assert _turboquant_attention_applies(active)

        keys, values = active.update_and_fetch(*_rand_kv(2, 1))
        queries = mx.random.normal((2, H, 1, D))

        def fail_dequantize(*args, **kwargs):
            raise AssertionError("aligned dynamic batch must not dequantize K/V")

        monkeypatch.setattr(active, "dequantize_for_attention", fail_dequantize)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=active, scale=SCALE, mask=None
        )
        mx.eval(output)
        assert output.shape == queries.shape

        active.filter(mx.array([0], dtype=mx.int32))
        keys, values = active.update_and_fetch(*_rand_kv(1, 1))
        queries = queries[:1]
        output = scaled_dot_product_attention(
            queries, keys, values, cache=active, scale=SCALE, mask=None
        )
        mx.eval(output)
        assert output.shape == queries.shape

    @pytest.mark.parametrize(
        ("left_padding", "seq_len"),
        [([3, 0], 16), ([2048, 0], 2050)],
    )
    def test_ragged_batch_decode_stays_packed_without_qtile(
        self, monkeypatch, left_padding, seq_len
    ):
        monkeypatch.setenv("MLX_VLM_TQ_BATCH_DECODE", "1")
        monkeypatch.delenv("MLX_VLM_TQ_BATCH_DECODE_QTILE", raising=False)
        monkeypatch.delenv("MLX_VLM_TQ_MTP_QTILE", raising=False)
        mx.random.seed(123)
        cache, keys, values = _filled(left_padding, seq_len)
        queries = mx.random.normal((2, H, 1, D))

        references = []
        for row in range(2):
            single = cache.extract(row)
            dq_k, dq_v = single.dequantize()
            references.append(
                mx.fast.scaled_dot_product_attention(
                    queries[row : row + 1],
                    dq_k.astype(queries.dtype),
                    dq_v.astype(queries.dtype),
                    scale=SCALE,
                    mask=None,
                )
            )
        reference = mx.concatenate(references, axis=0)

        def fail_dequantize(*args, **kwargs):
            raise AssertionError("ragged packed decode must not dequantize K/V")

        monkeypatch.setattr(cache, "dequantize_for_attention", fail_dequantize)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(output, reference)

        assert output.shape == queries.shape
        assert bool(mx.allclose(output, reference, rtol=2e-2, atol=2e-2).item())

    def test_mtp_qtile_flag_does_not_enable_ragged_ar_decode(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TQ_MTP_QTILE", "1")
        monkeypatch.delenv("MLX_VLM_TQ_BATCH_DECODE", raising=False)
        cache, keys, values = _filled([3, 0], 16)
        queries = mx.random.normal((2, H, 1, D))

        assert (
            cache.packed_decode_attention(
                queries,
                keys_state=keys,
                values_state=values,
                scale=SCALE,
                mask=None,
            )
            is None
        )

    def test_left_padded_matches_fallback(self):
        cache, keys, values = _filled([2], 10)
        queries = mx.random.normal((1, H, 1, D))
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        reference = self._reference(cache, queries, keys, values)
        mx.eval(out, reference)
        assert mx.allclose(out, reference, atol=2e-2).item()

    @pytest.mark.skipif(not hasattr(mx, "metal"), reason="requires Metal kernels")
    def test_fused_dequantize_matches_current_path(self, monkeypatch):
        cache, keys, values = _filled([0], 300)
        monkeypatch.delenv("MLX_VLM_TQ_FUSED_DEQUANT", raising=False)
        reference_k, reference_v = cache.dequantize(keys, values)
        monkeypatch.setenv("MLX_VLM_TQ_FUSED_DEQUANT", "1")
        fused_k, fused_v = cache.dequantize_for_attention(keys, values)
        mx.eval(reference_k, reference_v, fused_k, fused_v)

        assert fused_k.dtype == mx.float16
        assert fused_v.dtype == mx.float16
        assert mx.allclose(fused_k, reference_k, atol=2e-3).item()
        assert mx.allclose(fused_v, reference_v, atol=2e-3).item()

    def test_batch_invariant_mode_uses_reference_dequantize(self, monkeypatch):
        cache, keys, values = _filled([0], 300)
        monkeypatch.setenv("MLX_VLM_TQ_FUSED_DEQUANT", "1")
        monkeypatch.setenv("MLX_VLM_BATCH_INVARIANT", "1")

        def fail_fused(*args, **kwargs):
            raise AssertionError("approximate fused dequantize is not invariant")

        monkeypatch.setattr(cache, "_fused_mse_dequantize_pair", fail_fused)
        reference_k, reference_v = cache.dequantize(keys, values)
        actual_k, actual_v = cache.dequantize_for_attention(keys, values)
        mx.eval(reference_k, reference_v, actual_k, actual_v)

        assert bool(mx.array_equal(actual_k, reference_k).item())
        assert bool(mx.array_equal(actual_v, reference_v).item())

    @pytest.mark.skipif(not hasattr(mx, "metal"), reason="requires Metal kernels")
    def test_attention_fallback_uses_fused_dequantize(self, monkeypatch):
        cache, keys, values = _filled([0, 0], 300)
        queries = mx.random.normal((2, H, 3, D))
        monkeypatch.setenv("MLX_VLM_TQ_FUSED_DEQUANT", "1")

        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        reference = self._reference(cache, queries, keys, values)
        mx.eval(out, reference)

        assert mx.allclose(out, reference, atol=2e-2).item()


class TestShapesBeyondTheDefaultLayout:
    """The fused path has to survive head geometries other than the default.

    Models differ in head_dim and in how many query heads share a KV head, so
    exercise a couple of combinations rather than only 4 heads at 64 dims.
    """

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("q_per_kv", [1, 4, 6])
    def test_decode_matches_fallback(self, head_dim, q_per_kv):
        kv_heads, seq_len = 4, 96
        scale = head_dim**-0.5
        cache = BatchTurboQuantKVCache([0], bits=BITS)
        keys, values = cache.update_and_fetch(
            mx.random.normal((1, kv_heads, seq_len, head_dim)),
            mx.random.normal((1, kv_heads, seq_len, head_dim)),
        )
        queries = mx.random.normal((1, kv_heads * q_per_kv, 1, head_dim))

        fused = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=None
        )
        dq_k, dq_v = cache.dequantize(keys, values)
        reference = mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=scale,
            mask=None,
        )
        mx.eval(fused, reference)
        assert fused.shape == reference.shape
        assert mx.allclose(fused, reference, atol=2e-2).item()


class TestAttentionSinks:
    """Sinks must be applied, not dropped and not rejected.

    The fused kernels carry no sink term, so a request with sinks has to fall
    through to the dequantizing path, which can pass them to MLX.
    """

    def _with_sinks(self, cache, keys, values, queries, sinks):
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None, sinks=sinks
        )

    def test_sinks_are_applied(self):
        cache, keys, values = _filled([0], 64)
        queries = mx.random.normal((1, H, 1, D))
        sinks = mx.random.normal((H,))

        out = self._with_sinks(cache, keys, values, queries, sinks)
        dq_k, dq_v = cache.dequantize(keys, values)
        reference = mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=SCALE,
            mask=None,
            sinks=sinks,
        )
        mx.eval(out, reference)
        assert mx.allclose(out, reference, atol=2e-2).item()

    def test_sinks_change_the_result(self):
        # Guards against silently discarding them: the batch cache used to
        # drop sinks on the floor and return the no-sink answer.
        cache, keys, values = _filled([0], 64)
        queries = mx.random.normal((1, H, 1, D))
        sinks = mx.full((H,), 5.0)

        with_sinks = self._with_sinks(cache, keys, values, queries, sinks)
        without = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(with_sinks, without)
        assert not mx.allclose(with_sinks, without, atol=1e-3).item()


class TestDecodeMemoryIsFlat:
    """Regression guard for the bug this change fixes.

    The dequantizing fallback materialised the whole KV cache as float32 on
    every step, so peak memory scaled with the context length. The fused path
    reads the quantized state in place.
    """

    def _peak_delta_for(self, seq_len):
        cache, keys, values = _filled([0], seq_len)
        queries = mx.random.normal((1, H, 1, D))
        mx.eval(cache.keys, cache.values, queries)
        mx.clear_cache()

        before = mx.get_peak_memory()
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(out)
        return mx.get_peak_memory() - before

    def test_peak_does_not_scale_with_context(self):
        short = self._peak_delta_for(256)
        long = self._peak_delta_for(4096)
        # 16x the context. Dequantizing would grow the step's peak roughly in
        # step with it; the fused kernels keep it bounded.
        assert long <= max(short, 1 << 20) * 4
