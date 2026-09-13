"""Generator-scoped paged TurboQuant storage registry tests."""

import mlx.core as mx
import pytest

from mlx_vlm.paged_turboquant_kernel import PAGED_TURBOQUANT_PAGE_SIZE
from mlx_vlm.paged_turboquant_pool import (
    PagedTurboQuantLayerSpec,
    PagedTurboQuantPoolRegistry,
)


H_KV = 2
D = 256
PAGE = PAGED_TURBOQUANT_PAGE_SIZE


def _kv(length):
    return (
        mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16),
        mx.random.normal((1, H_KV, length, D)).astype(mx.bfloat16),
    )


def _registry(capacity_pages=8):
    return PagedTurboQuantPoolRegistry(
        {
            0: PagedTurboQuantLayerSpec(capacity_pages, H_KV),
            (3, "attention"): PagedTurboQuantLayerSpec(capacity_pages, H_KV),
        }
    )


def test_registry_eagerly_binds_independent_prompt_caches_to_stable_storage():
    registry = _registry()
    storage = registry.storage_for((3, "attention"))
    payload_ids = tuple(
        id(array) for state in (storage.keys, storage.values) for array in state
    )

    first = registry.new_cache((3, "attention"))
    second = registry.new_cache((3, "attention"))

    assert first.storage is second.storage is storage
    assert first.key_codec is second.key_codec
    assert first.value_codec is second.value_codec
    assert (
        tuple(id(array) for state in (storage.keys, storage.values) for array in state)
        == payload_ids
    )

    first.update_and_fetch(*_kv(PAGE + 3))
    second.update_and_fetch(*_kv(17))
    first_pages = first._rows.rows[0].page_ids
    second_pages = second._rows.rows[0].page_ids
    first.extend(second)

    # Admission changes only row metadata. The physical tensors and both
    # requests' page mappings remain exactly where prefill wrote them.
    assert first.storage is storage
    assert (
        tuple(id(array) for state in (storage.keys, storage.values) for array in state)
        == payload_ids
    )
    assert tuple(row.page_ids for row in first._rows.rows) == (
        first_pages,
        second_pages,
    )
    assert first.sequence_lengths == (PAGE + 3, 17)


def test_cache_sets_preserve_opaque_model_leaf_keys_and_share_per_leaf_only():
    registry = _registry()
    first_prompt = registry.new_cache_set()
    second_prompt = registry.new_cache_set(reversed(registry.leaf_keys))

    assert tuple(first_prompt) == (0, (3, "attention"))
    assert tuple(second_prompt) == ((3, "attention"), 0)
    for leaf_key in registry.leaf_keys:
        assert first_prompt[leaf_key].storage is registry.storage_for(leaf_key)
        assert second_prompt[leaf_key].storage is registry.storage_for(leaf_key)
    assert first_prompt[0].storage is not first_prompt[(3, "attention")].storage

    with pytest.raises(KeyError, match="unknown paged cache leaf"):
        registry.new_cache((99, "attention"))
    before = registry.stats().live_facades
    with pytest.raises(KeyError, match="unknown paged cache leaf"):
        registry.new_cache_set((0, "missing"))
    assert registry.stats().live_facades == before


def test_registry_reports_aggregate_capacity_usage_and_releases_every_row():
    registry = _registry(capacity_pages=4)
    first = registry.new_cache_set()
    second = registry.new_cache_set()
    for caches, length in ((first, PAGE + 1), (second, 1)):
        for cache in caches.values():
            cache.update_and_fetch(*_kv(length))

    stats = registry.stats()
    assert stats.capacity_layer_pages == 8
    assert stats.used_layer_pages == 6
    assert stats.free_layer_pages == 2
    assert stats.used_token_slots == 6 * PAGE
    assert stats.capacity_token_slots == 8 * PAGE
    assert stats.pool_nbytes > 0
    assert stats.live_facades == 4
    assert not stats.closed

    final = registry.release()
    assert final.used_layer_pages == 0
    assert final.free_layer_pages == 8
    assert final.live_facades == 0
    assert final.closed
    with pytest.raises(RuntimeError, match="released"):
        registry.new_cache(0)


def test_registry_enforces_uniform_capacity_and_exact_kernel_geometry():
    with pytest.raises(ValueError, match="same capacity_pages"):
        PagedTurboQuantPoolRegistry(
            {
                0: PagedTurboQuantLayerSpec(4, H_KV),
                1: PagedTurboQuantLayerSpec(5, H_KV),
            }
        )
    with pytest.raises(ValueError, match="page_size=256"):
        PagedTurboQuantLayerSpec(4, H_KV, page_size=128)
    with pytest.raises(ValueError, match="head_dim=256"):
        PagedTurboQuantLayerSpec(4, H_KV, head_dim=128)
    with pytest.raises(ValueError, match="Q4"):
        PagedTurboQuantLayerSpec(4, H_KV, bits=8)
    with pytest.raises(ValueError, match="float16 norms"):
        PagedTurboQuantLayerSpec(4, H_KV, norm_dtype=mx.float32)
    with pytest.raises(ValueError, match="uint32 packed"):
        PagedTurboQuantLayerSpec(4, H_KV, index_dtype=mx.uint16)


def test_registry_restores_packed_apc_row_directly_into_existing_page_pool():
    from mlx_vlm.turboquant import TurboQuantKVCache

    registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(4, H_KV)}
    )
    source = TurboQuantKVCache(bits=4)
    source.update_and_fetch(*_kv(PAGE + 7))
    mx.eval(source.state)

    restored = registry.restore_cache_list([source])

    assert len(restored) == 1
    assert restored[0].storage is registry.storage_for(0)
    assert restored[0].sequence_lengths == (PAGE + 7,)
    restored_keys, restored_values = restored[0].materialize(0)
    assert bool(mx.array_equal(restored_keys.norms, source.keys.norms).item())
    assert bool(mx.array_equal(restored_keys.indices, source.keys.indices).item())
    assert bool(mx.array_equal(restored_values.norms, source.values.norms).item())
    assert bool(mx.array_equal(restored_values.indices, source.values.indices).item())


def test_registry_marks_restored_float_tail_for_ragged_dynamic_join():
    from mlx_vlm.models.cache import BatchKVCache
    from mlx_vlm.turboquant import TurboQuantKVCache

    registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(16, H_KV)}
    )
    quantized = TurboQuantKVCache(bits=4)
    quantized.update_and_fetch(*_kv(PAGE + 7))
    float_tail = BatchKVCache([0])
    tail_state = mx.zeros((1, H_KV, 1024, D), dtype=mx.float16)
    float_tail.update_and_fetch(tail_state, tail_state)

    restored = registry.restore_cache_list([quantized, float_tail])

    assert restored[1].segment_on_extend
    peer = BatchKVCache([0])
    peer.segment_on_extend = True
    peer_state = mx.zeros((1, H_KV, 3371, D), dtype=mx.float16)
    peer.update_and_fetch(peer_state, peer_state)
    restored[1].extend(peer)
    assert restored[1].is_segmented
    assert [segment.offset for segment in restored[1]._segments] == [1024, 3371]


def test_exact_disk_restore_stays_lazy_until_streamed_into_page_pool(
    tmp_path, monkeypatch
):
    from mlx_vlm import apc
    from mlx_vlm.apc import DiskBlockStore, make_warm_batch_exact_cache_multi

    layer_specs = {0: PagedTurboQuantLayerSpec(4, H_KV)}
    source_registry = PagedTurboQuantPoolRegistry(layer_specs)
    source = source_registry.new_cache(0)
    source.update_and_fetch(*_kv(PAGE + 7))
    source.synchronize()
    expected_keys, expected_values = source.materialize(0)

    disk = DiskBlockStore(tmp_path, namespace="paged-direct-restore")
    cache_hash = 123
    token_ids = tuple(range(PAGE + 7))
    disk.save_exact_cache_sync(cache_hash, token_ids, 0, [source.extract_view(0)])

    reads = 0
    original_read = apc._read_safetensors_tensor

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(apc, "_read_safetensors_tensor", counted_read)
    loaded = disk.load_exact_cache(cache_hash, defer_paged_q4=True)
    assert loaded is not None
    stored_tokens, stored_extra_hash, row = loaded
    assert stored_tokens == token_ids
    assert stored_extra_hash == 0
    assert reads == 0

    warm, prefix_len = make_warm_batch_exact_cache_multi(
        [row], [len(token_ids)], consume_sources=True
    )
    assert prefix_len == len(token_ids)
    assert warm is not None
    assert reads == 0

    target_registry = PagedTurboQuantPoolRegistry(layer_specs)
    restored = target_registry.restore_cache_list(warm)
    assert reads == 4
    restored_keys, restored_values = restored[0].materialize(0)
    assert bool(mx.array_equal(restored_keys.norms, expected_keys.norms).item())
    assert bool(mx.array_equal(restored_keys.indices, expected_keys.indices).item())
    assert bool(mx.array_equal(restored_values.norms, expected_values.norms).item())
    assert bool(mx.array_equal(restored_values.indices, expected_values.indices).item())

    # A non-paged consumer (for example the v14 singleton fork sharing this
    # APC namespace) retains the legacy contiguous restore behavior.
    from mlx_vlm.turboquant import TurboQuantKVCache

    reads_before_compat = reads
    compatible = disk.load_exact_cache(cache_hash)
    assert compatible is not None
    assert isinstance(compatible[2][0], TurboQuantKVCache)
    assert reads == reads_before_compat + 4
    disk.close()


def test_streamed_page_restore_releases_reservation_after_read_failure():
    source_registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(2, H_KV)}
    )
    source = source_registry.new_cache(0)
    source.update_and_fetch(*_kv(PAGE))
    source.synchronize()
    page_id = source._rows.rows[0].page_ids[0]
    storage = source.storage

    def failing_runs():
        yield (
            storage.keys.norms[page_id : page_id + 1],
            storage.keys.indices[page_id : page_id + 1],
            storage.values.norms[page_id : page_id + 1],
            storage.values.indices[page_id : page_id + 1],
        )
        raise OSError("synthetic APC read failure")

    target_registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(2, H_KV)}
    )
    target = target_registry.new_cache(0)
    with pytest.raises(OSError, match="synthetic APC read failure"):
        target.restore_packed_page_runs(failing_runs(), PAGE + 1)

    assert target.sequence_lengths == (0,)
    assert target_registry.stats().used_layer_pages == 0


def test_cross_layer_reservation_exhaustion_is_atomic_before_forward():
    registry = _registry(capacity_pages=1)
    caches = registry.new_cache_set()
    blocker = registry.new_cache(0)
    blocker.update_and_fetch(*_kv(1))
    before = registry.stats()

    with pytest.raises(Exception, match="only 0 free"):
        with registry.reserve_append(caches, PAGE + 1):
            pytest.fail("forward must not start after reservation failure")

    after = registry.stats()
    assert after.used_layer_pages == before.used_layer_pages
    assert all(cache.sequence_lengths == (0,) for cache in caches.values())


def test_cross_layer_forward_exception_rolls_back_every_layer():
    registry = _registry(capacity_pages=4)
    caches = registry.new_cache_set()

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        with registry.reserve_append(caches, PAGE + 1):
            caches[0].update_and_fetch(*_kv(PAGE + 1))
            raise RuntimeError("synthetic forward failure")

    assert all(cache.sequence_lengths == (0,) for cache in caches.values())
    assert registry.stats().used_layer_pages == 0

    # The exact pages are reusable immediately; no layer retained a partial
    # reservation after the failed forward.
    with registry.reserve_append(caches, 1):
        for cache in caches.values():
            cache.update_and_fetch(*_kv(1))
    assert all(cache.sequence_lengths == (1,) for cache in caches.values())
    assert registry.stats().used_layer_pages == 2


def test_cross_layer_reservation_requires_every_layer_to_consume_append():
    registry = _registry(capacity_pages=2)
    caches = registry.new_cache_set()

    with pytest.raises(RuntimeError, match="did not consume"):
        with registry.reserve_append(caches, 1):
            caches[0].update_and_fetch(*_kv(1))

    assert all(cache.sequence_lengths == (0,) for cache in caches.values())
    assert registry.stats().used_layer_pages == 0
