"""Borrowed exact checkpoints must not materialize the paged cache first."""
import mlx.core as mx
import pytest

from mlx_vlm.apc import snapshot_prompt_cache_row
from mlx_vlm.paged_turboquant_cache import (
    PagedBatchTurboQuantKVCache,
    PagedTurboQuantAPCView,
)
from mlx_vlm.turboquant import TurboQuantKVCache


def make_cache():
    cache = PagedBatchTurboQuantKVCache([0], bits=4, capacity_pages=4)
    k = mx.ones((1, 4, 17, 256), dtype=mx.bfloat16)
    cache.update_and_fetch(k, k)
    mx.eval(cache.state)
    return cache


def test_borrowed_snapshot_never_materializes_paged_row(monkeypatch):
    cache = make_cache()
    def forbidden(*args, **kwargs):
        pytest.fail('borrowed snapshot materialized the full paged row')
    monkeypatch.setattr(cache, 'materialize', forbidden)
    snapshot = snapshot_prompt_cache_row([cache], clone=False, detach=False)
    assert isinstance(snapshot[0], PagedTurboQuantAPCView)
    assert snapshot[0].cache is cache
    assert snapshot[0].sequence_length == 17


def test_default_snapshot_remains_detached():
    cache = make_cache()
    snapshot = snapshot_prompt_cache_row([cache])
    assert isinstance(snapshot[0], TurboQuantKVCache)
    assert snapshot[0].offset == 17
    cache.update_and_fetch(mx.ones((1, 4, 1, 256)), mx.ones((1, 4, 1, 256)))
    assert snapshot[0].offset == 17


def test_coordinator_direct_disk_borrows_and_roundtrips(tmp_path, monkeypatch):
    from mlx_vlm.apc import APCManager, DiskBlockStore
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.models.cache import KVCache, ArraysCache
    class Hybrid:
        def make_cache(self):
            return [KVCache(), ArraysCache(2)]
    monkeypatch.setenv('APC_CHECKPOINT_ENTRIES', '0')
    monkeypatch.setenv('APC_EXACT_DIRECT_DISK_WRITE', '1')
    cache = make_cache()
    expected = cache.extract(0)
    mx.eval(expected.state)
    def forbidden(*args, **kwargs):
        pytest.fail('coordinator materialized borrowed KV before saving')
    monkeypatch.setattr(cache, 'materialize', forbidden)
    disk = DiskBlockStore(tmp_path, namespace='borrowed-regression')
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    try:
        assert APCCoordinator(manager, Hybrid()).store_checkpoint(list(range(17)), [cache])
        warm, count = manager.lookup_exact_cache(list(range(17)) + [99])
        assert count == 17
        for a, b in zip(warm[0].state, expected.state):
            assert mx.array_equal(a.norms, b.norms).item()
            assert mx.array_equal(a.indices, b.indices).item()
    finally:
        manager.close()
