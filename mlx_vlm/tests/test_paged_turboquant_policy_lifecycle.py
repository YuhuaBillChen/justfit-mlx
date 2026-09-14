"""Execution policy must survive request creation, joins, and cancellation."""

import pytest

from mlx_vlm.paged_turboquant_cache import PagedBatchTurboQuantKVCache
from mlx_vlm.paged_turboquant_config import PagedTurboQuantConfig
from mlx_vlm.paged_turboquant_pool import (
    PagedTurboQuantLayerSpec,
    PagedTurboQuantPoolRegistry,
)


def test_registry_snapshots_policy_before_independent_requests(monkeypatch):
    monkeypatch.setenv("MLX_VLM_PAGED_PREFILL_IMPL", "direct_inverse")
    registry = PagedTurboQuantPoolRegistry({0: PagedTurboQuantLayerSpec(4, 2)})
    first = registry.new_cache(0)
    monkeypatch.setenv("MLX_VLM_PAGED_PREFILL_IMPL", "compatibility")
    second = registry.new_cache(0)
    fork = first.new_empty()
    assert first.config is second.config is fork.config is registry.config
    assert first.config.prefill_impl == "direct_inverse"
    first.extend(second)
    first.extend(fork)
    first.filter([1])
    assert first.config is registry.config
    registry.release()


def test_incompatible_policy_join_is_rejected_before_ownership_moves():
    registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(4, 2)}, config=PagedTurboQuantConfig()
    )
    first = registry.new_cache(0)
    second = PagedBatchTurboQuantKVCache(
        [0], bits=4, storage=first.storage,
        config=PagedTurboQuantConfig(mtp_qtile=True),
    )
    before = (first.sequence_lengths, second.sequence_lengths)
    with pytest.raises(ValueError, match="execution policies"):
        first.extend(second)
    assert (first.sequence_lengths, second.sequence_lengths) == before
    assert not first._released and not second._released
    second.release()
    registry.release()


def test_explicit_policy_overrides_environment_and_is_read_only(monkeypatch):
    monkeypatch.setenv("MLX_VLM_PAGED_PREFILL_IMPL", "invalid")
    policy = PagedTurboQuantConfig()
    registry = PagedTurboQuantPoolRegistry(
        {0: PagedTurboQuantLayerSpec(4, 2)}, config=policy
    )
    cache = registry.new_cache(0)
    assert cache.config is policy
    with pytest.raises(AttributeError):
        cache.config = PagedTurboQuantConfig(mtp_qtile=True)
    with pytest.raises(AttributeError):
        registry.config = PagedTurboQuantConfig(mtp_qtile=True)
    registry.release()
