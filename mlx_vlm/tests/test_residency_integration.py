"""Cohort retirement must precede component/dependency reclamation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import weakref

import pytest
import mlx.core as mx

from mlx_vlm.generate.ar import (
    BatchGenerator,
    PromptProcessingBatch,
    SpeculativeGenerationBatch,
)
from mlx_vlm.server.component_residency import ComponentResidencyManager
from mlx_vlm.server.draft_lifecycle import LazyDrafter
from mlx_vlm.server.generation import ResponseGenerator


def managed_batch(monkeypatch):
    monkeypatch.setattr("mlx_vlm.server.draft_lifecycle.mx.clear_cache", lambda: None)
    manager = ComponentResidencyManager()
    head = Mock()
    embedding = Mock()
    manager.register("lm_head", head)
    manager.register("input_embedding", embedding, retain_on_release=True)
    lazy = LazyDrafter(
        path="unused", kind="mtp", config=None,
        loader=lambda *args: (Mock(spec=[]), "mtp"), validator=lambda *args: None,
        target_model=None,
    )
    manager.register("mtp_drafter", lazy, dependencies=("input_embedding", "lm_head"))
    batch = object.__new__(SpeculativeGenerationBatch)
    batch.model = SimpleNamespace(phase_residency_manager=manager)
    batch.prompt_cache = []
    batch._draft_owner = lazy
    batch._drafter_lease_owner = object()
    manager.acquire("mtp_drafter", batch._drafter_lease_owner)
    batch.draft_model = lazy.materialize()
    batch._rounds_iter = None
    return batch, manager, head, embedding, lazy


def test_drafter_alias_dropped_before_last_owner_unload(monkeypatch):
    batch, manager, head, embedding, lazy = managed_batch(monkeypatch)
    loaded_ref = weakref.ref(batch.draft_model)
    manager.acquire("lm_head", "generation")
    rounds = Mock()
    def assert_loaded():
        assert lazy.loaded
    rounds.close.side_effect = assert_loaded
    batch._rounds_iter = rounds
    batch.release_drafter()
    assert loaded_ref() is None
    assert not lazy.loaded
    assert batch.draft_model is lazy
    rounds.close.assert_called_once_with()
    assert manager.owners("lm_head") == frozenset({"generation"})
    head.unload.assert_not_called()
    embedding.unload.assert_not_called()


def test_retired_batch_cannot_release_successor_drafter(monkeypatch):
    batch, manager, _, _, lazy = managed_batch(monkeypatch)
    batch.release_drafter()
    successor = object()
    manager.acquire("mtp_drafter", successor)
    batch.release_drafter()
    assert manager.owners("mtp_drafter") == frozenset({successor})
    assert lazy.loaded
    manager.release("mtp_drafter", successor)


def test_round_close_failure_keeps_drafter_pinned(monkeypatch):
    batch, manager, _, _, lazy = managed_batch(monkeypatch)
    batch._rounds_iter = Mock()
    batch._rounds_iter.close.side_effect = RuntimeError("round not retired")
    with pytest.raises(RuntimeError, match="round not retired"):
        batch.release_drafter()
    assert manager.owners("mtp_drafter")
    assert lazy.loaded
    batch._rounds_iter.close.side_effect = None
    batch.release_drafter()


def test_server_idle_cleanup_does_not_bypass_active_drafter_lease(monkeypatch):
    batch, manager, _, _, lazy = managed_batch(monkeypatch)
    server = object.__new__(ResponseGenerator)
    server.component_residency = manager
    server.draft_model = lazy
    server._unload_deferred_drafter()
    assert lazy.loaded
    batch.release_drafter()
    server._unload_deferred_drafter()
    assert not lazy.loaded


def test_close_releases_drafter_before_head_and_keeps_embedding(monkeypatch):
    batch, manager, head, embedding, lazy = managed_batch(monkeypatch)
    manager.acquire("lm_head", "generation")
    manager.acquire("input_embedding", "generation")
    generator = object.__new__(BatchGenerator)
    generator.model = batch.model
    generator._generation_batch = batch
    def assert_detached():
        assert not lazy.loaded
        assert batch.draft_model is lazy
    head.unload.side_effect = assert_detached
    generator.close()
    head.unload.assert_called_once_with()
    embedding.unload.assert_not_called()
    assert not manager.owners("input_embedding")
    manager.acquire("lm_head", "generation")
    manager.acquire("input_embedding", "generation")
    generator.close()
    assert manager.owners("lm_head") == frozenset({"generation"})
    assert manager.owners("input_embedding") == frozenset({"generation"})
    manager.release("lm_head", "generation")
    manager.release("input_embedding", "generation")


def test_decode_sync_protects_embedding_until_last_row(monkeypatch):
    batch, manager, _, embedding, _ = managed_batch(monkeypatch)
    batch.release_drafter()
    generator = object.__new__(BatchGenerator)
    generator.model = batch.model
    generator._generation_batch = [object(), object()]
    generator._sync_generation_head_residency()
    assert not manager.unload_if_idle("input_embedding")
    generator._generation_batch = [object()]
    generator._sync_generation_head_residency()
    assert not manager.unload_if_idle("input_embedding")
    generator._generation_batch = []
    generator._sync_generation_head_residency()
    assert not manager.owners("input_embedding")
    embedding.unload.assert_not_called()
    generator.close()


@pytest.mark.parametrize("decode_active", [False, True])
def test_warm_spill_uses_embedding_lease_guard(monkeypatch, decode_active):
    manager = ComponentResidencyManager()
    embedding = Mock()
    manager.register("input_embedding", embedding, retain_on_release=True)
    if decode_active:
        manager.acquire("input_embedding", "generation")
    model = MagicMock()
    model.phase_residency_manager = manager
    model.prefill_embedding_phase_swap = embedding
    model.prefill_head_phase_swap = None
    model.supports_skip_logits = True
    batch = PromptProcessingBatch(
        model=model, uids=[1], input_ids=[[1, 2, 3]], max_tokens=[1],
        inputs_embeds=mx.ones((1, 3, 4)), prompt_kwargs={},
        prefill_step_size=2, warm_cache=[SimpleNamespace(state=mx.array([1]))],
    )
    # Constructor spill has finished and temporary providers have retired.
    batch._embedding_phase_swap_pending = True
    monkeypatch.setattr(mx, "async_eval", Mock())
    assert batch.prompt_step() == 2
    assert embedding.unload.call_count == (0 if decode_active else 1)


def test_new_request_restores_embedding_after_cancelled_spill():
    manager = ComponentResidencyManager()
    state = {"loaded": False}
    embedding = Mock()
    embedding.load.side_effect = lambda: state.update(loaded=True)
    manager.register("input_embedding", embedding, retain_on_release=True)
    server = object.__new__(ResponseGenerator)
    server.component_residency = manager
    server.vision_cache = None
    server.model = Mock()
    def get_embeddings(*args, **kwargs):
        assert state["loaded"]
        return SimpleNamespace(to_dict=lambda: {}, input_embedding_provider=None)
    server.model.get_input_embeddings.side_effect = get_embeddings
    server._gpu_embed({"input_ids": mx.array([[1]])})
    embedding.load.assert_called_once_with()
