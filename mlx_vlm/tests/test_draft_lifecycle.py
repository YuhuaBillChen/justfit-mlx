from types import SimpleNamespace

import pytest

from mlx_vlm.generate.ar import SpeculativeGenerationBatch
from mlx_vlm.server.draft_lifecycle import LazyDrafter


def test_lazy_drafter_loads_once_and_can_be_released(monkeypatch):
    loaded = []
    validated = []
    clear_calls = []
    target = object()

    def loader(path, kind):
        model = SimpleNamespace(
            generation=len(loaded) + 1,
            speculative_total_rounds=7,
        )
        loaded.append((path, kind, model))
        return model, kind

    def validator(target_model, model, kind):
        validated.append((target_model, model, kind))

    monkeypatch.setattr(
        "mlx_vlm.server.draft_lifecycle.mx.clear_cache",
        lambda: clear_calls.append(True),
    )
    lazy = LazyDrafter(
        path="draft-model",
        kind="mtp",
        config=SimpleNamespace(model_type="qwen3_5_mtp"),
        loader=loader,
        validator=validator,
        target_model=target,
    )

    assert not lazy.loaded
    assert lazy.accept_lens == []
    assert lazy.speculative_total_rounds == 0

    first = lazy.materialize()
    assert lazy.loaded
    assert lazy.materialize() is first
    assert len(loaded) == 1
    assert validated == [(target, first, "mtp")]
    assert lazy.speculative_total_rounds == 7

    lazy.unload()
    assert not lazy.loaded
    assert clear_calls == [True]
    second = lazy.materialize()
    assert second is not first
    assert len(loaded) == 2


def test_lazy_drafter_rejects_a_changed_resolved_kind():
    lazy = LazyDrafter(
        path="draft-model",
        kind="mtp",
        config=object(),
        loader=lambda path, kind: (object(), "dflash"),
        validator=lambda *args: None,
        target_model=object(),
    )

    with pytest.raises(ValueError, match="resolved as 'dflash'"):
        lazy.materialize()
    assert not lazy.loaded


def test_speculative_rounds_materialize_a_deferred_drafter():
    model = object()
    materialize_calls = []
    deferred = SimpleNamespace(
        materialize=lambda: materialize_calls.append(True) or model
    )
    batch = SpeculativeGenerationBatch.__new__(SpeculativeGenerationBatch)
    batch._rounds_iter = None
    batch.draft_model = deferred
    batch._finished = []
    batch.stop_criteria = lambda _token: False
    batch._num_tokens = []
    batch.max_tokens = []
    batch.model = object()
    batch.prompt_cache = []
    batch.hidden = None
    batch.draft_kind = "mtp"
    batch.first_tokens = SimpleNamespace(ndim=1, shape=(0,))
    batch.sampler = object()
    batch.draft_block_size = 3
    batch.token_dtype = object()
    batch.greedy_sampling = False
    batch.shared_kv_states = None
    batch.prompt_tokens = None
    batch._all_uids = []

    batch._start_rounds()

    assert materialize_calls == [True]
    assert batch.draft_model is model
