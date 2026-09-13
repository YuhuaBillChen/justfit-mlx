from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.server.language_lifecycle import (
    LanguageEmbeddingPhaseSwap,
    LanguageHeadPhaseSwap,
)


def test_language_head_is_rebuilt_only_after_prefill(monkeypatch, tmp_path):
    component = tmp_path / "lm-head.safetensors"
    component.touch()
    original = nn.QuantizedLinear(32, 64, bias=False, group_size=32, bits=4)
    language_model = SimpleNamespace(lm_head=original)
    clear_calls = []
    weights = {
        "weight": original.weight,
        "scales": original.scales,
        "biases": original.biases,
    }
    monkeypatch.setattr(
        "mlx_vlm.server.language_lifecycle.mx.load", lambda path: weights
    )
    monkeypatch.setattr(
        "mlx_vlm.server.language_lifecycle.mx.clear_cache",
        lambda: clear_calls.append(True),
    )

    phase = LanguageHeadPhaseSwap(language_model, str(component))
    assert phase.loaded
    phase.unload()
    assert not phase.loaded
    restored = phase.load()

    assert restored is language_model.lm_head
    assert restored.group_size == 32
    assert restored.bits == 4
    assert restored.mode == original.mode
    assert restored.bias is None
    assert mx.array_equal(restored.weight, original.weight).item()
    assert phase.load() is restored
    assert clear_calls == [True]


def test_input_embedding_is_rebuilt_for_decode(monkeypatch, tmp_path):
    component = tmp_path / "embedding.safetensors"
    component.touch()
    original = nn.QuantizedEmbedding(64, 32, group_size=32, bits=4)
    language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=original))
    weights = {
        "weight": original.weight,
        "scales": original.scales,
        "biases": original.biases,
    }
    monkeypatch.setattr(
        "mlx_vlm.server.language_lifecycle.mx.load", lambda path: weights
    )

    phase = LanguageEmbeddingPhaseSwap(language_model, str(component))
    phase.unload()
    assert not phase.loaded
    restored = phase.load()

    assert restored is language_model.model.embed_tokens
    assert restored.num_embeddings == 64
    assert restored.dims == 32
    assert restored.group_size == 32
    assert mx.array_equal(restored.weight, original.weight).item()


def test_mxfp4_components_restore_none_biases_sentinel(monkeypatch, tmp_path):
    head_path = tmp_path / "lm-head.safetensors"
    embedding_path = tmp_path / "embedding.safetensors"
    head_path.touch()
    embedding_path.touch()
    head = nn.QuantizedLinear(
        32, 64, bias=False, group_size=32, bits=4, mode="mxfp4"
    )
    embedding = nn.QuantizedEmbedding(
        64, 32, group_size=32, bits=4, mode="mxfp4"
    )
    language_model = SimpleNamespace(
        lm_head=head, model=SimpleNamespace(embed_tokens=embedding)
    )
    components = {
        str(head_path): {"weight": head.weight, "scales": head.scales},
        str(embedding_path): {
            "weight": embedding.weight,
            "scales": embedding.scales,
        },
    }
    monkeypatch.setattr(
        "mlx_vlm.server.language_lifecycle.mx.load",
        lambda path: components[path],
    )

    head_phase = LanguageHeadPhaseSwap(language_model, str(head_path))
    embedding_phase = LanguageEmbeddingPhaseSwap(language_model, str(embedding_path))
    head_phase.unload()
    embedding_phase.unload()

    restored_head = head_phase.load()
    restored_embedding = embedding_phase.load()

    assert restored_head.mode == "mxfp4"
    assert restored_head.biases is None
    assert restored_embedding.mode == "mxfp4"
    assert restored_embedding.biases is None
