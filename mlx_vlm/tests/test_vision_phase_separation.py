from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.models.qwen3_5.qwen3_5 import Model
from mlx_vlm.server.component_residency import ComponentResidencyManager
from mlx_vlm.server.generation import ResponseGenerator


def test_video_keeps_existing_embedding_path():
    events = []
    tower = SimpleNamespace(
        load=lambda: events.append("load"),
        unload=lambda: events.append("unload"),
    )

    class FakeModel:
        def encode_image_features(self, *args, **kwargs):
            pytest.fail("image-only phase separation must not intercept video")

        def get_input_embeddings(self, *args, **kwargs):
            events.append("video")
            return SimpleNamespace(
                to_dict=lambda: {"inputs_embeds": mx.ones((1, 1, 3))}
            )

    gen = SimpleNamespace(model=FakeModel(), vision_cache=None, vision_phase_swap=tower)
    ResponseGenerator._gpu_embed(
        gen,
        {
            "input_ids": mx.array([[1]]),
            "pixel_values": mx.ones((1, 3)),
            "video_grid_thw": mx.array([[1, 1, 1]]),
        },
    )
    assert events == ["load", "video", "unload"]


@pytest.mark.parametrize("hit", [False, True])
def test_features_finish_before_text_embedding_and_cache_hit_skips_tower(hit):
    events = []
    features = mx.ones((1, 3))

    class Cache:
        def get(self, key):
            return features if hit else None

        def put(self, key, value):
            assert mx.array_equal(value, features).item()

    class Tower:
        loaded = False

        def load(self):
            self.loaded = True
            events.append("load")

        def unload(self):
            self.loaded = False
            events.append("unload")

    tower = Tower()

    class FakeModel:
        def encode_image_features(self, pixels, grid, *, batch_size):
            assert tower.loaded
            events.append("encode")
            return features

        def get_input_embeddings(self, ids, pixels, **kwargs):
            assert not tower.loaded
            assert kwargs["cached_image_features"] is features
            events.append("text")
            return SimpleNamespace(to_dict=lambda: {"inputs_embeds": features})

    gen = SimpleNamespace(
        model=FakeModel(), vision_cache=Cache(), vision_phase_swap=tower
    )
    _, kwargs = ResponseGenerator._gpu_embed(
        gen, {"input_ids": mx.array([[1]]), "pixel_values": mx.ones((1, 3))}, ["image"]
    )
    assert events == (["text"] if hit else ["load", "encode", "unload", "text"])
    assert "cached_image_features" not in kwargs


def test_qwen_cached_features_work_with_vision_tower_absent():
    model = SimpleNamespace(
        vision_tower=None,
        config=SimpleNamespace(image_token_index=99, video_token_index=100),
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=lambda ids: mx.zeros((*ids.shape, 3))),
            get_rope_index=lambda *args, **kwargs: (None, None),
        ),
    )
    result = Model.get_input_embeddings(
        model,
        mx.array([[99]]),
        mx.ones((1, 3)),
        cached_image_features=mx.ones((1, 3)),
        chunked=True,
    )
    actual = result.input_embedding_provider(mx.array([[99]]), start=0)
    assert mx.array_equal(actual, mx.ones((1, 1, 3))).item()


def test_provider_does_not_keep_retired_embedding_alive():
    import gc
    import weakref

    from mlx_vlm.models.qwen3_5.qwen3_5 import CurrentInputEmbedding

    class Table:
        def __call__(self, ids):
            return ids + 1

    owner = SimpleNamespace(embed_tokens=Table())
    previous = weakref.ref(owner.embed_tokens)
    provider = CurrentInputEmbedding(owner)
    owner.embed_tokens = None
    gc.collect()
    assert previous() is None
    with pytest.raises(RuntimeError, match="unloaded"):
        provider(mx.array([1]))
    owner.embed_tokens = Table()
    assert provider(mx.array([1])).item() == 2


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_managed_embedding_swap_restores_after_vision_and_respects_owners(
    fail, pinned, active
):
    events = []

    class Component:
        def __init__(self, name, loaded):
            self.name, self.loaded = name, loaded

        def load(self):
            self.loaded = True
            events.append(self.name + "+")

        def unload(self):
            self.loaded = False
            events.append(self.name + "-")

    embedding = Component("embedding", True)
    tower = Component("tower", False)
    manager = ComponentResidencyManager()
    manager.register("input_embedding", embedding, retain_on_release=True)
    manager.register("vision_tower", tower)
    if pinned:
        manager.acquire("input_embedding", "other")
    if active:
        manager.acquire("input_embedding", "generation")
    events.clear()

    class FakeModel:
        def encode_image_features(self, *args, **kwargs):
            assert embedding.loaded == pinned
            assert tower.loaded
            if fail:
                raise ValueError("encoding failed")
            return mx.ones((1, 3))

        def get_input_embeddings(self, *args, **kwargs):
            assert embedding.loaded and not tower.loaded
            return SimpleNamespace(
                to_dict=lambda: {"inputs_embeds": mx.ones((1, 1, 3))}
            )

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.model, gen.vision_cache = FakeModel(), None
    gen.vision_phase_swap, gen.component_residency = tower, manager

    def run():
        gen._gpu_embed_at_active_decode_boundary(
            {"input_ids": mx.array([[1]]), "pixel_values": mx.ones((1, 3))},
            None,
            active=active,
            apc_semantic_hash=None,
        )

    if fail:
        with pytest.raises(ValueError, match="encoding failed"):
            run()
    else:
        run()
    assert embedding.loaded and not tower.loaded
    assert ("generation" in manager.owners("input_embedding")) == active
    if not pinned:
        assert events[:4] == ["embedding-", "tower+", "tower-", "embedding+"]
    else:
        assert "embedding-" not in events
