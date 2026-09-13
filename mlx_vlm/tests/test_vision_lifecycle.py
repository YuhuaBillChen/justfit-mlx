from types import SimpleNamespace

from mlx_vlm.server.vision_lifecycle import VisionTowerPhaseSwap


def test_vision_tower_is_loaded_only_for_the_active_phase(monkeypatch, tmp_path):
    component = tmp_path / "vision.safetensors"
    component.touch()
    calls = []

    class Tower:
        def __init__(self, config):
            self.config = config
            self.parameters_value = object()

        def load_weights(self, weights, strict):
            calls.append(("load_weights", weights, strict))

        def parameters(self):
            return self.parameters_value

        def eval(self):
            calls.append(("eval",))

    model = SimpleNamespace(vision_tower=Tower("vision-config"))
    clear_cache = []
    monkeypatch.setattr(
        "mlx_vlm.server.vision_lifecycle.mx.load",
        lambda path: {"layer.weight": "weight"},
    )
    monkeypatch.setattr(
        "mlx_vlm.server.vision_lifecycle.mx.eval",
        lambda value: calls.append(("mx.eval", value)),
    )
    monkeypatch.setattr(
        "mlx_vlm.server.vision_lifecycle.mx.clear_cache",
        lambda: clear_cache.append(True),
    )

    phase = VisionTowerPhaseSwap(model, str(component))

    assert model.vision_tower is None
    assert not phase.loaded
    loaded = phase.load()
    assert loaded is model.vision_tower
    assert loaded.config == "vision-config"
    assert phase.load() is loaded
    assert calls == [
        ("load_weights", [("layer.weight", "weight")], True),
        ("mx.eval", loaded.parameters_value),
        ("eval",),
    ]

    phase.unload()
    assert model.vision_tower is None
    assert not phase.loaded
    assert clear_cache == [True, True]
