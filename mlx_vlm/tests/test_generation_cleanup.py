"""A retired generator must not release a successor's shared component lease."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx_vlm.generate.ar import BatchGenerator
from mlx_vlm.server.language_lifecycle import ComponentResidencyManager


def test_repeated_close_does_not_release_successor_generation_head():
    manager = ComponentResidencyManager()
    head = Mock()
    manager.register("lm_head", head)
    retired = object.__new__(BatchGenerator)
    retired.model = SimpleNamespace(phase_residency_manager=manager)
    manager.acquire("lm_head", "generation")
    retired.close()
    head.unload.assert_called_once()

    # A new request has acquired the shared model's head. Python may finalize
    # the retired generator later, for example during warm APC restoration.
    manager.acquire("lm_head", "generation")
    retired.close()
    retired.__del__()
    assert manager.owners("lm_head") == frozenset({"generation"})
    head.unload.assert_called_once()
    manager.release("lm_head", "generation")


def test_close_retries_if_resource_cleanup_failed():
    manager = ComponentResidencyManager()
    head = Mock()
    manager.register("lm_head", head)
    generator = object.__new__(BatchGenerator)
    generator.model = SimpleNamespace(phase_residency_manager=manager)
    manager.acquire("lm_head", "generation")
    stack = Mock()
    stack.close.side_effect = [RuntimeError("temporary cleanup failure"), None]
    generator._wire_stack = stack
    with pytest.raises(RuntimeError, match="temporary cleanup"):
        generator.close()
    manager.acquire("lm_head", "generation")
    generator.close()
    generator.close()
    assert stack.close.call_count == 2
    assert manager.owners("lm_head") == frozenset({"generation"})
    head.unload.assert_called_once()
    manager.release("lm_head", "generation")
