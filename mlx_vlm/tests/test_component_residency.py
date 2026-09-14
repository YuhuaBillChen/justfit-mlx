"""Backend-independent lease contracts; also runnable with plain Python.

Load the module by path because mlx_vlm.__init__ eagerly imports MLX. This
keeps these tests runnable on CPU-only development hosts without mocking MLX.
"""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock


def load_manager():
    path = Path(__file__).parents[1] / "server" / "component_residency.py"
    spec = importlib.util.spec_from_file_location("component_residency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ComponentResidencyManager


class TestComponentResidency(unittest.TestCase):
    def setUp(self):
        self.manager = load_manager()()
        self.component = Mock()
        self.manager.register("head", self.component)

    def test_registration_is_idempotent_but_replacement_is_rejected(self):
        self.assertIs(self.manager.register("head", self.component), self.component)
        with self.assertRaises(ValueError):
            self.manager.register("head", Mock())

    def test_acquire_same_owner_is_idempotent(self):
        self.assertIs(self.manager.acquire("head", "decode"), self.component)
        self.manager.acquire("head", "decode")
        self.component.load.assert_called_once_with()
        self.assertEqual(self.manager.owners("head"), frozenset({"decode"}))

    def test_last_owner_controls_unload(self):
        self.manager.acquire("head", "decode")
        self.manager.acquire("head", "prefill")
        self.assertFalse(self.manager.unload_if_idle("head"))
        self.assertFalse(self.manager.release("head", "prefill"))
        self.component.unload.assert_not_called()
        self.assertTrue(self.manager.release("head", "decode"))
        self.component.unload.assert_called_once_with()

    def test_duplicate_release_does_not_unload_again(self):
        self.manager.acquire("head", "decode")
        self.manager.release("head", "decode")
        self.assertFalse(self.manager.release("head", "decode"))
        self.component.unload.assert_called_once_with()

    def test_failed_load_does_not_publish_owner(self):
        self.component.load.side_effect = [RuntimeError("load failed"), None]
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.manager.acquire("head", "decode")
        self.assertEqual(self.manager.owners("head"), frozenset())
        self.manager.acquire("head", "decode")
        self.assertEqual(self.manager.owners("head"), frozenset({"decode"}))

    def test_failed_second_load_preserves_existing_owner(self):
        self.manager.acquire("head", "decode")
        self.component.load.side_effect = RuntimeError("load failed")
        with self.assertRaises(RuntimeError):
            self.manager.acquire("head", "prefill")
        self.assertEqual(self.manager.owners("head"), frozenset({"decode"}))

    def test_unload_failure_consumes_lease_without_claiming_success(self):
        self.manager.acquire("head", "decode")
        self.component.unload.side_effect = [RuntimeError("unload failed"), None]
        with self.assertRaisesRegex(RuntimeError, "unload failed"):
            self.manager.release("head", "decode")
        self.assertEqual(self.manager.owners("head"), frozenset())
        self.assertFalse(self.manager.release("head", "decode"))
        # Retry reclamation explicitly, not by reviving a retired owner.
        self.assertTrue(self.manager.unload_if_idle("head"))

    def test_new_owner_prevents_idle_retry_after_unload_failure(self):
        self.manager.acquire("head", "old")
        self.component.unload.side_effect = RuntimeError("unload failed")
        with self.assertRaises(RuntimeError):
            self.manager.release("head", "old")
        self.manager.acquire("head", "new")
        self.assertFalse(self.manager.unload_if_idle("head"))
        self.component.unload.assert_called_once_with()

    def test_unknown_components_and_owners(self):
        self.assertFalse(self.manager.contains("unknown"))
        self.assertFalse(self.manager.release("unknown", "decode"))
        self.assertFalse(self.manager.unload_if_idle("unknown"))
        self.assertFalse(self.manager.release("head", "unknown"))
        with self.assertRaises(KeyError):
            self.manager.acquire("unknown", "decode")

    def test_owner_snapshot_is_not_live(self):
        self.manager.acquire("head", "decode")
        owners = self.manager.owners("head")
        self.manager.release("head", "decode")
        self.assertEqual(owners, frozenset({"decode"}))

    def test_components_have_independent_owners(self):
        vision = Mock()
        self.manager.register("vision", vision)
        self.manager.acquire("head", "decode")
        self.manager.acquire("vision", "media")
        self.manager.release("vision", "media")
        self.component.unload.assert_not_called()
        vision.unload.assert_called_once_with()

    def test_dependency_outlives_drafter_and_preserves_decode_owner(self):
        draft = Mock()
        self.manager.register("draft", draft, dependencies=("head",))
        self.manager.acquire("head", "decode")
        self.manager.acquire("draft", "mtp")
        draft.unload.side_effect = lambda: self.assertTrue(self.manager.owners("head"))
        self.manager.release("draft", "mtp")
        self.assertEqual(self.manager.owners("head"), frozenset({"decode"}))
        self.component.unload.assert_not_called()

    def test_failed_dependency_load_rolls_back_only_new_leases(self):
        draft = Mock()
        draft.load.side_effect = RuntimeError("draft failed")
        self.manager.register("draft", draft, dependencies=("head",))
        self.manager.acquire("head", "decode")
        with self.assertRaisesRegex(RuntimeError, "draft failed"):
            self.manager.acquire("draft", "mtp")
        self.assertEqual(self.manager.owners("head"), frozenset({"decode"}))
        self.assertFalse(self.manager.owners("draft"))

    def test_failed_drafter_unload_keeps_dependencies_until_retry(self):
        draft = Mock()
        draft.unload.side_effect = [RuntimeError("unload failed"), None]
        self.manager.register("draft", draft, dependencies=("head",))
        self.manager.acquire("draft", "mtp")
        with self.assertRaises(RuntimeError):
            self.manager.release("draft", "mtp")
        self.assertFalse(self.manager.unload_if_idle("head"))
        self.manager.unload_if_idle("draft")
        self.assertFalse(self.manager.owners("head"))

    def test_retained_component_requires_explicit_idle_eviction(self):
        embedding = Mock()
        self.manager.register("embedding", embedding, retain_on_release=True)
        self.manager.acquire("embedding", "decode")
        self.assertFalse(self.manager.release("embedding", "decode"))
        embedding.unload.assert_not_called()
        self.assertTrue(self.manager.unload_if_idle("embedding"))
        embedding.unload.assert_called_once_with()

    def test_dependency_graph_must_be_registered_in_order(self):
        with self.assertRaises(ValueError):
            self.manager.register("draft", Mock(), dependencies=("missing",))
        with self.assertRaises(ValueError):
            self.manager.register("draft", Mock(), dependencies=("draft",))
        self.assertFalse(self.manager.contains("draft"))

    def test_registered_policy_cannot_be_silently_changed(self):
        with self.assertRaises(ValueError):
            self.manager.register("head", self.component, retain_on_release=True)


if __name__ == "__main__":
    unittest.main()
