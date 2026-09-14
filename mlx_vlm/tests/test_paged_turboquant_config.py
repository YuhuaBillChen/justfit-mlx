"""Configuration contracts, runnable without MLX on the host CPU.

Run directly with Python to avoid the package's model-loading imports.
"""

import importlib.util
import json
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location(
    "_paged_config_under_test",
    Path(__file__).resolve().parents[1] / "paged_turboquant_config.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
Config = _module.PagedTurboQuantConfig


class TestPagedTurboQuantConfig(unittest.TestCase):
    def test_defaults_preserve_unset_environment_behavior(self):
        self.assertEqual(Config.from_env({}), Config())

    def test_explicit_configuration_ignores_process_environment(self):
        with patch.dict("os.environ", {"MLX_VLM_PAGED_PREFILL_IMPL": "typo"}):
            self.assertEqual(Config().prefill_impl, "compatibility")

    def test_legacy_environment_is_snapshotted(self):
        source = {
            "MLX_VLM_PAGED_PREFILL_IMPL": "direct_inverse",
            "MLX_VLM_PAGED_PREFILL_EAGER_RELEASE": "1",
            "MLX_VLM_TQ_MTP_QTILE": "1",
        }
        config = Config.from_env(source)
        source.clear()
        self.assertEqual(config, Config("direct_inverse", True, True))

    def test_empty_legacy_values_preserve_disabled_behavior(self):
        self.assertEqual(Config.from_env({
            "MLX_VLM_PAGED_PREFILL_IMPL": "",
            "MLX_VLM_PAGED_PREFILL_EAGER_RELEASE": "",
            "MLX_VLM_TQ_MTP_QTILE": "",
        }), Config())

    def test_invalid_environment_fails_before_execution(self):
        for name in (
            "MLX_VLM_PAGED_PREFILL_IMPL",
            "MLX_VLM_PAGED_PREFILL_EAGER_RELEASE",
            "MLX_VLM_TQ_MTP_QTILE",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Config.from_env({name: "typo"})

    def test_explicit_flags_require_booleans(self):
        for name in ("prefill_eager_release", "mtp_qtile"):
            for value in (1, "0", None):
                with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                    Config(**{name: value})

    def test_execution_policy_is_immutable(self):
        config = Config()
        with self.assertRaises(FrozenInstanceError):
            config.mtp_qtile = True

    def test_effective_settings_round_trip_through_json(self):
        config = Config("direct_inverse", True, True)
        record = json.loads(json.dumps(config.to_dict()))
        self.assertEqual(Config(**record), config)
        record["mtp_qtile"] = False
        self.assertTrue(config.mtp_qtile)


if __name__ == "__main__":
    unittest.main()
