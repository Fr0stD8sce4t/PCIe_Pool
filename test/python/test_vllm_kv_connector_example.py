from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


def load_example_module():
    path = Path(__file__).resolve().parents[2] / "examples" / "vllm_turbobus_kv_connector.py"
    spec = importlib.util.spec_from_file_location("vllm_turbobus_kv_connector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


example = load_example_module()


class VllmKVConnectorExampleTest(unittest.TestCase):
    def test_save_is_enabled_by_default_independent_of_restore(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "vllm_turbobus_kv_connector.py",
                "--model",
                "model",
                "--target-gpu",
                "6",
            ],
        ):
            args = example.parse_args()

        self.assertTrue(args.save_enabled)
        self.assertFalse(args.restore_enabled)

    def test_no_save_disables_first_request_save_intent(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "vllm_turbobus_kv_connector.py",
                "--model",
                "model",
                "--target-gpu",
                "6",
                "--no-save",
                "--restore-enabled",
            ],
        ):
            args = example.parse_args()

        self.assertFalse(args.save_enabled)
        self.assertTrue(args.restore_enabled)


if __name__ == "__main__":
    unittest.main()
