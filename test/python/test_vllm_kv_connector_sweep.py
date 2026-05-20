from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def load_sweep_module():
    path = Path(__file__).resolve().parents[2] / "examples" / "vllm_turbobus_kv_connector_sweep.py"
    spec = importlib.util.spec_from_file_location("vllm_turbobus_kv_connector_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sweep = load_sweep_module()


class VllmKVConnectorSweepTest(unittest.TestCase):
    def test_parse_summary_line_handles_quoted_values(self) -> None:
        name, values = sweep.parse_summary_line(
            "vllm_kv_connector_config mode=pool second_prompt_suffix=' Italy'"
        )

        self.assertEqual(name, "vllm_kv_connector_config")
        self.assertEqual(values["mode"], "pool")
        self.assertEqual(values["second_prompt_suffix"], " Italy")

    def test_parse_copy_summary_extracts_named_records(self) -> None:
        parsed = sweep.parse_copy_summary(
            "\n".join(
                [
                    "before",
                    "COPY_SUMMARY_BEGIN",
                    "vllm_kv_connector_save elapsed_ms=12.5 bytes=64",
                    "vllm_kv_connector_result shared_prefix=True",
                    "COPY_SUMMARY_END",
                    "after",
                ]
            )
        )

        self.assertEqual(parsed["vllm_kv_connector_save"]["elapsed_ms"], "12.5")
        self.assertEqual(parsed["vllm_kv_connector_result"]["shared_prefix"], "True")


if __name__ == "__main__":
    unittest.main()
