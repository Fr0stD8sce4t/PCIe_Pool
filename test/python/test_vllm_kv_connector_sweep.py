from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import tempfile
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

    def test_gib_per_second_formats_restore_bandwidth(self) -> None:
        self.assertEqual(sweep.gib_per_second(str(1024**3), "1000"), "1.000")
        self.assertEqual(sweep.gib_per_second("0", "1000"), "NA")

    def test_speedup_formats_latency_ratio(self) -> None:
        self.assertEqual(sweep.speedup("40", "20"), "2.000")
        self.assertEqual(sweep.speedup("40", "0"), "NA")

    def test_build_sweep_summary_lines_includes_bandwidth_and_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            direct_log = Path(tmpdir) / "direct.log"
            pool_log = Path(tmpdir) / "pool.log"
            direct_log.write_text(
                "\n".join(
                    [
                        "turbobus_kv_connector_event event=restore elapsed_ms=40 prepare_ms=2 transfer_ms=40 total_ms=42 layers=28 ranges=224 bytes=1073741824 direct_chunks=1 relay_chunks=0",
                        "turbobus_kv_connector_event event=start_load_done requests=1 restore_enabled=True elapsed_ms=43",
                    ]
                ),
                encoding="utf-8",
            )
            pool_log.write_text(
                "\n".join(
                    [
                        "turbobus_kv_connector_event event=restore elapsed_ms=20 prepare_ms=1 transfer_ms=20 total_ms=21 layers=28 ranges=224 bytes=1073741824 direct_chunks=1 relay_chunks=1 auto_resolved_mode=pool auto_reason=pool_speedup_1.500 auto_direct_bw_gbps=7.500 auto_relay_bw_gbps=7.600",
                        "turbobus_kv_connector_event event=start_load_done requests=1 restore_enabled=True elapsed_ms=22",
                    ]
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                target_gpu=6,
                relay_gpus="5",
                model="model",
                prompt_repeat=64,
                modes=["direct", "pool"],
                restore_blocks_list=[8],
                chunk_bytes=4194304,
                profile_bytes=16777216,
            )
            results = [
                {
                    "mode": "direct",
                    "restore_blocks": 8,
                    "matched_tokens": 128,
                    "returncode": 0,
                    "log_path": str(direct_log),
                    "summary": {
                        "vllm_kv_connector_save": {
                            "elapsed_ms": "41",
                            "bytes": "1073741824",
                            "save_layer_count": "28",
                            "save_layer_ranges": "56",
                        },
                        "vllm_kv_connector_result": {"prompt_tokens": "321", "shared_prefix": "True"},
                    },
                },
                {
                    "mode": "pool",
                    "restore_blocks": 8,
                    "matched_tokens": 128,
                    "returncode": 0,
                    "log_path": str(pool_log),
                    "summary": {
                        "vllm_kv_connector_save": {
                            "elapsed_ms": "21",
                            "bytes": "1073741824",
                            "save_layer_count": "28",
                            "save_layer_ranges": "56",
                        },
                        "vllm_kv_connector_result": {"prompt_tokens": "321", "shared_prefix": "True"},
                    },
                },
            ]

            lines = sweep.build_sweep_summary_lines(args, results)

        self.assertTrue(any("mode=pool" in line and "restore_gib_s=50.000" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "restore_prepare_ms=1" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "start_load_ms=22" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "layers=28" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "ranges=224" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "auto_resolved_mode=pool" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "auto_reason=pool_speedup_1.500" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "auto_direct_bw_gbps=7.500" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "auto_relay_bw_gbps=7.600" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "save_layer_count=28" in line for line in lines))
        self.assertTrue(any("mode=pool" in line and "save_layer_ranges=56" in line for line in lines))
        self.assertTrue(any("direct_over_pool_restore=2.000" in line for line in lines))

    def test_print_sweep_summary_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.txt"
            args = SimpleNamespace(
                target_gpu=6,
                relay_gpus="5",
                model="model",
                prompt_repeat=64,
                modes=[],
                restore_blocks_list=[],
                chunk_bytes=4194304,
                profile_bytes=16777216,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.print_sweep_summary(args, [], output)

            self.assertIn("SWEEP_SUMMARY_BEGIN", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
