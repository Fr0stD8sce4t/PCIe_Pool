from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


def load_sweep_module():
    path = Path(__file__).resolve().parents[3] / "examples" / "vllm_turbobus_kv_connector_sweep.py"
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
            "vllm_kv_connector_config case_id=daemon second_prompt_suffix=' Italy'"
        )

        self.assertEqual(name, "vllm_kv_connector_config")
        self.assertEqual(values["case_id"], "daemon")
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

    def test_build_sweep_summary_lines_reports_daemon_receipt_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "daemon.log"
            log.write_text(
                "\n".join(
                    [
                        "turbobus_kv_connector_event event=save elapsed_ms=12 transfer_ms=10 total_ms=13 receipt_ids=save-r decision_ids=save-d topology_snapshot_ids=save-t ticket_ids=save-ticket direct_bytes=8 relay_bytes=0 fallback_reason=",
                        "turbobus_kv_connector_event event=restore elapsed_ms=20 prepare_ms=1 transfer_ms=20 total_ms=21 layers=28 ranges=224 bytes=1073741824 direct_chunks=1 relay_chunks=1 direct_bytes=536870912 relay_bytes=536870912 receipt_ids=restore-r decision_ids=restore-d topology_snapshot_ids=restore-t ticket_ids=restore-ticket fallback_reason=direct_saturated",
                        "turbobus_kv_connector_event event=start_load_done requests=1 restore_enabled=True elapsed_ms=22",
                    ]
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model="model",
                job_id="job-a",
                session_id="session-a",
                cpu_buffer_id="cpu-buffer",
                gpu_buffer_id="gpu-buffer",
                prompt_repeat=64,
                case_ids=["daemon"],
                restore_blocks_list=[8],
                chunk_bytes=4194304,
                daemon_socket_path="/tmp/turbobusd.sock",
                wait_timeout_seconds=None,
            )
            results = [
                {
                    "case_id": "daemon",
                    "restore_blocks": 8,
                    "matched_tokens": 128,
                    "returncode": 0,
                    "log_path": str(log),
                    "summary": {
                        "vllm_kv_connector_config": {
                            "job_id": "job-a",
                            "session_id": "session-a",
                        },
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
            rows = sweep.build_case_rows(args, results)

        self.assertTrue(any("case_id=daemon" in line and "restore_gib_s=50.000" in line for line in lines))
        self.assertEqual(rows[0]["case_id"], "daemon")
        self.assertEqual(rows[0]["restore_gib_s"], "50.000")
        self.assertEqual(rows[0]["receipt_ids"], "restore-r")
        self.assertEqual(rows[0]["decision_ids"], "restore-d")
        self.assertEqual(rows[0]["topology_snapshot_ids"], "restore-t")
        self.assertEqual(rows[0]["ticket_ids"], "restore-ticket")
        self.assertEqual(rows[0]["direct_bytes"], "536870912")
        self.assertEqual(rows[0]["relay_bytes"], "536870912")
        self.assertEqual(rows[0]["fallback_reason"], "direct_saturated")
        self.assertTrue(any("case_id=daemon" in line and "start_load_ms=22" in line for line in lines))
        forbidden = ("auto_" + "resolved_mode", "direct_" + "over_pool")
        self.assertTrue(all(all(item not in line for item in forbidden) for line in lines))

    def test_print_sweep_summary_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.txt"
            json_output = Path(tmpdir) / "cases.json"
            csv_output = Path(tmpdir) / "cases.csv"
            log = Path(tmpdir) / "daemon.log"
            log.write_text(
                "turbobus_kv_connector_event event=restore elapsed_ms=20 bytes=1073741824\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model="model",
                job_id="job-a",
                session_id="session-a",
                cpu_buffer_id="cpu-buffer",
                gpu_buffer_id="gpu-buffer",
                prompt_repeat=64,
                case_ids=[],
                restore_blocks_list=[],
                chunk_bytes=4194304,
                daemon_socket_path="/tmp/turbobusd.sock",
                wait_timeout_seconds=None,
            )
            results = [
                {
                    "case_id": "daemon",
                    "restore_blocks": 8,
                    "matched_tokens": 128,
                    "returncode": 0,
                    "log_path": str(log),
                    "summary": {},
                },
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.print_sweep_summary(
                    args,
                    results,
                    output,
                    cases_json_output=json_output,
                    cases_csv_output=csv_output,
                )

            self.assertIn("SWEEP_SUMMARY_BEGIN", output.read_text(encoding="utf-8"))
            cases = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["case_id"], "daemon")
            self.assertIn("restore_gib_s", csv_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
