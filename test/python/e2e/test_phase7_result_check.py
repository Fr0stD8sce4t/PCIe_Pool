from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_result_check  # noqa: E402


class Phase7ResultCheckTest(unittest.TestCase):
    def test_accepts_complete_paper_validation_result(self) -> None:
        report = phase7_result_check.check_phase7_result(
            paper_result(
                [
                    workload_result("model-loading", [metric("model-loading")]),
                    workload_result("training-offload", [metric("training-offload")]),
                    workload_result("optimizer-offload", [metric("optimizer-offload")]),
                    workload_result("vllm-kv", [metric("vllm-kv")]),
                ]
            )
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])

    def test_reports_missing_trace_and_completion_errors(self) -> None:
        bad_metric = metric("model-loading")
        bad_metric["receipt_ids"] = ""
        bad_metric["decision_ids"] = ""
        bad_metric["bytes_completed"] = 32
        bad_metric["correctness_status"] = "incomplete"

        report = phase7_result_check.check_phase7_result(
            paper_result([workload_result("model-loading", [bad_metric])])
        )

        self.assertFalse(report["ok"])
        workload_errors = report["workloads"][0]["errors"]
        self.assertIn("metric_0_missing_receipt_ids", workload_errors)
        self.assertIn("metric_0_missing_decision_ids", workload_errors)
        self.assertIn("metric_0_bytes_not_fully_completed", workload_errors)
        self.assertIn("metric_0_invalid_correctness_status", workload_errors)

    def test_reports_multi_job_vllm_identity_problems(self) -> None:
        first = metric("vllm-kv")
        second = metric("vllm-kv")
        second["job_index"] = 1
        second["job_id"] = first["job_id"]

        report = phase7_result_check.check_phase7_result(
            paper_result(
                [
                    {
                        **workload_result("vllm-kv", [first, second]),
                        "data": {"vllm_kv_multi_job": {"job_count": 2}},
                    }
                ]
            )
        )

        self.assertFalse(report["ok"])
        self.assertIn("multi_job_job_id_not_distinct", report["workloads"][0]["errors"])

    def test_accepts_compact_summary_input(self) -> None:
        summary = "\n".join(
            [
                "PAPER_VALIDATION_SUMMARY_BEGIN",
                "paper_workload workload=model-loading status=ok validation_errors=",
                "paper_metric " + " ".join(
                    f"{key}={value}" for key, value in metric("model-loading").items()
                ),
                "PAPER_VALIDATION_SUMMARY_END",
            ]
        )

        report = phase7_result_check.check_phase7_result(
            phase7_result_check.parse_summary_text(summary)
        )

        self.assertTrue(report["ok"])

    def test_cli_writes_machine_readable_report_and_returns_nonzero_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            report_path = Path(tmpdir) / "check.json"
            bad_metric = metric("model-loading")
            bad_metric["ticket_ids"] = ""
            result_path.write_text(
                json.dumps(paper_result([workload_result("model-loading", [bad_metric])])),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_result_check.py"),
                    str(result_path),
                    "--json-output",
                    str(report_path),
                ],
                cwd=BENCHMARKS.parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertIn("model-loading:metric_0_missing_ticket_ids", report["errors"])


def paper_result(workloads: list[dict]) -> dict:
    return {
        "config": {
            "workloads": [workload["workload"] for workload in workloads],
            "policy": "turbobus-daemon",
        },
        "workloads": workloads,
    }


def workload_result(workload: str, metrics: list[dict]) -> dict:
    return {
        "workload": workload,
        "status": "ok",
        "returncode": 0,
        "validation_errors": [],
        "metrics": metrics,
    }


def metric(workload: str) -> dict:
    workload_kind = {
        "model-loading": "model_weights",
        "training-offload": "training_state",
        "optimizer-offload": "optimizer_state",
        "vllm-kv": "kv_cache",
    }[workload]
    return {
        "report_schema": "phase6_unified_v1",
        "workload": workload,
        "policy": "turbobus-daemon",
        "job_id": f"{workload}-job",
        "session_id": f"{workload}-session",
        "workload_kind": workload_kind,
        "cpu_buffer_id": f"{workload}-cpu",
        "gpu_buffer_id": f"{workload}-gpu",
        "receipt_ids": f"{workload}-receipt",
        "decision_ids": f"{workload}-decision",
        "topology_snapshot_ids": f"{workload}-topology",
        "ticket_ids": f"{workload}-ticket",
        "transfer_bytes": 64,
        "bytes_completed": 64,
        "direct_bytes": 32,
        "relay_bytes": 32,
        "direct_chunks": 1,
        "relay_chunks": 1,
        "transfer_ms": 1.0,
        "performance_ms": 1.0,
        "fallback_reason": "none",
        "correctness_status": "complete",
    }


if __name__ == "__main__":
    unittest.main()
