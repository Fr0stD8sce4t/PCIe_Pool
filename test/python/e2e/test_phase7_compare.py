from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_compare  # noqa: E402


class Phase7CompareTest(unittest.TestCase):
    def test_compares_checker_approved_results_and_preserves_trace_ids(self) -> None:
        baseline = paper_result(
            "paper-baseline",
            [
                workload_result(
                    "model-loading",
                    [metric("model-loading", transfer_ms=20.0, throughput_gib_s=4.0)],
                    data={
                        "samples": [
                            {"load_ms": 30.0, "load_gib_per_second": 3.0},
                            {"load_ms": 20.0, "load_gib_per_second": 4.0},
                            {"load_ms": 10.0, "load_gib_per_second": 8.0},
                        ]
                    },
                )
            ],
        )
        turbobus_metric = metric(
            "model-loading",
            transfer_ms=10.0,
            throughput_gib_s=8.0,
            direct_bytes=32,
            relay_bytes=32,
            fallback_reason="none",
        )
        turbobus = paper_result(
            "turbobus-daemon",
            [
                workload_result(
                    "model-loading",
                    [turbobus_metric],
                    data={
                        "samples": [
                            {"load_ms": 12.0, "load_gib_per_second": 7.0},
                            {"load_ms": 10.0, "load_gib_per_second": 8.0},
                            {"load_ms": 8.0, "load_gib_per_second": 10.0},
                        ]
                    },
                )
            ],
        )

        report = phase7_compare.compare_result_data(baseline, turbobus)

        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["baseline_policy"], "paper-baseline")
        self.assertEqual(report["turbobus_policy"], "turbobus-daemon")
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["key"], "model-loading")
        self.assertEqual(comparison["comparison"]["transfer_ms_speedup"], 2.0)
        self.assertEqual(comparison["comparison"]["throughput_gib_s_ratio"], 2.0)
        self.assertEqual(comparison["comparison"]["relay_bytes_delta"], 32)
        self.assertEqual(
            comparison["turbobus"]["trace_ids"]["decision_ids"],
            "model-loading-decision",
        )
        self.assertEqual(
            comparison["turbobus"]["path_split"]["source"],
            "daemon_transfer_receipt",
        )
        self.assertIn(
            "transfer_ms_p99",
            comparison["baseline"]["timing"]["percentiles"],
        )

    def test_reports_checker_failure_before_comparing(self) -> None:
        bad_baseline_metric = metric("model-loading")
        bad_baseline_metric["ticket_ids"] = ""
        baseline = paper_result(
            "paper-baseline",
            [workload_result("model-loading", [bad_baseline_metric])],
        )
        turbobus = paper_result(
            "turbobus-daemon",
            [workload_result("model-loading", [metric("model-loading")])],
        )

        report = phase7_compare.compare_result_data(baseline, turbobus)

        self.assertFalse(report["ok"])
        self.assertIn("baseline_checker_failed", report["errors"])
        self.assertEqual(report["comparisons"], [])

    def test_reports_missing_matching_workload_metric(self) -> None:
        baseline = paper_result(
            "paper-baseline",
            [workload_result("model-loading", [metric("model-loading")])],
        )
        turbobus = paper_result(
            "turbobus-daemon",
            [workload_result("training-offload", [metric("training-offload")])],
        )

        report = phase7_compare.compare_result_data(baseline, turbobus)

        self.assertFalse(report["ok"])
        self.assertIn("missing_turbobus_metric:model-loading", report["errors"])
        self.assertIn("missing_baseline_metric:training-offload", report["errors"])

    def test_cli_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_path = tmp / "baseline.json"
            turbobus_path = tmp / "turbobus.json"
            output_path = tmp / "comparison.json"
            baseline_path.write_text(
                json.dumps(
                    paper_result(
                        "paper-baseline",
                        [workload_result("model-loading", [metric("model-loading")])],
                    )
                ),
                encoding="utf-8",
            )
            turbobus_path.write_text(
                json.dumps(
                    paper_result(
                        "turbobus-daemon",
                        [workload_result("model-loading", [metric("model-loading")])],
                    )
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_compare.py"),
                    "--baseline",
                    str(baseline_path),
                    "--turbobus",
                    str(turbobus_path),
                    "--json-output",
                    str(output_path),
                ],
                cwd=BENCHMARKS.parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["comparisons"][0]["key"], "model-loading")


def paper_result(policy: str, workloads: list[dict]) -> dict:
    return {
        "config": {
            "workloads": [workload["workload"] for workload in workloads],
            "policy": policy,
        },
        "workloads": workloads,
    }


def workload_result(workload: str, metrics: list[dict], data: dict | None = None) -> dict:
    return {
        "workload": workload,
        "status": "ok",
        "returncode": 0,
        "validation_errors": [],
        "data": data or {},
        "metrics": metrics,
    }


def metric(
    workload: str,
    *,
    transfer_ms: float = 10.0,
    performance_ms: float | None = None,
    throughput_gib_s: float = 8.0,
    direct_bytes: int = 64,
    relay_bytes: int = 0,
    fallback_reason: str = "none",
) -> dict:
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
        "direct_bytes": direct_bytes,
        "relay_bytes": relay_bytes,
        "direct_chunks": 1 if direct_bytes else 0,
        "relay_chunks": 1 if relay_bytes else 0,
        "transfer_ms": transfer_ms,
        "performance_ms": performance_ms if performance_ms is not None else transfer_ms,
        "throughput_gib_s": throughput_gib_s,
        "fallback_reason": fallback_reason,
        "correctness_status": "complete",
    }


if __name__ == "__main__":
    unittest.main()
