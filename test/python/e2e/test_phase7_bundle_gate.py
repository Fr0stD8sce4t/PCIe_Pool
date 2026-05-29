from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_bundle_gate  # noqa: E402
import phase7_result_check  # noqa: E402


class Phase7BundleGateTest(unittest.TestCase):
    def test_accepts_complete_run_bundle(self) -> None:
        baseline = paper_result("paper-baseline", WORKLOADS)
        turbobus = paper_result("turbobus-daemon", WORKLOADS)

        report = phase7_bundle_gate.build_bundle_report(
            baseline_result=baseline,
            turbobus_result=turbobus,
            baseline_check=phase7_result_check.check_phase7_result(baseline),
            turbobus_check=phase7_result_check.check_phase7_result(turbobus),
            comparison=comparison_report(WORKLOADS),
            evidence_reports=[evidence_report(WORKLOADS)],
            correctness_reports=[{"ok": True, "summary": {"commands": 6}}],
            server_class="2gpu",
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["server_class"], "2gpu")
        self.assertEqual(report["comparison"]["comparison_count"], len(WORKLOADS))
        self.assertTrue(report["correctness"]["provided"])
        self.assertTrue(report["correctness"]["ok"])
        self.assertIn("vllm-kv", report["evidence"]["workloads"])

    def test_reports_missing_required_workload_and_real_workload(self) -> None:
        baseline = paper_result("paper-baseline", WORKLOADS)
        turbobus_workloads = ("model-loading", "training-offload", "optimizer-offload")
        turbobus = paper_result("turbobus-daemon", turbobus_workloads)

        report = phase7_bundle_gate.build_bundle_report(
            baseline_result=baseline,
            turbobus_result=turbobus,
            comparison=comparison_report(turbobus_workloads),
            evidence_reports=[evidence_report(turbobus_workloads)],
        )

        self.assertFalse(report["ok"])
        self.assertIn("turbobus:missing_workload:vllm-kv", report["errors"])
        self.assertIn("comparison:missing_workload:vllm-kv", report["errors"])
        self.assertIn("evidence:missing_workload:vllm-kv", report["errors"])
        self.assertIn("real_workload:vllm_kv_missing", report["errors"])

    def test_reports_bad_check_and_bad_evidence(self) -> None:
        baseline = paper_result("paper-baseline", WORKLOADS)
        turbobus = paper_result("turbobus-daemon", WORKLOADS)

        report = phase7_bundle_gate.build_bundle_report(
            baseline_result=baseline,
            turbobus_result=turbobus,
            baseline_check={"ok": False, "errors": ["stale report"]},
            comparison=comparison_report(WORKLOADS),
            evidence_reports=[{"ok": False, "errors": ["missing daemon evidence"], "workloads": []}],
        )

        self.assertFalse(report["ok"])
        self.assertIn("baseline:provided_check_failed", report["errors"])
        self.assertIn("evidence:0:not_ok", report["errors"])
        self.assertIn("evidence:missing_workload:model-loading", report["errors"])

    def test_cli_writes_json_output(self) -> None:
        baseline = paper_result("paper-baseline", WORKLOADS)
        turbobus = paper_result("turbobus-daemon", WORKLOADS)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_result = write_json(tmp / "baseline_result.json", baseline)
            turbobus_result = write_json(tmp / "turbobus_result.json", turbobus)
            baseline_check = write_json(
                tmp / "baseline_check.json",
                phase7_result_check.check_phase7_result(baseline),
            )
            turbobus_check = write_json(
                tmp / "turbobus_check.json",
                phase7_result_check.check_phase7_result(turbobus),
            )
            comparison = write_json(tmp / "comparison.json", comparison_report(WORKLOADS))
            evidence = write_json(tmp / "evidence.json", evidence_report(WORKLOADS))
            correctness = write_json(tmp / "correctness.json", {"ok": True})
            output = tmp / "bundle.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_bundle_gate.py"),
                    "--server-class",
                    "2gpu",
                    "--baseline-result",
                    str(baseline_result),
                    "--turbobus-result",
                    str(turbobus_result),
                    "--baseline-check",
                    str(baseline_check),
                    "--turbobus-check",
                    str(turbobus_check),
                    "--comparison",
                    str(comparison),
                    "--evidence",
                    str(evidence),
                    "--correctness",
                    str(correctness),
                    "--json-output",
                    str(output),
                ],
                cwd=BENCHMARKS.parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["artifacts"]["baseline_result"], str(baseline_result))


WORKLOADS = (
    "model-loading",
    "training-offload",
    "optimizer-offload",
    "vllm-kv",
)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def paper_result(policy: str, workloads: tuple[str, ...]) -> dict:
    return {
        "config": {
            "workloads": list(workloads),
            "policy": policy,
        },
        "workloads": [
            {
                "workload": workload,
                "status": "ok",
                "returncode": 0,
                "validation_errors": [],
                "data": {"vllm_kv_multi_job": {"job_count": 1}} if workload == "vllm-kv" else {},
                "metrics": [metric(workload)],
            }
            for workload in workloads
        ],
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
        "transfer_ms": 10.0,
        "performance_ms": 10.0,
        "fallback_reason": "none",
        "correctness_status": "complete",
    }


def comparison_report(workloads: tuple[str, ...]) -> dict:
    return {
        "ok": True,
        "errors": [],
        "baseline_policy": "paper-baseline",
        "turbobus_policy": "turbobus-daemon",
        "path_selection_note": "path split comes from daemon receipts",
        "comparisons": [
            {
                "key": workload,
                "workload": workload,
                "baseline": {"trace_ids": {"decision_ids": f"{workload}-baseline-decision"}},
                "turbobus": {"trace_ids": {"decision_ids": f"{workload}-decision"}},
            }
            for workload in workloads
        ],
    }


def evidence_report(workloads: tuple[str, ...]) -> dict:
    return {
        "ok": True,
        "errors": [],
        "comparison": {"ok": True, "comparison_count": len(workloads)},
        "profile_summary": {
            "active_transfer_count": 1,
            "relay_path_bytes_total": 32,
            "audit_record_count": len(workloads),
        },
        "workloads": [
            {
                "key": workload,
                "workload": workload,
                "flags": {
                    "has_transfer_record": True,
                    "has_audit_record": True,
                    "has_runtime_job_state": True,
                    "has_relay_evidence": True,
                },
            }
            for workload in workloads
        ],
    }


if __name__ == "__main__":
    unittest.main()
