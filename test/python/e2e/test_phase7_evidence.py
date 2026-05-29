from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_evidence  # noqa: E402


class Phase7EvidenceTest(unittest.TestCase):
    def test_attaches_daemon_profile_evidence_to_checked_result(self) -> None:
        result = paper_result([workload_result("model-loading", [metric("model-loading")])])
        comparison = {
            "ok": True,
            "errors": [],
            "path_selection_note": "path split comes from daemon receipts",
            "comparisons": [{"key": "model-loading"}],
        }

        report = phase7_evidence.build_evidence_report(
            result,
            daemon_profile(),
            comparison=comparison,
        )

        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["profile_summary"]["active_transfer_count"], 2)
        self.assertEqual(report["profile_summary"]["relay_path_bytes_total"], 32)
        self.assertEqual(report["comparison"]["comparison_count"], 1)
        workload = report["workloads"][0]
        self.assertEqual(workload["key"], "model-loading")
        self.assertTrue(workload["flags"]["has_transfer_record"])
        self.assertTrue(workload["flags"]["has_audit_record"])
        self.assertTrue(workload["flags"]["has_runtime_job_state"])
        self.assertTrue(workload["flags"]["has_relay_evidence"])
        self.assertTrue(workload["flags"]["has_active_contention_state"])
        self.assertEqual(
            workload["audit_records"][0]["ticket_id"],
            "model-loading-ticket",
        )
        self.assertIn("decision_ids", workload["trace_ids"])

    def test_reports_missing_daemon_trace_evidence(self) -> None:
        result = paper_result([workload_result("model-loading", [metric("model-loading")])])
        profile = {
            "runtime_resource_state": {"version": 1, "summary": {}},
            "audit_records": [],
        }

        report = phase7_evidence.build_evidence_report(result, profile)

        self.assertFalse(report["ok"])
        self.assertIn("model-loading:missing_daemon_trace_evidence", report["errors"])

    def test_reports_checker_failure_before_attaching_evidence(self) -> None:
        bad = metric("model-loading")
        bad["decision_ids"] = ""
        result = paper_result([workload_result("model-loading", [bad])])

        report = phase7_evidence.build_evidence_report(result, daemon_profile())

        self.assertFalse(report["ok"])
        self.assertIn("result_checker_failed", report["errors"])
        self.assertEqual(report["workloads"], [])

    def test_cli_writes_json_output_from_profile_response_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "result.json"
            profile_path = tmp / "profile.json"
            output_path = tmp / "evidence.json"
            result_path.write_text(
                json.dumps(
                    paper_result(
                        [workload_result("model-loading", [metric("model-loading")])]
                    )
                ),
                encoding="utf-8",
            )
            profile_path.write_text(
                json.dumps({"ok": True, "payload": daemon_profile()}),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_evidence.py"),
                    "--result",
                    str(result_path),
                    "--profile",
                    str(profile_path),
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
            self.assertEqual(report["profile_source"], str(profile_path))


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
        "transfer_ms": 10.0,
        "performance_ms": 10.0,
        "throughput_gib_s": 8.0,
        "fallback_reason": "none",
        "correctness_status": "complete",
    }


def daemon_profile() -> dict:
    return {
        "runtime_resource_state": {
            "version": 7,
            "transfers": [
                {
                    "transfer_id": "transfer-1",
                    "intent_id": "intent-1",
                    "decision_id": "model-loading-decision",
                    "topology_snapshot_id": "model-loading-topology",
                    "job_id": "model-loading-job",
                    "session_id": "model-loading-session",
                    "state": "running",
                    "direction": "h2d",
                    "bytes_total": 64,
                    "bytes_completed": 32,
                    "source_buffer_id": "model-loading-cpu",
                    "destination_buffer_id": "model-loading-gpu",
                    "buffer_ids": ("model-loading-cpu", "model-loading-gpu"),
                    "workload_kind": "model_weights",
                    "fallback_reason": "none",
                },
                {
                    "transfer_id": "transfer-2",
                    "intent_id": "intent-2",
                    "decision_id": "other-decision",
                    "topology_snapshot_id": "other-topology",
                    "job_id": "other-job",
                    "session_id": "other-session",
                    "state": "running",
                    "direction": "h2d",
                    "bytes_total": 64,
                    "bytes_completed": 0,
                },
            ],
            "active_transfers": [
                {
                    "transfer_id": "transfer-1",
                    "decision_id": "model-loading-decision",
                    "job_id": "model-loading-job",
                    "session_id": "model-loading-session",
                }
            ],
            "active_paths": [
                {
                    "transfer_id": "transfer-1",
                    "kind": "relay",
                    "direction": "h2d",
                    "relay_device": 1,
                    "target_device": 0,
                    "bytes_total": 32,
                    "chunk_count": 1,
                }
            ],
            "relay_staging": [
                {
                    "staging_record_id": "staging-lease-1",
                    "lease_id": "lease-1",
                    "transfer_id": "transfer-1",
                    "session_id": "model-loading-session",
                    "job_id": "model-loading-job",
                    "relay_gpu": 1,
                    "buffer_ids": ("model-loading-cpu", "model-loading-gpu"),
                }
            ],
            "job_runtime_state": {
                "model-loading-job": {
                    "job_id": "model-loading-job",
                    "weight": 1.0,
                    "active_transfer_count": 1,
                    "active_bytes_total": 64,
                    "active_bytes_remaining": 32,
                },
                "other-job": {
                    "job_id": "other-job",
                    "weight": 1.0,
                    "active_transfer_count": 1,
                    "active_bytes_total": 64,
                    "active_bytes_remaining": 64,
                },
            },
            "summary": {
                "queued_transfer_count": 0,
                "running_transfer_count": 2,
                "active_transfer_count": 2,
                "terminal_transfer_count": 0,
                "active_reservation_count": 1,
                "active_lease_count": 1,
                "relay_staging_count": 1,
                "relay_path_count": 1,
                "relay_path_bytes_total": 32,
                "active_resource_usage": {
                    "h2d": {"transfer_count": 2, "bytes_total": 128},
                    "p2p": {"path_count": 1, "chunk_count": 1, "bytes_total": 32},
                    "relay_staging": {
                        "count": 1,
                        "active_reservation_count": 1,
                        "active_lease_count": 1,
                    },
                },
            },
        },
        "audit_records": [
            {
                "audit_id": "audit-1",
                "event_type": "worker_completion",
                "transfer_id": "transfer-1",
                "decision_id": "model-loading-decision",
                "ticket_id": "model-loading-ticket",
                "topology_snapshot_id": "model-loading-topology",
                "lease_id": "lease-1",
                "session_id": "model-loading-session",
                "job_id": "model-loading-job",
                "relay_gpu": 1,
                "direction": "h2d",
                "bytes_total": 32,
                "bytes_completed": 32,
                "duration_seconds": 0.1,
                "state": "complete",
                "failure_reason": None,
                "source_buffer_id": "model-loading-cpu",
                "destination_buffer_id": "model-loading-gpu",
                "buffer_ids": ("model-loading-cpu", "model-loading-gpu"),
                "staging_record_id": "staging-lease-1",
            }
        ],
        "staging_records": {},
        "relay_quotas": {
            "1": {
                "relay_gpu": 1,
                "max_sessions": 2,
                "max_inflight_chunks": 8,
                "active_chunks": 1,
                "sessions": ["model-loading-session"],
            }
        },
        "cleanup_events": [],
        "system_cleanup_events": [],
    }


if __name__ == "__main__":
    unittest.main()
