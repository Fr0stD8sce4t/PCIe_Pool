from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import paper_validation  # noqa: E402


def make_args(**overrides):
    values = {
        "session_id": "session-1",
        "job_id": "job-1",
        "cpu_buffer_id": "cpu-buffer",
        "gpu_buffer_id": "gpu-buffer",
        "policy": "daemon-default",
        "run_id": "run-1",
        "chunk_bytes": 4,
        "warmup": 0,
        "iterations": 1,
        "bucket_count": 4,
        "active_buckets": None,
        "bucket_bytes": 8,
        "storage_layout": "packed",
        "training_workload_kind": "training_state",
        "compute_delay_ms": 0.0,
        "keep_going": False,
        "output_dir": "benchmarks/results/paper_validation",
        "json_output": None,
        "summary_output": None,
        "no_copy_summary": False,
        "daemon_socket_path": "/tmp/turbobusd.sock",
        "daemon_max_inflight_chunks": 12,
        "daemon_profile_max_age_seconds": 45.0,
        "workloads": "all",
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PaperValidationTest(unittest.TestCase):
    def test_selected_workloads_expands_daemon_first_targets_and_defers_vllm(self) -> None:
        self.assertEqual(
            paper_validation.selected_workloads("all"),
            ["model-loading", "training-offload"],
        )
        self.assertEqual(
            paper_validation.selected_workloads("model-loading"),
            ["model-loading"],
        )
        with self.assertRaisesRegex(ValueError, "not daemon-first"):
            paper_validation.selected_workloads("vllm-kv")
        with self.assertRaises(ValueError):
            paper_validation.selected_workloads("missing")

    def test_build_commands_use_registered_buffers_not_physical_paths(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "model-loading")
            model = paper_validation.build_model_loading_command(args, paths)
            training = paper_validation.build_training_offload_command(args, paths)

        self.assertIn(str(BENCHMARKS / "model_loading.py"), model)
        self.assertIn(str(BENCHMARKS / "training_offload.py"), training)
        self.assertIn("--session-id", model)
        self.assertIn("--source-buffer-id", model)
        self.assertIn("--destination-buffer-id", model)
        self.assertIn("--cpu-buffer-id", training)
        self.assertIn("--gpu-buffer-id", training)
        self.assertIn("--daemon-socket-path", model)
        self.assertIn("--daemon-profile-max-age-seconds", training)
        forbidden = {"--target-gpu", "--relay-gpus", "--mode", "--modes", "--min-pool-bytes"}
        self.assertTrue(forbidden.isdisjoint(model))
        self.assertTrue(forbidden.isdisjoint(training))

    def test_collect_model_and_training_metrics_from_daemon_receipts(self) -> None:
        model = {
            "config": {"policy": "daemon-default"},
            "summary": {
                "iterations": 2,
                "median_load_ms": 12.5,
                "median_gib_per_second": 8.0,
                "bytes": 96,
                "bytes_completed": 96,
                "direct_bytes": 64,
                "relay_bytes": 32,
                "direct_chunks": 2,
                "relay_chunks": 1,
                "decision_ids": ["decision-1"],
                "topology_snapshot_ids": ["topology-1"],
                "ticket_ids": ["ticket-1"],
                "fallback_reasons": ["daemon fallback"],
            },
        }
        training = {
            "config": {"policy": "daemon-default"},
            "summary": {
                "iterations": 2,
                "median_iteration_ms": 20.0,
                "median_transfer_ms": 15.0,
                "median_compute_ms": 5.0,
                "median_gib_per_second": 4.0,
                "prefetch": {
                    "bytes": 96,
                    "bytes_completed": 96,
                    "direct_bytes": 64,
                    "relay_bytes": 32,
                    "direct_chunks": 2,
                    "relay_chunks": 1,
                    "decision_ids": ["prefetch-decision"],
                    "topology_snapshot_ids": ["topology-1"],
                    "ticket_ids": ["prefetch-ticket"],
                    "fallback_reasons": [],
                },
                "offload": {
                    "bytes": 24,
                    "bytes_completed": 24,
                    "direct_bytes": 16,
                    "relay_bytes": 8,
                    "direct_chunks": 1,
                    "relay_chunks": 1,
                    "decision_ids": ["offload-decision"],
                    "topology_snapshot_ids": ["topology-1"],
                    "ticket_ids": ["offload-ticket"],
                    "fallback_reasons": ["quota"],
                },
            },
        }

        model_metric = paper_validation.collect_model_metrics(model)[0]
        training_metric = paper_validation.collect_training_metrics(training)[0]

        self.assertEqual(model_metric["ttft_proxy_ms"], 12.5)
        self.assertEqual(model_metric["transfer_bytes"], 96)
        self.assertEqual(model_metric["decision_ids"], "decision-1")
        self.assertEqual(model_metric["ticket_ids"], "ticket-1")
        self.assertEqual(training_metric["iteration_ms"], 20.0)
        self.assertEqual(training_metric["transfer_bytes"], 120)
        self.assertEqual(training_metric["direct_bytes"], 80)
        self.assertEqual(training_metric["relay_chunks"], 2)
        self.assertEqual(
            training_metric["decision_ids"],
            "prefetch-decision,offload-decision",
        )
        self.assertEqual(training_metric["fallback_reason"], "quota")

    def test_collect_workload_metrics_reads_daemon_first_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "model-loading")
            paths["json"].write_text(
                json.dumps(
                    {
                        "config": {"policy": "daemon-default"},
                        "summary": {
                            "iterations": 1,
                            "median_load_ms": 1,
                            "median_gib_per_second": 2,
                            "bytes": 64,
                            "bytes_completed": 64,
                            "decision_ids": ["decision-1"],
                            "topology_snapshot_ids": ["topology-1"],
                            "ticket_ids": ["ticket-1"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            data, metrics = paper_validation.collect_workload_metrics("model-loading", paths)

        self.assertEqual(data["summary"]["bytes"], 64)
        self.assertEqual(metrics[0]["decision_ids"], "decision-1")

    def test_compact_summary_reports_policy_and_trace_ids(self) -> None:
        result = self._summary_result(
            "model-loading",
            [
                {
                    "workload": "model-loading",
                    "policy": "daemon-default",
                    "ttft_proxy_ms": 12.5,
                    "throughput_gib_s": 8.0,
                    "transfer_bytes": 96,
                    "decision_ids": "decision-1",
                    "topology_snapshot_ids": "topology-1",
                    "ticket_ids": "ticket-1",
                }
            ],
        )

        summary = paper_validation.compact_summary(result)

        self.assertIn("PAPER_VALIDATION_SUMMARY_BEGIN", summary)
        self.assertIn("session_id=session-1", summary)
        self.assertIn("policy=daemon-default", summary)
        self.assertIn("paper_metric workload=model-loading", summary)
        self.assertIn("decision_ids=decision-1", summary)
        self.assertIn("topology_snapshot_ids=topology-1", summary)
        self.assertIn("ticket_ids=ticket-1", summary)
        self.assertNotIn("paper_speedup", summary)

    def test_dry_run_builds_commands_without_reading_stale_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            model_paths = paper_validation.output_paths(output_dir, "model-loading")
            model_paths["json"].write_text(
                json.dumps({"summary": {"decision_ids": ["stale"]}}),
                encoding="utf-8",
            )
            args = make_args(output_dir=tmpdir, dry_run=True)

            with mock.patch.object(
                paper_validation,
                "run_command",
                side_effect=AssertionError("dry-run must not execute child commands"),
            ):
                result = paper_validation.run_validation(args)

        self.assertEqual([item["status"] for item in result["workloads"]], ["dry-run"] * 2)
        self.assertTrue(all(item["returncode"] == 0 for item in result["workloads"]))
        self.assertTrue(all(item["metrics"] == [] for item in result["workloads"]))
        self.assertEqual(result["workloads"][0]["data"], {})
        self.assertIn("--session-id", result["workloads"][0]["command"])
        self.assertIn("dry_run=True", paper_validation.compact_summary(result))

    def test_run_validation_rejects_missing_fresh_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = paper_validation.output_paths(output_dir, "model-loading")
            paths["json"].write_text(
                json.dumps({"summary": {"median_load_ms": 1}}),
                encoding="utf-8",
            )
            args = make_args(output_dir=tmpdir, workloads="model-loading")

            with mock.patch.object(
                paper_validation,
                "run_command",
                return_value=paper_validation.subprocess.CompletedProcess(
                    ["model-loading"],
                    0,
                    "",
                    "",
                ),
            ):
                result = paper_validation.run_validation(args)

        workload = result["workloads"][0]
        self.assertEqual(workload["status"], "missing-output")
        self.assertIn("missing_output_file", workload["validation_errors"])
        self.assertIn("missing_paper_metrics", workload["validation_errors"])
        self.assertEqual(workload["data"], {})
        self.assertFalse(paths["json"].exists())
        self.assertIn(
            "validation_errors=missing_output_file,missing_paper_metrics",
            paper_validation.compact_summary(result),
        )

    def test_keep_going_continues_after_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(
                output_dir=tmpdir,
                workloads="model-loading,training-offload",
                keep_going=True,
            )

            with mock.patch.object(
                paper_validation,
                "run_command",
                return_value=paper_validation.subprocess.CompletedProcess(
                    ["workload"],
                    0,
                    "",
                    "",
                ),
            ):
                result = paper_validation.run_validation(args)

        self.assertEqual(
            [item["workload"] for item in result["workloads"]],
            ["model-loading", "training-offload"],
        )
        self.assertEqual([item["status"] for item in result["workloads"]], ["missing-output"] * 2)

    def _summary_result(self, workload: str, metrics: list[dict]) -> dict:
        return {
            "config": {
                "session_id": "session-1",
                "job_id": "job-1",
                "cpu_buffer_id": "cpu-buffer",
                "gpu_buffer_id": "gpu-buffer",
                "workloads": [workload],
                "policy": "daemon-default",
                "output_dir": "out",
            },
            "workloads": [
                {
                    "workload": workload,
                    "status": "ok",
                    "returncode": 0,
                    "summary_path": "summary.txt",
                    "data_path": "data.json",
                    "validation_errors": [],
                    "metrics": metrics,
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
