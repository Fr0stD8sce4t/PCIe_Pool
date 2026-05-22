from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import paper_validation  # noqa: E402


def make_args(**overrides):
    values = {
        "target_gpu": 6,
        "relay_gpus": "5",
        "mode": "pool",
        "chunk_bytes": 4,
        "profile_bytes": 8,
        "min_pool_bytes": 6,
        "warmup": 0,
        "iterations": 1,
        "bucket_count": 4,
        "active_buckets": None,
        "bucket_bytes": 8,
        "storage_layout": "packed",
        "compute_elements": 16,
        "compute_iterations": 2,
        "vllm_model": "model",
        "vllm_prompt_repeat": 8,
        "vllm_restore_blocks_list": "2",
        "vllm_tokens_per_block": 16,
        "vllm_enforce_eager": True,
        "vllm_enable_multiproc_executor": False,
        "vllm_no_map_physical_gpus": False,
        "force_profile": True,
        "verify": True,
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
    def test_selected_workloads_expands_all_and_rejects_unknown(self) -> None:
        self.assertEqual(
            paper_validation.selected_workloads("all"),
            ["model-loading", "vllm-kv", "training-offload"],
        )
        self.assertEqual(paper_validation.selected_workloads("vllm-kv"), ["vllm-kv"])
        with self.assertRaises(ValueError):
            paper_validation.selected_workloads("missing")

    def test_build_commands_use_existing_workload_entry_points(self) -> None:
        args = make_args(mode="all")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "model-loading")
            model = paper_validation.build_model_loading_command(args, paths)
            training = paper_validation.build_training_offload_command(args, paths)
            vllm = paper_validation.build_vllm_kv_command(args, paths)

        self.assertIn(str(BENCHMARKS / "model_loading.py"), model)
        self.assertIn(str(BENCHMARKS / "training_offload.py"), training)
        self.assertIn(str(BENCHMARKS.parent / "examples" / "vllm_turbobus_kv_connector_sweep.py"), vllm)
        self.assertIn("--daemon-socket-path", model)
        self.assertIn("--daemon-profile-max-age-seconds", training)
        self.assertIn("--cases-json-output", vllm)
        self.assertIn("auto,direct,relay,pool", vllm)

    def test_collect_model_and_training_metrics(self) -> None:
        model = {
            "modes": {
                "pool": {
                    "summary": {
                        "median_load_ms": 12.5,
                        "median_gib_per_second": 8.0,
                        "direct_bytes": 64,
                        "relay_bytes": 32,
                        "direct_chunks": 2,
                        "relay_chunks": 1,
                    },
                    "daemon_reservation": {"daemon_reservation_status": "granted"},
                    "last_auto_decision": {"auto_reason": "pool_speedup"},
                }
            }
        }
        training = {
            "modes": {
                "pool": {
                    "summary": {
                        "median_iteration_ms": 20.0,
                        "median_transfer_ms": 15.0,
                        "median_compute_ms": 5.0,
                        "median_gib_per_second": 4.0,
                        "prefetch": {
                            "direct_bytes": 64,
                            "relay_bytes": 32,
                            "direct_chunks": 2,
                            "relay_chunks": 1,
                        },
                        "offload": {
                            "direct_bytes": 16,
                            "relay_bytes": 8,
                            "direct_chunks": 1,
                            "relay_chunks": 1,
                        },
                    },
                    "prefetch_daemon_reservation": {"daemon_reservation_status": "granted"},
                    "last_auto_decision": {"auto_reason": "explicit_transfer_mode"},
                }
            }
        }

        model_metric = paper_validation.collect_model_metrics(model)[0]
        training_metric = paper_validation.collect_training_metrics(training)[0]

        self.assertEqual(model_metric["ttft_proxy_ms"], 12.5)
        self.assertEqual(model_metric["transfer_bytes"], 96)
        self.assertEqual(model_metric["fallback_reason"], "pool_speedup")
        self.assertEqual(training_metric["iteration_ms"], 20.0)
        self.assertEqual(training_metric["transfer_bytes"], 120)
        self.assertEqual(training_metric["relay_chunks"], 2)

    def test_collect_vllm_metrics_and_summary_lines(self) -> None:
        metrics = paper_validation.collect_vllm_metrics(
            [
                {
                    "mode": "pool",
                    "restore_blocks": "8",
                    "matched_tokens": "128",
                    "start_load_ms": "22",
                    "save_ms": "11",
                    "restore_ms": "20",
                    "restore_transfer_ms": "18",
                    "restore_gib_s": "50",
                    "bytes": "1073741824",
                    "direct_chunks": "1",
                    "relay_chunks": "2",
                    "daemon_reservation_status": "granted",
                    "auto_reason": "pool_speedup_1.500",
                }
            ]
        )
        result = {
            "config": {
                "target_gpu": 6,
                "relay_gpus": "5",
                "workloads": ["vllm-kv"],
                "mode": "pool",
                "output_dir": "out",
            },
            "workloads": [
                {
                    "workload": "vllm-kv",
                    "status": "ok",
                    "returncode": 0,
                    "summary_path": "summary.txt",
                    "data_path": "cases.json",
                    "metrics": metrics,
                }
            ],
        }

        summary = paper_validation.compact_summary(result)

        self.assertEqual(metrics[0]["restore_latency_ms"], 20.0)
        self.assertEqual(metrics[0]["throughput_gib_s"], 50.0)
        self.assertIn("PAPER_VALIDATION_SUMMARY_BEGIN", summary)
        self.assertIn("workload=vllm-kv", summary)
        self.assertIn("restore_latency_ms=20.000", summary)
        self.assertIn("fallback_reason=pool_speedup_1.500", summary)

    def test_model_loading_speedup_summary_line(self) -> None:
        metrics = [
            {
                "workload": "model-loading",
                "mode": "direct",
                "ttft_proxy_ms": 20.0,
                "throughput_gib_s": 4.0,
            },
            {
                "workload": "model-loading",
                "mode": "relay",
                "ttft_proxy_ms": 18.0,
                "throughput_gib_s": 5.0,
            },
            {
                "workload": "model-loading",
                "mode": "pool",
                "ttft_proxy_ms": 10.0,
                "throughput_gib_s": 8.0,
            },
            {
                "workload": "model-loading",
                "mode": "auto",
                "ttft_proxy_ms": 12.5,
                "throughput_gib_s": 7.5,
            },
        ]
        result = self._summary_result("model-loading", metrics)

        summary = paper_validation.compact_summary(result)

        self.assertIn("paper_metric workload=model-loading mode=direct", summary)
        self.assertIn("paper_speedup workload=model-loading", summary)
        self.assertIn("direct_over_pool_ttft_proxy=2.000", summary)
        self.assertIn("relay_over_auto_ttft_proxy=1.440", summary)
        self.assertIn("pool_over_direct_throughput=2.000", summary)
        self.assertIn("auto_over_relay_throughput=1.500", summary)

    def test_vllm_speedup_summary_line_groups_restore_blocks(self) -> None:
        metrics = [
            {
                "workload": "vllm-kv",
                "mode": "direct",
                "restore_blocks": 8,
                "matched_tokens": 128,
                "restore_latency_ms": 30.0,
                "restore_transfer_ms": 24.0,
                "throughput_gib_s": 3.0,
            },
            {
                "workload": "vllm-kv",
                "mode": "relay",
                "restore_blocks": 8,
                "matched_tokens": 128,
                "restore_latency_ms": 27.0,
                "restore_transfer_ms": 21.0,
                "throughput_gib_s": 4.0,
            },
            {
                "workload": "vllm-kv",
                "mode": "pool",
                "restore_blocks": 8,
                "matched_tokens": 128,
                "restore_latency_ms": 15.0,
                "restore_transfer_ms": 12.0,
                "throughput_gib_s": 6.0,
            },
            {
                "workload": "vllm-kv",
                "mode": "auto",
                "restore_blocks": 8,
                "matched_tokens": 128,
                "restore_latency_ms": 18.0,
                "restore_transfer_ms": 14.0,
                "throughput_gib_s": 5.0,
            },
        ]
        result = self._summary_result("vllm-kv", metrics)

        summary = paper_validation.compact_summary(result)

        self.assertIn(
            "paper_speedup workload=vllm-kv restore_blocks=8 matched_tokens=128",
            summary,
        )
        self.assertIn("direct_over_pool_restore_latency=2.000", summary)
        self.assertIn("relay_over_auto_restore_latency=1.500", summary)
        self.assertIn("direct_over_pool_restore_transfer=2.000", summary)
        self.assertIn("pool_over_relay_throughput=1.500", summary)

    def test_training_speedup_summary_line_and_missing_modes_are_na(self) -> None:
        metrics = [
            {
                "workload": "training-offload",
                "mode": "direct",
                "iteration_ms": 40.0,
                "transfer_ms": 36.0,
                "throughput_gib_s": 4.0,
            },
            {
                "workload": "training-offload",
                "mode": "pool",
                "iteration_ms": 20.0,
                "transfer_ms": 18.0,
                "throughput_gib_s": 8.0,
            },
        ]
        result = self._summary_result("training-offload", metrics)

        summary = paper_validation.compact_summary(result)

        self.assertIn("paper_metric workload=training-offload mode=direct", summary)
        self.assertIn("direct_over_pool_iteration=2.000", summary)
        self.assertIn("direct_over_pool_transfer=2.000", summary)
        self.assertIn("pool_over_direct_throughput=2.000", summary)
        self.assertIn("relay_over_pool_iteration=NA", summary)
        self.assertIn("direct_over_auto_transfer=NA", summary)
        self.assertIn("auto_over_relay_throughput=NA", summary)

    def test_collect_workload_metrics_reads_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "vllm-kv")
            paths["cases_json"].write_text(
                json.dumps([{"mode": "pool", "restore_ms": "20"}]),
                encoding="utf-8",
            )

            data, metrics = paper_validation.collect_workload_metrics("vllm-kv", paths)

        self.assertEqual(data[0]["mode"], "pool")
        self.assertEqual(metrics[0]["restore_latency_ms"], 20.0)

    def test_dry_run_builds_commands_without_reading_stale_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            model_paths = paper_validation.output_paths(output_dir, "model-loading")
            model_paths["json"].write_text(
                json.dumps(
                    {
                        "modes": {
                            "pool": {
                                "summary": {
                                    "median_load_ms": 1,
                                    "median_gib_per_second": 2,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            vllm_paths = paper_validation.output_paths(output_dir, "vllm-kv")
            vllm_paths["cases_json"].write_text(
                json.dumps([{"mode": "pool", "restore_ms": "20"}]),
                encoding="utf-8",
            )
            args = make_args(output_dir=tmpdir, dry_run=True)

            with mock.patch.object(
                paper_validation,
                "run_command",
                side_effect=AssertionError("dry-run must not execute child commands"),
            ):
                result = paper_validation.run_validation(args)

        self.assertEqual([item["status"] for item in result["workloads"]], ["dry-run"] * 3)
        self.assertTrue(all(item["returncode"] == 0 for item in result["workloads"]))
        self.assertTrue(all(item["metrics"] == [] for item in result["workloads"]))
        self.assertEqual(result["workloads"][0]["data"], {})
        self.assertEqual(result["workloads"][1]["data"], [])
        self.assertIn("--daemon-socket-path", result["workloads"][0]["command"])
        self.assertIn("dry_run=True", paper_validation.compact_summary(result))

    def test_run_validation_rejects_missing_fresh_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = paper_validation.output_paths(output_dir, "model-loading")
            paths["json"].write_text(
                json.dumps({"modes": {"pool": {"summary": {"median_load_ms": 1}}}}),
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
        self.assertIn("validation_errors=missing_output_file,missing_paper_metrics", paper_validation.compact_summary(result))

    def test_keep_going_continues_after_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
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
                "target_gpu": 6,
                "relay_gpus": "5",
                "workloads": [workload],
                "mode": "all",
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
