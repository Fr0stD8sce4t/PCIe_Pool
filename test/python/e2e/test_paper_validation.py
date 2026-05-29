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
        "vllm_model": "vllm-model",
        "vllm_prompt": "",
        "vllm_prompt_repeat": 64,
        "vllm_second_prompt_suffix": " Italy",
        "vllm_prefix_key": "paper-validation-vllm-kv",
        "vllm_restore_blocks": 8,
        "vllm_matched_tokens": 128,
        "vllm_job_count": 1,
        "vllm_wait_timeout_seconds": None,
        "vllm_enforce_eager": False,
        "vllm_enable_multiproc_executor": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PaperValidationTest(unittest.TestCase):
    def test_selected_workloads_expands_daemon_first_targets(self) -> None:
        self.assertEqual(
            paper_validation.selected_workloads("all"),
            ["model-loading", "training-offload", "optimizer-offload", "vllm-kv"],
        )
        self.assertEqual(
            paper_validation.selected_workloads("model-loading,vllm-kv"),
            ["model-loading", "vllm-kv"],
        )
        with self.assertRaises(ValueError):
            paper_validation.selected_workloads("missing")

    def test_build_commands_use_registered_buffers_not_physical_paths(self) -> None:
        args = make_args()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "model-loading")
            optimizer_paths = paper_validation.output_paths(Path(tmpdir), "optimizer-offload")
            vllm_paths = paper_validation.output_paths(Path(tmpdir), "vllm-kv")
            model = paper_validation.build_model_loading_command(args, paths)
            training = paper_validation.build_training_offload_command(args, paths)
            optimizer = paper_validation.build_optimizer_offload_command(args, optimizer_paths)
            vllm = paper_validation.build_vllm_kv_command(args, vllm_paths)

        self.assertIn(str(BENCHMARKS / "model_loading.py"), model)
        self.assertIn(str(BENCHMARKS / "training_offload.py"), training)
        self.assertIn(str(BENCHMARKS / "training_offload.py"), optimizer)
        self.assertIn(str(BENCHMARKS.parent / "examples" / "vllm_turbobus_kv_connector.py"), vllm)
        self.assertIn("--session-id", model)
        self.assertIn("--source-buffer-id", model)
        self.assertIn("--destination-buffer-id", model)
        self.assertIn("--cpu-buffer-id", training)
        self.assertIn("--gpu-buffer-id", training)
        self.assertIn("--cpu-buffer-id", optimizer)
        self.assertIn("--gpu-buffer-id", optimizer)
        self.assertEqual(value_after(training, "--workload-kind"), "training_state")
        self.assertEqual(value_after(optimizer, "--workload-kind"), "optimizer_state")
        self.assertEqual(value_after(training, "--intent-prefix"), "training-offload")
        self.assertEqual(value_after(optimizer, "--intent-prefix"), "optimizer-offload")
        self.assertIn("--cpu-buffer-id", vllm)
        self.assertIn("--gpu-buffer-id", vllm)
        self.assertIn("--restore-enabled", vllm)
        self.assertIn("--daemon-socket-path", model)
        self.assertIn("--daemon-profile-max-age-seconds", training)
        self.assertIn("--daemon-socket-path", vllm)
        forbidden = {"--target-gpu", "--relay-gpus", "--mode", "--modes", "--min-pool-bytes"}
        self.assertTrue(forbidden.isdisjoint(model))
        self.assertTrue(forbidden.isdisjoint(training))
        self.assertTrue(forbidden.isdisjoint(optimizer))
        self.assertTrue(forbidden.isdisjoint(vllm))

    def test_build_multi_job_vllm_commands_use_distinct_identity_without_physical_paths(self) -> None:
        args = make_args(vllm_job_count=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "vllm-kv")
            commands = paper_validation.build_vllm_kv_commands(args, paths)

        self.assertEqual(len(commands), 2)
        job_ids = {value_after(command, "--job-id") for command in commands}
        session_ids = {value_after(command, "--session-id") for command in commands}
        cpu_buffer_ids = {value_after(command, "--cpu-buffer-id") for command in commands}
        gpu_buffer_ids = {value_after(command, "--gpu-buffer-id") for command in commands}
        prefix_keys = {value_after(command, "--prefix-key") for command in commands}
        log_outputs = {value_after(command, "--log-output") for command in commands}

        self.assertEqual(job_ids, {"job-1-job1", "job-1-job2"})
        self.assertEqual(session_ids, {"session-1-job1", "session-1-job2"})
        self.assertEqual(cpu_buffer_ids, {"cpu-buffer-job1", "cpu-buffer-job2"})
        self.assertEqual(gpu_buffer_ids, {"gpu-buffer-job1", "gpu-buffer-job2"})
        self.assertEqual(prefix_keys, {"paper-validation-vllm-kv-job1", "paper-validation-vllm-kv-job2"})
        self.assertEqual(len(log_outputs), 2)

        forbidden = {"--target-gpu", "--relay-gpus", "--mode", "--modes", "--min-pool-bytes"}
        for command in commands:
            self.assertTrue(forbidden.isdisjoint(command))

    def test_collect_model_and_training_metrics_from_daemon_receipts(self) -> None:
        model = {
            "config": {
                "policy": "daemon-default",
                "job_id": "model-job",
                "session_id": "model-session",
                "source_buffer_id": "cpu-buffer",
                "destination_buffer_id": "gpu-buffer",
                "workload_kind": "model_weights",
            },
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
            "config": {
                "policy": "daemon-default",
                "job_id": "training-job",
                "session_id": "training-session",
                "cpu_buffer_id": "cpu-buffer",
                "gpu_buffer_id": "gpu-buffer",
                "workload_kind": "training_state",
            },
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
        optimizer = {
            **training,
            "config": {
                **training["config"],
                "job_id": "optimizer-job",
                "workload_kind": "optimizer_state",
            },
        }

        model_metric = paper_validation.collect_model_metrics(model)[0]
        training_metric = paper_validation.collect_training_metrics(training)[0]
        optimizer_metric = paper_validation.collect_training_metrics(
            optimizer,
            workload="optimizer-offload",
        )[0]

        self.assertEqual(model_metric["ttft_proxy_ms"], 12.5)
        self.assertEqual(model_metric["job_id"], "model-job")
        self.assertEqual(model_metric["session_id"], "model-session")
        self.assertEqual(model_metric["source_buffer_id"], "cpu-buffer")
        self.assertEqual(model_metric["destination_buffer_id"], "gpu-buffer")
        self.assertEqual(model_metric["workload_kind"], "model_weights")
        self.assertEqual(model_metric["transfer_bytes"], 96)
        self.assertEqual(model_metric["decision_ids"], "decision-1")
        self.assertEqual(model_metric["ticket_ids"], "ticket-1")
        self.assertEqual(training_metric["iteration_ms"], 20.0)
        self.assertEqual(training_metric["job_id"], "training-job")
        self.assertEqual(training_metric["session_id"], "training-session")
        self.assertEqual(training_metric["cpu_buffer_id"], "cpu-buffer")
        self.assertEqual(training_metric["gpu_buffer_id"], "gpu-buffer")
        self.assertEqual(training_metric["workload"], "training-offload")
        self.assertEqual(training_metric["workload_kind"], "training_state")
        self.assertEqual(training_metric["transfer_bytes"], 120)
        self.assertEqual(training_metric["direct_bytes"], 80)
        self.assertEqual(training_metric["relay_chunks"], 2)
        self.assertEqual(
            training_metric["decision_ids"],
            "prefetch-decision,offload-decision",
        )
        self.assertEqual(training_metric["fallback_reason"], "quota")
        self.assertEqual(optimizer_metric["workload"], "optimizer-offload")
        self.assertEqual(optimizer_metric["job_id"], "optimizer-job")
        self.assertEqual(optimizer_metric["workload_kind"], "optimizer_state")

    def test_collect_vllm_kv_metrics_from_connector_summary(self) -> None:
        summary = {
            "vllm_kv_connector_config": {"model": "model"},
            "vllm_kv_connector_save": {
                "elapsed_ms": "10.0",
                "transfer_ms": "8.0",
                "bytes": "64",
                "direct_chunks": "1",
                "relay_chunks": "1",
                "direct_bytes": "32",
                "relay_bytes": "32",
                "receipt_ids": "save-r",
                "decision_ids": "save-d",
                "topology_snapshot_ids": "save-t",
                "ticket_ids": "save-ticket",
                "fallback_reason": "",
                "save_layer_count": "2",
                "save_layer_ranges": "4",
            },
            "vllm_kv_connector_restore": {
                "elapsed_ms": "20.0",
                "transfer_ms": "16.0",
                "total_ms": "22.0",
                "bytes": "128",
                "direct_chunks": "2",
                "relay_chunks": "1",
                "direct_bytes": "96",
                "relay_bytes": "32",
                "receipt_ids": "restore-r",
                "decision_ids": "restore-d",
                "topology_snapshot_ids": "restore-t",
                "ticket_ids": "restore-ticket",
                "fallback_reason": "quota",
                "layers": "2",
                "ranges": "4",
            },
            "vllm_kv_connector_result": {
                "prompt_tokens": "256",
                "shared_prefix": "True",
            },
        }

        metric = paper_validation.collect_vllm_kv_metrics(summary)[0]

        self.assertEqual(metric["workload"], "vllm-kv")
        self.assertEqual(metric["transfer_bytes"], 128)
        self.assertEqual(metric["direct_bytes"], 96)
        self.assertEqual(metric["relay_bytes"], 32)
        self.assertEqual(metric["decision_ids"], "restore-d")
        self.assertEqual(metric["save_decision_ids"], "save-d")
        self.assertEqual(metric["ticket_ids"], "restore-ticket")
        self.assertEqual(metric["save_ticket_ids"], "save-ticket")
        self.assertEqual(metric["fallback_reason"], "quota")
        self.assertEqual(metric["save_layer_count"], 2)
        self.assertEqual(metric["restore_layers"], 2)
        self.assertEqual(metric["prompt_tokens"], 256)

    def test_collect_workload_metrics_reads_daemon_first_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "model-loading")
            paths["json"].write_text(
                json.dumps(
                    {
                        "config": {
                            "policy": "daemon-default",
                            "job_id": "model-job",
                            "session_id": "model-session",
                            "source_buffer_id": "cpu-buffer",
                            "destination_buffer_id": "gpu-buffer",
                            "workload_kind": "model_weights",
                        },
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
        self.assertEqual(metrics[0]["workload_kind"], "model_weights")

    def test_phase6_workload_validation_requires_identity_and_workload_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "result.json"
            data_path.write_text("{}", encoding="utf-8")
            model_errors = paper_validation.phase6_workload_validation_errors(
                "model-loading",
                data_path,
                [
                    {
                        "workload": "model-loading",
                        "decision_ids": "decision-1",
                        "topology_snapshot_ids": "topology-1",
                        "ticket_ids": "ticket-1",
                        "job_id": "job-1",
                        "session_id": "session-1",
                        "source_buffer_id": "",
                        "destination_buffer_id": "gpu-buffer",
                        "workload_kind": "generic",
                    }
                ],
            )
            training_errors = paper_validation.phase6_workload_validation_errors(
                "training-offload",
                data_path,
                [
                    {
                        "workload": "training-offload",
                        "decision_ids": "decision-1",
                        "topology_snapshot_ids": "topology-1",
                        "ticket_ids": "ticket-1",
                        "job_id": "job-1",
                        "session_id": "session-1",
                        "cpu_buffer_id": "cpu-buffer",
                        "gpu_buffer_id": "gpu-buffer",
                        "workload_kind": "model_weights",
                    }
                ],
            )
            optimizer_errors = paper_validation.phase6_workload_validation_errors(
                "optimizer-offload",
                data_path,
                [
                    {
                        "workload": "optimizer-offload",
                        "decision_ids": "decision-1",
                        "topology_snapshot_ids": "topology-1",
                        "ticket_ids": "ticket-1",
                        "job_id": "job-1",
                        "session_id": "session-1",
                        "cpu_buffer_id": "cpu-buffer",
                        "gpu_buffer_id": "gpu-buffer",
                        "workload_kind": "training_state",
                    }
                ],
            )

        self.assertIn("missing_source_buffer_id", model_errors)
        self.assertIn("invalid_model_loading_workload_kind", model_errors)
        self.assertIn("invalid_training_offload_workload_kind", training_errors)
        self.assertIn("invalid_optimizer_offload_workload_kind", optimizer_errors)

    def test_collect_vllm_kv_metrics_reads_connector_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = paper_validation.output_paths(Path(tmpdir), "vllm-kv")
            paths["log"].write_text(vllm_log_text(), encoding="utf-8")

            data, metrics = paper_validation.collect_workload_metrics("vllm-kv", paths)

        self.assertEqual(data["vllm_kv_connector_restore"]["decision_ids"], "restore-d")
        self.assertEqual(metrics[0]["workload"], "vllm-kv")
        self.assertEqual(metrics[0]["decision_ids"], "restore-d")

    def test_compact_summary_reports_policy_and_trace_ids(self) -> None:
        result = self._summary_result(
            "model-loading",
            [
                {
                    "workload": "model-loading",
                    "policy": "daemon-default",
                    "job_id": "job-1",
                    "session_id": "session-1",
                    "source_buffer_id": "cpu-buffer",
                    "destination_buffer_id": "gpu-buffer",
                    "workload_kind": "model_weights",
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
        self.assertIn("workload_kind=model_weights", summary)
        self.assertIn("source_buffer_id=cpu-buffer", summary)
        self.assertIn("destination_buffer_id=gpu-buffer", summary)
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

        self.assertEqual([item["status"] for item in result["workloads"]], ["dry-run"] * 4)
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

    def test_run_validation_reports_training_and_optimizer_as_distinct_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(
                output_dir=tmpdir,
                workloads="training-offload,optimizer-offload",
                keep_going=True,
            )

            def fake_run(command):
                workload_kind = value_after(command, "--workload-kind")
                output_path = Path(value_after(command, "--json-output"))
                output_path.write_text(
                    json.dumps(training_output(workload_kind)),
                    encoding="utf-8",
                )
                return paper_validation.subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(paper_validation, "run_command", side_effect=fake_run):
                result = paper_validation.run_validation(args)

        self.assertEqual(
            [item["workload"] for item in result["workloads"]],
            ["training-offload", "optimizer-offload"],
        )
        self.assertEqual([item["status"] for item in result["workloads"]], ["ok", "ok"])
        self.assertEqual(
            [item["metrics"][0]["workload_kind"] for item in result["workloads"]],
            ["training_state", "optimizer_state"],
        )
        self.assertEqual(result["workloads"][0]["metrics"][0]["workload"], "training-offload")
        self.assertEqual(result["workloads"][1]["metrics"][0]["workload"], "optimizer-offload")
        summary = paper_validation.compact_summary(result)
        self.assertIn("paper_metric workload=training-offload", summary)
        self.assertIn("paper_metric workload=optimizer-offload", summary)
        self.assertIn("workload_kind=training_state", summary)
        self.assertIn("workload_kind=optimizer_state", summary)

    def test_run_validation_collects_vllm_kv_log_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(output_dir=tmpdir, workloads="vllm-kv")

            def fake_run(command):
                log_path = Path(command[command.index("--log-output") + 1])
                log_path.write_text(vllm_log_text(), encoding="utf-8")
                return paper_validation.subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(paper_validation, "run_command", side_effect=fake_run):
                result = paper_validation.run_validation(args)
                output_exists = Path(result["workloads"][0]["data_path"]).exists()

        workload = result["workloads"][0]
        self.assertEqual(workload["status"], "ok")
        self.assertEqual(workload["metrics"][0]["workload"], "vllm-kv")
        self.assertEqual(workload["metrics"][0]["decision_ids"], "restore-d")
        self.assertEqual(workload["metrics"][0]["save_decision_ids"], "save-d")
        self.assertEqual(workload["data"]["vllm_kv_connector_restore"]["decision_ids"], "restore-d")
        self.assertTrue(output_exists)
        summary = paper_validation.compact_summary(result)
        self.assertIn("paper_metric workload=vllm-kv", summary)
        self.assertIn("save_decision_ids=save-d", summary)

    def test_run_validation_collects_multi_job_vllm_kv_logs_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(output_dir=tmpdir, workloads="vllm-kv", vllm_job_count=2)
            run_calls = []

            def fake_run(commands):
                run_calls.append(commands)
                for index, command in enumerate(commands):
                    log_path = Path(command[command.index("--log-output") + 1])
                    job_id = command[command.index("--job-id") + 1]
                    session_id = command[command.index("--session-id") + 1]
                    cpu_buffer_id = command[command.index("--cpu-buffer-id") + 1]
                    gpu_buffer_id = command[command.index("--gpu-buffer-id") + 1]
                    log_path.write_text(
                        vllm_log_text(
                            job_id=job_id,
                            session_id=session_id,
                            cpu_buffer_id=cpu_buffer_id,
                            gpu_buffer_id=gpu_buffer_id,
                            save_decision_id=f"save-d-{index}",
                            restore_decision_id=f"restore-d-{index}",
                            restore_ticket_id=f"restore-ticket-{index}",
                        ),
                        encoding="utf-8",
                    )
                return [
                    paper_validation.subprocess.CompletedProcess(command, 0, "", "")
                    for command in commands
                ]

            with mock.patch.object(paper_validation, "run_commands_concurrent", side_effect=fake_run):
                result = paper_validation.run_validation(args)
                output_path = Path(result["workloads"][0]["data_path"])
                output_data = json.loads(output_path.read_text(encoding="utf-8"))

        workload = result["workloads"][0]
        self.assertEqual(workload["status"], "ok")
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(len(run_calls[0]), 2)
        self.assertEqual(len(workload["metrics"]), 2)
        self.assertEqual(
            {metric["job_id"] for metric in workload["metrics"]},
            {"job-1-job1", "job-1-job2"},
        )
        self.assertEqual(
            {metric["session_id"] for metric in workload["metrics"]},
            {"session-1-job1", "session-1-job2"},
        )
        self.assertEqual(
            {metric["cpu_buffer_id"] for metric in workload["metrics"]},
            {"cpu-buffer-job1", "cpu-buffer-job2"},
        )
        self.assertEqual(
            {metric["gpu_buffer_id"] for metric in workload["metrics"]},
            {"gpu-buffer-job1", "gpu-buffer-job2"},
        )
        self.assertEqual(output_data["vllm_kv_multi_job"]["job_count"], 2)
        self.assertEqual(len(output_data["jobs"]), 2)
        summary = paper_validation.compact_summary(result)
        self.assertIn("job_id=job-1-job1", summary)
        self.assertIn("job_id=job-1-job2", summary)
        self.assertIn("decision_ids=restore-d-0", summary)
        self.assertIn("decision_ids=restore-d-1", summary)

    def test_multi_job_vllm_kv_validation_fails_on_missing_per_job_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(output_dir=tmpdir, workloads="vllm-kv", vllm_job_count=2)

            def fake_run(commands):
                for index, command in enumerate(commands):
                    log_path = Path(command[command.index("--log-output") + 1])
                    job_id = command[command.index("--job-id") + 1]
                    session_id = command[command.index("--session-id") + 1]
                    cpu_buffer_id = command[command.index("--cpu-buffer-id") + 1]
                    gpu_buffer_id = command[command.index("--gpu-buffer-id") + 1]
                    decision_id = "" if index == 1 else f"restore-d-{index}"
                    log_path.write_text(
                        vllm_log_text(
                            job_id=job_id,
                            session_id=session_id,
                            cpu_buffer_id=cpu_buffer_id,
                            gpu_buffer_id=gpu_buffer_id,
                            restore_decision_id=decision_id,
                        ),
                        encoding="utf-8",
                    )
                return [
                    paper_validation.subprocess.CompletedProcess(command, 0, "", "")
                    for command in commands
                ]

            with mock.patch.object(paper_validation, "run_commands_concurrent", side_effect=fake_run):
                result = paper_validation.run_validation(args)

        workload = result["workloads"][0]
        self.assertEqual(workload["status"], "missing-metrics")
        self.assertIn("missing_daemon_trace", workload["validation_errors"])

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

    def test_vllm_kv_workload_requires_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(output_dir=tmpdir, workloads="vllm-kv", vllm_model="")

            with self.assertRaisesRegex(ValueError, "--vllm-model"):
                paper_validation.run_validation(args)

    def test_vllm_kv_workload_requires_daemon_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args(output_dir=tmpdir, workloads="vllm-kv", daemon_socket_path="")

            with self.assertRaisesRegex(ValueError, "--daemon-socket-path"):
                paper_validation.run_validation(args)

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


def value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def training_output(workload_kind: str) -> dict:
    return {
        "config": {
            "policy": "daemon-default",
            "job_id": "job-1",
            "session_id": "session-1",
            "cpu_buffer_id": "cpu-buffer",
            "gpu_buffer_id": "gpu-buffer",
            "workload_kind": workload_kind,
        },
        "summary": {
            "iterations": 1,
            "median_iteration_ms": 10.0,
            "median_transfer_ms": 8.0,
            "median_compute_ms": 2.0,
            "median_gib_per_second": 1.0,
            "prefetch": {
                "bytes": 64,
                "bytes_completed": 64,
                "direct_bytes": 32,
                "relay_bytes": 32,
                "direct_chunks": 1,
                "relay_chunks": 1,
                "decision_ids": [f"{workload_kind}-prefetch-decision"],
                "topology_snapshot_ids": ["topology-1"],
                "ticket_ids": [f"{workload_kind}-prefetch-ticket"],
                "fallback_reasons": [],
            },
            "offload": {
                "bytes": 64,
                "bytes_completed": 64,
                "direct_bytes": 32,
                "relay_bytes": 32,
                "direct_chunks": 1,
                "relay_chunks": 1,
                "decision_ids": [f"{workload_kind}-offload-decision"],
                "topology_snapshot_ids": ["topology-1"],
                "ticket_ids": [f"{workload_kind}-offload-ticket"],
                "fallback_reasons": [],
            },
        },
    }


def vllm_log_text(
    *,
    job_id: str = "job-1",
    session_id: str = "session-1",
    cpu_buffer_id: str = "cpu-buffer",
    gpu_buffer_id: str = "gpu-buffer",
    save_decision_id: str = "save-d",
    restore_decision_id: str = "restore-d",
    restore_ticket_id: str = "restore-ticket",
) -> str:
    return "\n".join(
        [
            "turbobus_kv_connector_event event=register_kv_caches layers=2",
            "turbobus_kv_connector_event event=save_layer request_id=req0 prefix_key=prefix receipt_ids=save-layer-r decision_ids=save-layer-d topology_snapshot_ids=save-layer-t ticket_ids=save-layer-ticket bytes=64 direct_bytes=32 relay_bytes=32 elapsed_ms=5 transfer_ms=5",
            "turbobus_kv_connector_event event=wait_for_save_done requests=1",
            f"turbobus_kv_connector_event event=save request_id=req0 prefix_key=prefix receipt_ids=save-r decision_ids={save_decision_id} topology_snapshot_ids=save-t ticket_ids=save-ticket bytes=64 direct_bytes=32 relay_bytes=32 direct_chunks=1 relay_chunks=1 elapsed_ms=10 transfer_ms=8 total_ms=12 save_layer_count=2 save_layer_ranges=4",
            f"turbobus_kv_connector_event event=restore request_id=req1 prefix_key=prefix receipt_ids=restore-r decision_ids={restore_decision_id} topology_snapshot_ids=restore-t ticket_ids={restore_ticket_id} bytes=128 direct_bytes=96 relay_bytes=32 direct_chunks=2 relay_chunks=1 elapsed_ms=20 transfer_ms=16 total_ms=22 layers=2 ranges=4 fallback_reason=quota",
            "turbobus_kv_connector_event event=start_load_done requests=1 restore_enabled=True elapsed_ms=21",
            "COPY_SUMMARY_BEGIN",
            f"vllm_kv_connector_config model=model job_id={job_id} session_id={session_id} cpu_buffer_id={cpu_buffer_id} gpu_buffer_id={gpu_buffer_id}",
            "vllm_kv_connector_scenario type=real_vllm_kv_transfer_connector boundary=KVConnectorBase_V1 entry=start_load_kv",
            f"vllm_kv_connector_save source_request=req0 source_blocks=8 blocks=8 bytes=64 elapsed_ms=10.0 transfer_ms=8.0 direct_chunks=1 relay_chunks=1 direct_bytes=32 relay_bytes=32 receipt_ids=save-r decision_ids={save_decision_id} topology_snapshot_ids=save-t ticket_ids=save-ticket fallback_reason= save_layer_count=2 save_layer_ranges=4",
            f"vllm_kv_connector_restore request_id=req1 prefix_key=prefix bytes=128 elapsed_ms=20.0 transfer_ms=16.0 total_ms=22.0 layers=2 ranges=4 direct_chunks=2 relay_chunks=1 direct_bytes=96 relay_bytes=32 receipt_ids=restore-r decision_ids={restore_decision_id} topology_snapshot_ids=restore-t ticket_ids={restore_ticket_id} fallback_reason=quota",
            "vllm_kv_connector_result source_request=req0 source_blocks=8 shared_prefix=True prompt_tokens=256",
            "COPY_SUMMARY_END",
        ]
    )
