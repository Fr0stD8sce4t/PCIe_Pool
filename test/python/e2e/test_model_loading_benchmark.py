from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from turbobus.schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import model_loading  # noqa: E402


class ModelLoadingBenchmarkTest(unittest.TestCase):
    def test_run_benchmark_submits_model_weight_intents_and_reads_receipts(self) -> None:
        args = make_args(
            "--bucket-count",
            "2",
            "--bucket-bytes",
            "64",
            "--chunk-bytes",
            "32",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--policy",
            "paper-baseline",
            "--run-id",
            "run-1",
            "--wait-timeout-seconds",
            "2.5",
        )
        client = FakeClient()

        result = model_loading.run_benchmark(args, client=client)

        self.assertEqual(len(client.submitted), 3)
        self.assertEqual(
            client.waited,
            [
                ("model-load-run-1-warmup-0", 2.5),
                ("model-load-run-1-measure-0", 2.5),
                ("model-load-run-1-measure-1", 2.5),
            ],
        )
        for intent in client.submitted:
            self.assertIsInstance(intent, TransferIntent)
            self.assertEqual(intent.workload_kind, WorkloadKind.MODEL_WEIGHTS)
            self.assertEqual(intent.direction, "h2d")
            self.assertEqual(intent.total_bytes, 128)
            self.assertEqual(
                intent.ranges,
                (
                    {"src_offset": 0, "dst_offset": 0, "bytes": 64},
                    {"src_offset": 64, "dst_offset": 64, "bytes": 64},
                ),
            )
            self.assertEqual(intent.policy_hints, {})
            self.assertEqual(intent.metadata["policy"], "paper-baseline")
            self.assertEqual(intent.metadata["chunk_bytes"], 32)
            for physical_key in ("mode", "path", "relay_gpus", "target_gpu"):
                self.assertNotIn(physical_key, intent.policy_hints)
                self.assertNotIn(physical_key, intent.metadata)

        self.assertNotIn("modes", result)
        self.assertEqual(result["config"]["policy"], "paper-baseline")
        self.assertEqual(result["summary"]["iterations"], 2)
        self.assertEqual(result["summary"]["bytes"], 128)
        self.assertEqual(result["summary"]["bytes_completed"], 128)
        self.assertEqual(result["summary"]["direct_bytes"], 64)
        self.assertEqual(result["summary"]["relay_bytes"], 64)
        self.assertEqual(
            result["summary"]["decision_ids"],
            ["decision-model-load-run-1-measure-0", "decision-model-load-run-1-measure-1"],
        )
        self.assertEqual(result["samples"][0]["decision_id"], "decision-model-load-run-1-measure-0")
        self.assertEqual(result["samples"][0]["topology_snapshot_id"], "topology-1")
        self.assertEqual(result["samples"][0]["ticket_id"], "ticket-model-load-run-1-measure-0")

        summary = model_loading.compact_summary(result)
        self.assertIn("MODEL_LOAD_SUMMARY_BEGIN", summary)
        self.assertIn("model_load_receipt", summary)
        self.assertIn("decision_id=decision-model-load-run-1-measure-0", summary)
        self.assertIn("topology_snapshot_id=topology-1", summary)
        self.assertIn("ticket_id=ticket-model-load-run-1-measure-0", summary)

    def test_json_output_is_serializable_receipt_trace(self) -> None:
        args = make_args(
            "--bucket-count",
            "1",
            "--bucket-bytes",
            "96",
            "--iterations",
            "1",
            "--run-id",
            "json-run",
        )
        result = model_loading.run_benchmark(args, client=FakeClient())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.json"
            model_loading.write_json(str(output_path), result)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["samples"][0]["receipt"]["state"], "complete")
        self.assertEqual(data["samples"][0]["receipt"]["path_stats"][0]["kind"], "direct")
        self.assertEqual(data["samples"][0]["receipt"]["path_split"]["relay_bytes"], 48)

    def test_parser_rejects_old_application_side_path_options(self) -> None:
        parser = model_loading.build_parser()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base_cli() + ["--target-gpu", "0"])

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base_cli() + ["--mode", "pool"])

    def test_validate_args_requires_daemon_socket(self) -> None:
        args = model_loading.build_parser().parse_args(
            [
                "--session-id",
                "session-1",
                "--source-buffer-id",
                "cpu-buffer",
                "--destination-buffer-id",
                "gpu-buffer",
            ]
        )

        with self.assertRaisesRegex(ValueError, "--daemon-socket-path is required"):
            model_loading.validate_args(args)


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[TransferIntent] = []
        self.waited: list[tuple[str, float | None]] = []

    def submit_transfer_intent(self, intent: TransferIntent) -> TransferReceipt:
        self.submitted.append(intent)
        return make_receipt(intent, receipt_id=f"submitted-{intent.intent_id}")

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> TransferReceipt:
        self.waited.append((intent_id, timeout_seconds))
        intent = next(item for item in self.submitted if item.intent_id == intent_id)
        return make_receipt(intent, receipt_id=f"receipt-{intent_id}")


def make_receipt(intent: TransferIntent, *, receipt_id: str) -> TransferReceipt:
    direct_bytes = intent.total_bytes // 2
    relay_bytes = intent.total_bytes - direct_bytes
    return TransferReceipt(
        receipt_id=receipt_id,
        ticket_id=f"ticket-{intent.intent_id}",
        intent_id=intent.intent_id,
        decision_id=f"decision-{intent.intent_id}",
        topology_snapshot_id="topology-1",
        job_id=intent.job_id,
        session_id=intent.session_id,
        state=TransferStatusState.COMPLETE,
        bytes_total=intent.total_bytes,
        bytes_completed=intent.total_bytes,
        path_stats=(
            {"kind": "direct", "bytes": direct_bytes, "chunk_count": 1},
            {"kind": "relay", "bytes": relay_bytes, "chunk_count": 1},
        ),
        metadata={"fallback_reason": "daemon default"},
    )


def make_args(*extra: str):
    args = model_loading.build_parser().parse_args(base_cli() + list(extra))
    model_loading.validate_args(args)
    return args


def base_cli() -> list[str]:
    return [
        "--session-id",
        "session-1",
        "--job-id",
        "job-1",
        "--source-buffer-id",
        "cpu-buffer",
        "--destination-buffer-id",
        "gpu-buffer",
        "--daemon-socket-path",
        "/tmp/turbobusd.sock",
    ]


if __name__ == "__main__":
    unittest.main()
