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

import training_offload  # noqa: E402


class TrainingOffloadBenchmarkTest(unittest.TestCase):
    def test_run_benchmark_submits_prefetch_and_offload_intents(self) -> None:
        args = make_args(
            "--bucket-count",
            "4",
            "--active-buckets",
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

        result = training_offload.run_benchmark(args, client=client)

        self.assertEqual(len(client.submitted), 6)
        self.assertEqual(
            client.waited,
            [
                ("training-offload-run-1-warmup-0-prefetch", 2.5),
                ("training-offload-run-1-warmup-0-offload", 2.5),
                ("training-offload-run-1-measure-0-prefetch", 2.5),
                ("training-offload-run-1-measure-0-offload", 2.5),
                ("training-offload-run-1-measure-1-prefetch", 2.5),
                ("training-offload-run-1-measure-1-offload", 2.5),
            ],
        )

        first_prefetch = client.submitted[2]
        first_offload = client.submitted[3]
        second_prefetch = client.submitted[4]
        self.assert_intent_contract(first_prefetch, "prefetch", "h2d", "cpu-buffer", "gpu-buffer")
        self.assert_intent_contract(first_offload, "offload", "d2h", "gpu-buffer", "cpu-buffer")
        self.assertEqual(
            first_prefetch.ranges,
            (
                {"src_offset": 0, "dst_offset": 0, "bytes": 64},
                {"src_offset": 64, "dst_offset": 64, "bytes": 64},
            ),
        )
        self.assertEqual(
            second_prefetch.ranges,
            (
                {"src_offset": 128, "dst_offset": 128, "bytes": 64},
                {"src_offset": 192, "dst_offset": 192, "bytes": 64},
            ),
        )

        self.assertNotIn("modes", result)
        self.assertEqual(result["config"]["policy"], "paper-baseline")
        self.assertEqual(result["summary"]["iterations"], 2)
        self.assertEqual(result["summary"]["prefetch"]["bytes"], 128)
        self.assertEqual(result["summary"]["offload"]["bytes"], 128)
        self.assertEqual(result["summary"]["prefetch"]["direct_bytes"], 64)
        self.assertEqual(result["summary"]["offload"]["relay_bytes"], 64)
        self.assertEqual(
            result["summary"]["prefetch"]["decision_ids"],
            [
                "decision-training-offload-run-1-measure-0-prefetch",
                "decision-training-offload-run-1-measure-1-prefetch",
            ],
        )
        self.assertEqual(
            result["summary"]["offload"]["ticket_ids"],
            [
                "ticket-training-offload-run-1-measure-0-offload",
                "ticket-training-offload-run-1-measure-1-offload",
            ],
        )

        summary = training_offload.compact_summary(result)
        self.assertIn("TRAINING_OFFLOAD_SUMMARY_BEGIN", summary)
        self.assertIn("training_prefetch_receipt", summary)
        self.assertIn("training_offload_receipt", summary)
        self.assertIn("prefetch_decision_id=decision-training-offload-run-1-measure-0-prefetch", summary)
        self.assertIn("offload_ticket_id=ticket-training-offload-run-1-measure-0-offload", summary)

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
        result = training_offload.run_benchmark(args, client=FakeClient())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "training.json"
            training_offload.write_json(str(output_path), result)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["samples"][0]["prefetch"]["receipt"]["state"], "complete")
        self.assertEqual(data["samples"][0]["offload"]["receipt"]["path_split"]["relay_bytes"], 48)

    def test_parser_rejects_old_application_side_path_options(self) -> None:
        parser = training_offload.build_parser()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base_cli() + ["--target-gpu", "0"])

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base_cli() + ["--relay-gpus", "1"])

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base_cli() + ["--mode", "pool"])

    def test_validate_args_requires_daemon_socket_and_valid_active_buckets(self) -> None:
        args = training_offload.build_parser().parse_args(
            [
                "--session-id",
                "session-1",
                "--cpu-buffer-id",
                "cpu-buffer",
                "--gpu-buffer-id",
                "gpu-buffer",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--daemon-socket-path is required"):
            training_offload.validate_args(args)

        args = training_offload.build_parser().parse_args(
            base_cli() + ["--bucket-count", "2", "--active-buckets", "0"]
        )
        with self.assertRaisesRegex(ValueError, "--active-buckets"):
            training_offload.validate_args(args)

    def assert_intent_contract(
        self,
        intent: TransferIntent,
        operation: str,
        direction: str,
        source: str,
        destination: str,
    ) -> None:
        self.assertIsInstance(intent, TransferIntent)
        self.assertEqual(intent.workload_kind, WorkloadKind.TRAINING_STATE)
        self.assertEqual(intent.direction, direction)
        self.assertEqual(intent.source_buffer_id, source)
        self.assertEqual(intent.destination_buffer_id, destination)
        self.assertEqual(intent.total_bytes, 128)
        self.assertEqual(intent.policy_hints, {})
        self.assertEqual(intent.metadata["operation"], operation)
        self.assertEqual(intent.metadata["policy"], "paper-baseline")
        self.assertEqual(intent.metadata["chunk_bytes"], 32)
        for physical_key in ("mode", "path", "relay_gpus", "target_gpu"):
            self.assertNotIn(physical_key, intent.policy_hints)
            self.assertNotIn(physical_key, intent.metadata)


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
    args = training_offload.build_parser().parse_args(base_cli() + list(extra))
    training_offload.validate_args(args)
    return args


def base_cli() -> list[str]:
    return [
        "--session-id",
        "session-1",
        "--job-id",
        "job-1",
        "--cpu-buffer-id",
        "cpu-buffer",
        "--gpu-buffer-id",
        "gpu-buffer",
        "--daemon-socket-path",
        "/tmp/turbobusd.sock",
    ]


if __name__ == "__main__":
    unittest.main()
