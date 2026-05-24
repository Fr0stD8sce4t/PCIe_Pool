from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from turbobus.schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind


def load_example_module():
    path = Path(__file__).resolve().parents[3] / "examples" / "torch_tensor_fetch.py"
    spec = importlib.util.spec_from_file_location("torch_tensor_fetch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


example = load_example_module()


class PublicIntentExampleTest(unittest.TestCase):
    def test_example_builds_public_transfer_intent_without_physical_path_policy(self) -> None:
        args = example.build_parser().parse_args(
            [
                "--daemon-socket-path",
                "/tmp/turbobusd.sock",
                "--session-id",
                "session-1",
                "--job-id",
                "job-1",
                "--intent-id",
                "intent-1",
                "--source-buffer-id",
                "cpu-buffer",
                "--destination-buffer-id",
                "gpu-buffer",
                "--bytes",
                "4096",
            ]
        )

        intent = example.build_intent(args)

        self.assertIsInstance(intent, TransferIntent)
        self.assertEqual(intent.workload_kind, WorkloadKind.GENERIC)
        self.assertEqual(intent.direction, "h2d")
        self.assertEqual(intent.source_buffer_id, "cpu-buffer")
        self.assertEqual(intent.destination_buffer_id, "gpu-buffer")
        self.assertEqual(intent.total_bytes, 4096)
        self.assertEqual(intent.policy_hints, {})
        for physical_key in ("mode", "target_gpu", "relay_gpus", "path"):
            self.assertNotIn(physical_key, intent.policy_hints)
            self.assertNotIn(physical_key, intent.metadata)

    def test_example_receipt_line_reports_daemon_trace_ids(self) -> None:
        receipt = TransferReceipt(
            receipt_id="receipt-1",
            ticket_id="ticket-1",
            intent_id="intent-1",
            decision_id="decision-1",
            topology_snapshot_id="topology-1",
            job_id="job-1",
            session_id="session-1",
            state=TransferStatusState.COMPLETE,
            bytes_total=96,
            bytes_completed=96,
            path_stats=(
                {"kind": "direct", "bytes": 64},
                {"kind": "relay", "bytes": 32},
            ),
        )

        line = example.receipt_line(receipt)

        self.assertIn("decision_id=decision-1", line)
        self.assertIn("topology_snapshot_id=topology-1", line)
        self.assertIn("ticket_id=ticket-1", line)
        self.assertIn("direct_bytes=64", line)
        self.assertIn("relay_bytes=32", line)


if __name__ == "__main__":
    unittest.main()
