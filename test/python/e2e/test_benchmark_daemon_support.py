from __future__ import annotations

import argparse
import unittest

import sys
from pathlib import Path

from turbobus.schema import (
    TransferReceipt,
    TransferStatusState,
    WorkloadKind,
)

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from daemon_support import (  # noqa: E402
    add_daemon_options,
    benchmark_job_id,
    collect_daemon_reservation_info,
    daemon_profile_line,
    daemon_reservation_line,
    make_benchmark_transfer_intent,
    receipt_to_trace,
    receipt_trace_line,
    runtime_options_kwargs,
)


class BenchmarkDaemonSupportTest(unittest.TestCase):
    def test_add_daemon_options_and_runtime_kwargs(self) -> None:
        parser = argparse.ArgumentParser()
        add_daemon_options(parser)

        args = parser.parse_args(
            [
                "--daemon-socket-path",
                "/tmp/turbobusd.sock",
                "--daemon-max-inflight-chunks",
                "3",
            ]
        )

        self.assertEqual(
            runtime_options_kwargs(args),
            {
                "daemon_socket_path": "/tmp/turbobusd.sock",
                "daemon_max_inflight_chunks": 3,
                "daemon_profile_max_age_seconds": 3600.0,
            },
        )

    def test_daemon_summary_lines_are_stable(self) -> None:
        profile = {
            "daemon_profile_status": "hit",
            "daemon_profile_bytes": 4096,
        }
        reservation = {
            "daemon_reservation_status": "granted",
            "daemon_reserved_relays": "5",
            "daemon_reserved_chunks_per_relay": 32,
        }

        self.assertEqual(
            daemon_profile_line(profile),
            "daemon_profile daemon_profile_bytes=4096 daemon_profile_status=hit",
        )
        self.assertEqual(
            daemon_reservation_line(reservation),
            "daemon_reservation daemon_reservation_status=granted "
            "daemon_reserved_chunks_per_relay=32 daemon_reserved_relays=5",
        )

    def test_collect_daemon_reservation_info_uses_first_handle_with_info(self) -> None:
        first = type("Handle", (), {"daemon_reservation_info": {}})()
        second = type(
            "Handle",
            (),
            {"daemon_reservation_info": {"daemon_reservation_status": "granted"}},
        )()

        self.assertEqual(
            collect_daemon_reservation_info([first, second]),
            {"daemon_reservation_status": "granted"},
        )

    def test_benchmark_intent_uses_workload_policy_not_physical_path(self) -> None:
        intent = make_benchmark_transfer_intent(
            intent_id="intent-1",
            workload_kind=WorkloadKind.MODEL_WEIGHTS,
            job_id=benchmark_job_id("model-loading"),
            session_id="session-1",
            source_buffer_id="cpu-buffer",
            destination_buffer_id="gpu-buffer",
            direction="h2d",
            total_bytes=64,
            ranges=[{"src_offset": 0, "dst_offset": 0, "bytes": 64}],
            policy_hints={"latency_sensitive": True},
            metadata={"chunk_bytes": 16},
        )

        self.assertEqual(intent.workload_kind, WorkloadKind.MODEL_WEIGHTS)
        self.assertEqual(intent.metadata["chunk_bytes"], 16)
        self.assertEqual(intent.policy_hints, {"latency_sensitive": True})
        for physical_key in ("mode", "relay_gpus", "target_gpu", "path"):
            self.assertNotIn(physical_key, intent.policy_hints)

    def test_benchmark_intent_rejects_physical_policy_hints(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical paths"):
            make_benchmark_transfer_intent(
                workload_kind=WorkloadKind.MODEL_WEIGHTS,
                job_id="job-1",
                session_id="session-1",
                source_buffer_id="cpu-buffer",
                destination_buffer_id="gpu-buffer",
                direction="h2d",
                total_bytes=64,
                ranges=[{"src_offset": 0, "dst_offset": 0, "bytes": 64}],
                policy_hints={"mode": "pool"},
            )

    def test_receipt_trace_reports_paper_parity_ids_and_path_split(self) -> None:
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
                {"kind": "direct", "bytes": 64, "chunk_count": 2},
                {"kind": "relay", "bytes": 32, "chunk_count": 1},
            ),
            metadata={"fallback_reason": "daemon profile miss"},
        )

        trace = receipt_to_trace(receipt)
        line = receipt_trace_line(receipt, prefix="model_load_receipt")

        self.assertEqual(trace["direct_bytes"], 64)
        self.assertEqual(trace["relay_bytes"], 32)
        self.assertEqual(trace["direct_chunks"], 2)
        self.assertEqual(trace["relay_chunks"], 1)
        self.assertIn("decision_id=decision-1", line)
        self.assertIn("topology_snapshot_id=topology-1", line)
        self.assertIn("ticket_id=ticket-1", line)
        self.assertIn("fallback_reason=daemon_profile_miss", line)


if __name__ == "__main__":
    unittest.main()
