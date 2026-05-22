from __future__ import annotations

import argparse
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from daemon_smoke import (  # noqa: E402
    build_parser,
    build_client_command,
    build_daemon_command,
    build_smoke_result,
    collect_summary_fields,
    print_smoke_summary,
)


class DaemonSmokeTest(unittest.TestCase):
    def test_build_daemon_command_uses_relay_and_quota_args(self) -> None:
        args = argparse.Namespace(
            relay_gpus="5",
            daemon_max_sessions_per_relay=2,
            daemon_max_inflight_chunks_per_relay=128,
        )

        command = build_daemon_command(args, "/tmp/turbobusd.sock")

        self.assertEqual(command[:3], [sys.executable, "-m", "turbobus.daemon"])
        self.assertIn("--socket-path", command)
        self.assertIn("/tmp/turbobusd.sock", command)
        self.assertIn("--relay-gpus", command)
        self.assertIn("5", command)
        self.assertIn("--max-inflight-chunks-per-relay", command)
        self.assertIn("128", command)

    def test_build_parser_accepts_daemon_socket_and_quota_arguments(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--target-gpu",
                "6",
                "--relay-gpus",
                "5",
                "--daemon-socket-path",
                "/tmp/turbobusd.sock",
                "--daemon-max-inflight-chunks",
                "128",
            ]
        )

        self.assertEqual(args.daemon_socket_path, "/tmp/turbobusd.sock")
        self.assertEqual(args.daemon_max_inflight_chunks, 128)

    def test_build_client_command_uses_bandwidth_client_options(self) -> None:
        args = argparse.Namespace(
            target_gpu=6,
            relay_gpus="5",
            bytes=64,
            chunk_bytes=4,
            profile_bytes=8,
            min_pool_bytes=6,
            mode="pool",
            iterations=1,
            warmup=0,
            verify=True,
            force_profile_first=True,
            daemon_max_inflight_chunks=128,
            bucket_count=4,
            bucket_bytes=8,
            storage_layout="packed",
        )

        command = build_client_command(args, "/tmp/turbobusd.sock", "bandwidth", 1)

        self.assertIn(str(BENCHMARKS / "bandwidth_pool.py"), command)
        self.assertIn("--daemon-socket-path", command)
        self.assertIn("/tmp/turbobusd.sock", command)
        self.assertIn("--force-profile", command)
        self.assertIn("--verify", command)

    def test_build_client_command_supports_training_offload(self) -> None:
        args = argparse.Namespace(
            target_gpu=6,
            relay_gpus="5",
            bytes=64,
            chunk_bytes=4,
            profile_bytes=8,
            min_pool_bytes=6,
            mode="pool",
            iterations=1,
            warmup=0,
            verify=True,
            force_profile_first=False,
            daemon_max_inflight_chunks=128,
            bucket_count=4,
            bucket_bytes=8,
            storage_layout="packed",
        )

        command = build_client_command(args, "/tmp/turbobusd.sock", "training-offload", 1)

        self.assertIn(str(BENCHMARKS / "training_offload.py"), command)
        self.assertIn("--bucket-count", command)
        self.assertIn("--bucket-bytes", command)

    def test_build_smoke_result_reports_publish_hit_and_reservation(self) -> None:
        first_output = """
COPY_SUMMARY_BEGIN
daemon_profile daemon_profile_status=miss phase=initial
daemon_profile daemon_profile_status=published daemon_profile_bytes=8388608 phase=after_profile
daemon_reservation daemon_reservation_status=granted daemon_reserved_relays=5 daemon_reserved_chunks_per_relay=32 mode=pool
COPY_SUMMARY_END
""".strip()
        second_output = """
COPY_SUMMARY_BEGIN
daemon_profile daemon_profile_status=hit daemon_profile_bytes=8388608 phase=initial
daemon_profile daemon_profile_status=published daemon_profile_bytes=8388608 phase=after_profile
daemon_reservation daemon_reservation_status=granted daemon_reserved_relays=5 daemon_reserved_chunks_per_relay=32 mode=pool
COPY_SUMMARY_END
""".strip()

        result = build_smoke_result(first_output, second_output, "bandwidth")
        first = result["clients"]["first"]
        second = result["clients"]["second"]

        self.assertEqual(first["daemon_profile"]["daemon_profile_status"], "published")
        self.assertEqual(second["daemon_profile"]["daemon_profile_status"], "hit")
        self.assertEqual(first["daemon_reservation"]["daemon_reservation_status"], "granted")

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_smoke_summary(result)
        output = buffer.getvalue()
        self.assertIn("daemon_smoke_client client=first", output)
        self.assertIn("daemon_profile_status=published", output)
        self.assertIn("daemon_profile_status=hit", output)
        self.assertIn("daemon_reservation_status=granted", output)

    def test_collect_summary_fields_ignores_unrelated_lines(self) -> None:
        fields = collect_summary_fields(
            """
noise
daemon_profile daemon_profile_status=hit phase=initial
other line
daemon_reservation daemon_reservation_status=denied mode=pool
""".strip()
        )

        self.assertEqual(len(fields["daemon_profiles"]), 1)
        self.assertEqual(len(fields["daemon_reservations"]), 1)


if __name__ == "__main__":
    unittest.main()
