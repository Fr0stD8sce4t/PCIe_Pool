from __future__ import annotations

import argparse
import unittest

import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from daemon_support import (  # noqa: E402
    add_daemon_options,
    collect_daemon_reservation_info,
    daemon_profile_line,
    daemon_reservation_line,
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


if __name__ == "__main__":
    unittest.main()
