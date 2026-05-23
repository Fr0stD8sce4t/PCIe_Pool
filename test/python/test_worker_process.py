from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from threading import Event
from unittest.mock import Mock, patch

from turbobus.worker import (
    WorkerServiceEndpoint,
    WorkerServiceObservabilityRequestEnvelope,
    WorkerServiceUnixSocketTransport,
    decode_worker_observability_snapshot,
    decode_worker_response_envelope,
    encode_worker_observability_request_envelope,
)
from turbobus.worker.process import (
    build_worker_helper_transport,
    main,
    run_worker_helper_process,
)


class WorkerProcessTest(unittest.TestCase):
    def test_build_worker_helper_transport_wires_daemon_and_worker_sockets(self) -> None:
        transport = build_worker_helper_transport(
            "/tmp/turbobusd.sock",
            "/tmp/turbobus-worker.sock",
        )

        self.assertIsInstance(transport, WorkerServiceUnixSocketTransport)
        self.assertIsInstance(transport.endpoint, WorkerServiceEndpoint)
        self.assertEqual(transport.socket_path, "/tmp/turbobus-worker.sock")
        self.assertEqual(
            transport.endpoint.service.transfer_client.authorizer.daemon_client.socket_path,
            "/tmp/turbobusd.sock",
        )

    def test_run_worker_helper_process_uses_the_transport(self) -> None:
        stop_event = Event()
        fake_transport = Mock()

        with patch(
            "turbobus.worker.process.build_worker_helper_transport",
            return_value=fake_transport,
        ) as build:
            run_worker_helper_process(
                "/tmp/turbobusd.sock",
                "/tmp/turbobus-worker.sock",
                stop_event=stop_event,
                max_requests=2,
            )

        build.assert_called_once_with(
            "/tmp/turbobusd.sock",
            "/tmp/turbobus-worker.sock",
        )
        fake_transport.serve_forever.assert_called_once_with(
            stop_event=stop_event,
            max_requests=2,
        )

    def test_main_parses_args_and_runs_helper_process(self) -> None:
        with patch("turbobus.worker.process.run_worker_helper_process") as run:
            exit_code = main(
                [
                    "--daemon-socket-path",
                    "/tmp/turbobusd.sock",
                    "--socket-path",
                    "/tmp/turbobus-worker.sock",
                    "--max-requests",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        run.assert_called_once_with(
            "/tmp/turbobusd.sock",
            "/tmp/turbobus-worker.sock",
            max_requests=2,
        )

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_worker_module_subprocess_serves_worker_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker_socket = os.path.join(tmpdir, "worker.sock")
            daemon_socket = os.path.join(tmpdir, "daemon.sock")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "turbobus.worker",
                    "--daemon-socket-path",
                    daemon_socket,
                    "--socket-path",
                    worker_socket,
                    "--max-requests",
                    "2",
                ],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self._wait_for_socket(worker_socket)

                worker_response = _send_worker_message(worker_socket, "{not-json")
                worker_payload = decode_worker_response_envelope(worker_response)
                self.assertFalse(worker_payload.ok)
                self.assertEqual(worker_payload.final_state, "parse_failed")

                observability_request = encode_worker_observability_request_envelope(
                    WorkerServiceObservabilityRequestEnvelope()
                )
                observability_response = _send_worker_message(
                    worker_socket,
                    observability_request,
                )
                observability = decode_worker_observability_snapshot(
                    observability_response
                )

                self.assertEqual(observability["describe"]["total_requests"], 1)
                self.assertEqual(
                    observability["describe"]["observability_total_requests"],
                    0,
                )
                self.assertEqual(observability["events"][0]["final_state"], "parse_failed")
                self.assertEqual(observability["health"]["status"], "degraded")

                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout, "")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    def _wait_for_socket(self, socket_path: str) -> None:
        for _ in range(100):
            if os.path.exists(socket_path):
                return
            time.sleep(0.01)
        self.fail(f"worker socket was not created: {socket_path}")


def _send_worker_message(socket_path: str, message: str | bytes) -> str:
    payload = message if isinstance(message, bytes) else message.encode("utf-8")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(socket_path)
        client.sendall(payload + b"\n")
        data = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        return data.partition(b"\n")[0].decode("utf-8")
    finally:
        client.close()


if __name__ == "__main__":
    unittest.main()
