from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import unittest

from turbobus.worker import (
    WorkerServiceEndpoint,
    WorkerServiceLoopbackTransport,
    WorkerServiceTransport,
    WorkerServiceUnixSocketTransport,
    decode_worker_response_envelope,
)


class RecordingWorkerServiceEndpoint:
    def __init__(self) -> None:
        self.messages: list[str | bytes] = []

    def handle_message(self, message: str | bytes) -> str:
        self.messages.append(message)
        return "worker-response"


class WorkerTransportTest(unittest.TestCase):
    def test_worker_service_endpoint_matches_transport_protocol(self) -> None:
        endpoint = WorkerServiceEndpoint(daemon_client=object())

        self.assertIsInstance(endpoint, WorkerServiceTransport)

    def test_loopback_transport_forwards_messages_without_modifying_payloads(
        self,
    ) -> None:
        endpoint = RecordingWorkerServiceEndpoint()
        transport = WorkerServiceLoopbackTransport(endpoint)

        response = transport.handle_message("worker-request")

        self.assertEqual(response, "worker-response")
        self.assertEqual(endpoint.messages, ["worker-request"])

    def test_loopback_transport_rejects_non_transport_endpoints(self) -> None:
        with self.assertRaisesRegex(TypeError, "endpoint"):
            WorkerServiceLoopbackTransport(object())

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_unix_socket_transport_round_trip_keeps_endpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "worker.sock")
            endpoint = WorkerServiceEndpoint(daemon_client=object())
            transport = WorkerServiceUnixSocketTransport(endpoint, socket_path)
            stop_event = threading.Event()
            thread = threading.Thread(
                target=transport.serve_forever,
                args=(stop_event,),
                daemon=True,
            )
            thread.start()

            for _ in range(100):
                if os.path.exists(socket_path):
                    break
                time.sleep(0.01)
            self.assertTrue(os.path.exists(socket_path))

            worker_response = transport.handle_message("{not-json")
            worker_payload = decode_worker_response_envelope(worker_response)
            self.assertFalse(worker_payload.ok)
            self.assertEqual(worker_payload.final_state, "parse_failed")

            stop_event.set()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
