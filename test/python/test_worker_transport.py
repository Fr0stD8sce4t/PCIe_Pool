from __future__ import annotations

import unittest

from turbobus.worker import (
    WorkerServiceEndpoint,
    WorkerServiceLoopbackTransport,
    WorkerServiceTransport,
)


class RecordingWorkerServiceEndpoint:
    def __init__(self) -> None:
        self.messages: list[str | bytes] = []
        self.observability_messages: list[str | bytes] = []

    def handle_message(self, message: str | bytes) -> str:
        self.messages.append(message)
        return "worker-response"

    def handle_observability_message(self, message: str | bytes) -> str:
        self.observability_messages.append(message)
        return "observability-response"


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
        observability_response = transport.handle_observability_message(
            b"worker-observability"
        )

        self.assertEqual(response, "worker-response")
        self.assertEqual(observability_response, "observability-response")
        self.assertEqual(endpoint.messages, ["worker-request"])
        self.assertEqual(endpoint.observability_messages, [b"worker-observability"])

    def test_loopback_transport_rejects_non_transport_endpoints(self) -> None:
        with self.assertRaisesRegex(TypeError, "endpoint"):
            WorkerServiceLoopbackTransport(object())


if __name__ == "__main__":
    unittest.main()
