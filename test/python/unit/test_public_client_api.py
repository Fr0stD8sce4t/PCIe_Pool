from __future__ import annotations

from dataclasses import asdict
import importlib
import unittest

import turbobus
from turbobus.api import TurboBusClient
from turbobus.daemon.client import TurboBusDaemonClient
from turbobus.schema import (
    DaemonRequest,
    DaemonResponse,
    RequestType,
    TransferIntent,
    TransferReceipt,
    TransferStatusState,
    WorkloadKind,
)


class PublicClientApiTest(unittest.TestCase):
    def test_submit_sends_transfer_intent_and_returns_receipt(self) -> None:
        daemon = FakeDaemon()
        intent = make_intent(policy_hints={"latency_sensitive": True})

        receipt = TurboBusClient(daemon=daemon).submit_transfer_intent(intent)

        self.assertIsInstance(receipt, TransferReceipt)
        self.assertEqual(receipt.intent_id, intent.intent_id)
        self.assertEqual(receipt.bytes_completed, intent.total_bytes)
        self.assertEqual(daemon.submitted_intents, [intent])
        submitted_payload = asdict(daemon.submitted_intents[0])
        self.assertEqual(submitted_payload["policy_hints"], {"latency_sensitive": True})
        for physical_key in ("mode", "path", "relay_gpu", "relay_gpus", "target_gpu"):
            self.assertNotIn(physical_key, submitted_payload)

    def test_wait_uses_receipt_oriented_public_completion(self) -> None:
        daemon = FakeDaemon()
        intent = make_intent()

        receipt = TurboBusClient(daemon=daemon).wait_transfer_receipt(
            intent.intent_id,
            timeout_seconds=1.5,
        )

        self.assertEqual(receipt.receipt_id, "receipt-wait")
        self.assertEqual(daemon.waited, [(intent.intent_id, 1.5)])

    def test_daemon_failure_is_raised_at_public_boundary(self) -> None:
        client = TurboBusClient(daemon=FailingDaemon())

        with self.assertRaisesRegex(RuntimeError, "daemon unavailable"):
            client.submit(make_intent())

    def test_mismatched_receipt_is_rejected(self) -> None:
        client = TurboBusClient(daemon=MismatchedReceiptDaemon())

        with self.assertRaisesRegex(ValueError, "intent_id"):
            client.submit(make_intent())

    def test_low_level_daemon_client_sends_intent_wire_request(self) -> None:
        daemon = RecordingDaemonClient()
        intent = make_intent()

        response = daemon.submit_transfer_intent(intent)

        self.assertTrue(response.ok)
        self.assertEqual(len(daemon.requests), 1)
        request = daemon.requests[0]
        self.assertEqual(request.request_type, RequestType.SUBMIT_TRANSFER_INTENT)
        self.assertEqual(request.session_id, intent.session_id)
        self.assertEqual(request.payload["intent"]["intent_id"], intent.intent_id)
        self.assertNotIn("mode", request.payload["intent"])

    def test_low_level_daemon_client_sends_receipt_wait_request(self) -> None:
        daemon = RecordingDaemonClient()

        response = daemon.wait_transfer_receipt("intent-1", timeout_seconds=2.0)

        self.assertTrue(response.ok)
        request = daemon.requests[0]
        self.assertEqual(request.request_type, RequestType.WAIT_TRANSFER_RECEIPT)
        self.assertEqual(request.payload["intent_id"], "intent-1")
        self.assertEqual(request.payload["timeout_seconds"], 2.0)

    def test_root_exports_daemon_first_public_contract(self) -> None:
        self.assertIs(turbobus.TurboBusClient, TurboBusClient)
        self.assertIs(turbobus.TransferIntent, TransferIntent)
        self.assertIs(turbobus.TransferReceipt, TransferReceipt)
        self.assertIn("TurboBusClient", turbobus.__all__)
        self.assertIn("TransferIntent", turbobus.__all__)
        self.assertIn("TransferReceipt", turbobus.__all__)
        self.assertNotIn("Runtime", turbobus.__all__)
        self.assertNotIn("plan_transfer", turbobus.__all__)

    def test_legacy_runtime_module_is_not_available_as_public_entrypoint(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("turbobus.runtime")


class FakeDaemon:
    def __init__(self) -> None:
        self.submitted_intents: list[TransferIntent] = []
        self.waited: list[tuple[str, float | None]] = []

    def submit_transfer_intent(self, intent: TransferIntent) -> DaemonResponse:
        self.submitted_intents.append(intent)
        return DaemonResponse(
            ok=True,
            payload={"receipt": asdict(make_receipt(intent, receipt_id="receipt-submit"))},
        )

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> DaemonResponse:
        self.waited.append((str(intent_id), timeout_seconds))
        return DaemonResponse(
            ok=True,
            payload={
                "receipt": asdict(
                    make_receipt(make_intent(intent_id=intent_id), receipt_id="receipt-wait")
                )
            },
        )


class FailingDaemon(FakeDaemon):
    def submit_transfer_intent(self, intent: TransferIntent) -> DaemonResponse:
        return DaemonResponse(ok=False, error="daemon unavailable")


class MismatchedReceiptDaemon(FakeDaemon):
    def submit_transfer_intent(self, intent: TransferIntent) -> DaemonResponse:
        receipt = make_receipt(make_intent(intent_id="other-intent"))
        return DaemonResponse(ok=True, payload={"receipt": asdict(receipt)})


class RecordingDaemonClient(TurboBusDaemonClient):
    def __init__(self) -> None:
        self.requests: list[DaemonRequest] = []

    def send(self, request: DaemonRequest) -> DaemonResponse:
        self.requests.append(request)
        intent_payload = request.payload.get("intent", {})
        intent_id = str(intent_payload.get("intent_id", "intent-1"))
        return DaemonResponse(
            ok=True,
            payload={"receipt": asdict(make_receipt(make_intent(intent_id=intent_id)))},
        )


def make_intent(
    *,
    intent_id: str = "intent-1",
    policy_hints: dict[str, object] | None = None,
) -> TransferIntent:
    return TransferIntent(
        intent_id=intent_id,
        job_id="job-1",
        session_id="session-1",
        source_buffer_id="cpu-buffer",
        destination_buffer_id="gpu-buffer",
        direction="h2d",
        total_bytes=4096,
        ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 4096},),
        workload_kind=WorkloadKind.KV_CACHE,
        policy_hints={} if policy_hints is None else policy_hints,
    )


def make_receipt(
    intent: TransferIntent,
    *,
    receipt_id: str = "receipt-1",
) -> TransferReceipt:
    return TransferReceipt(
        receipt_id=receipt_id,
        ticket_id="ticket-1",
        intent_id=intent.intent_id,
        decision_id="decision-1",
        topology_snapshot_id="topology-1",
        job_id=intent.job_id,
        session_id=intent.session_id,
        state=TransferStatusState.COMPLETE,
        bytes_total=intent.total_bytes,
        bytes_completed=intent.total_bytes,
        path_stats=({"kind": "daemon_decision", "bytes": intent.total_bytes},),
    )


if __name__ == "__main__":
    unittest.main()
