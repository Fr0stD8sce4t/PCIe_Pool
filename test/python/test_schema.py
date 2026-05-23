from __future__ import annotations

from dataclasses import asdict
import json
import unittest

from turbobus.schema import (
    AutoTransferDecision,
    DaemonRequest,
    DaemonResponse,
    RelayQuota,
    RequestType,
    Session,
    TransferMode,
    TransferReservation,
)


class SchemaTest(unittest.TestCase):
    def test_auto_transfer_decision_is_json_serializable(self) -> None:
        decision = AutoTransferDecision(
            requested_mode=TransferMode.AUTO,
            resolved_mode=TransferMode.POOL,
            request_bytes=1024,
            request_chunks=2,
            direct_h2d_bw_gbps=7.5,
            relay_effective_bw_gbps=8.0,
            eligible_relay_devices=(1, 2),
            reason="pool speedup",
        )

        payload = json.loads(json.dumps(asdict(decision)))

        self.assertEqual(payload["resolved_mode"], "pool")
        self.assertEqual(payload["eligible_relay_devices"], [1, 2])

    def test_daemon_protocol_round_trip(self) -> None:
        request = DaemonRequest(
            request_type=RequestType.RESERVE_TRANSFER,
            session_id="session-1",
            payload={"relay_gpu": 1, "chunks": 2},
        )
        session = Session(
            session_id="session-1",
            target_gpu=0,
            relay_gpus=[1],
            max_inflight_chunks=4,
        )
        reservation = TransferReservation(
            reservation_id="reservation-1",
            session_id="session-1",
            relay_gpu=1,
            chunks=2,
            bytes=4096,
            direction="h2d",
        )
        response = DaemonResponse(
            ok=True,
            payload={
                "session": asdict(session),
                "reservation": asdict(reservation),
            },
        )

        request_payload = json.loads(json.dumps(asdict(request)))
        response_payload = json.loads(json.dumps(asdict(response)))

        self.assertEqual(request_payload["request_type"], "RESERVE_TRANSFER")
        self.assertEqual(response_payload["payload"]["session"]["session_id"], "session-1")
        self.assertEqual(
            response_payload["payload"]["reservation"]["reservation_id"],
            "reservation-1",
        )

    def test_plan_transfer_request_is_serializable(self) -> None:
        request = DaemonRequest(
            request_type=RequestType.PLAN_TRANSFER,
            session_id="session-1",
            payload={
                "total_bytes": 64,
                "chunk_bytes": 16,
                "mode": "pool",
                "direction": "h2d",
            },
        )

        payload = json.loads(json.dumps(asdict(request)))

        self.assertEqual(payload["request_type"], "PLAN_TRANSFER")
        self.assertEqual(payload["payload"]["mode"], "pool")

    def test_relay_quota_limits(self) -> None:
        quota = RelayQuota(relay_gpu=1, max_sessions=1, max_inflight_chunks=4)

        self.assertTrue(quota.can_attach())
        self.assertTrue(quota.can_reserve(4))

        quota.sessions.add("session-1")
        quota.active_chunks = 2

        self.assertFalse(quota.can_attach())
        self.assertTrue(quota.can_reserve(2))
        self.assertFalse(quota.can_reserve(3))


if __name__ == "__main__":
    unittest.main()
