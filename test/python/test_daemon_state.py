from __future__ import annotations

import time
import unittest

from turbobus.daemon.protocol import DaemonRequest, RequestType
from turbobus.daemon.server import TurboBusDaemon


class DaemonStateTest(unittest.TestCase):
    def test_session_lifecycle_releases_quota(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1], max_sessions_per_relay=1)

        first = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(first.ok)
        session_id = first.payload["session"]["session_id"]

        second = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertFalse(second.ok)
        self.assertIn("unavailable", second.error)

        closed = daemon.close_session(session_id)
        self.assertTrue(closed.ok)

        third = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(third.ok)

    def test_handle_request_profile(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1, 2], max_sessions_per_relay=2)
        register = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_SESSION,
                payload={"target_gpu": 0, "relay_gpus": [1]},
            )
        )
        self.assertTrue(register.ok)

        profile = daemon.handle_request(DaemonRequest(request_type=RequestType.PROFILE))
        self.assertTrue(profile.ok)
        self.assertIn("sessions", profile.payload)
        self.assertEqual(len(profile.payload["sessions"]), 1)

        session_id = register.payload["session"]["session_id"]
        closed = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.CLOSE_SESSION,
                session_id=session_id,
            )
        )
        self.assertTrue(closed.ok)

    def test_profile_cache_get_put_round_trip(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])
        profile = {
            "target_device": 0,
            "direct_h2d_bw_gbps": 7.5,
            "direct_d2h_bw_gbps": 8.5,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 7.6,
                    "d2h_bw_gbps": 8.6,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 7.6,
                    "effective_d2h_bw_gbps": 8.6,
                    "p2p_enabled": True,
                }
            ],
        }

        missing = daemon.get_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(missing.ok)
        self.assertIsNone(missing.payload["profile"])

        stored = daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile=profile,
            profile_bytes=1234,
            updated_at=time.time(),
        )
        self.assertTrue(stored.ok)

        loaded = daemon.get_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(loaded.ok)
        self.assertEqual(loaded.payload["profile"]["profile_bytes"], 1234)
        self.assertEqual(
            loaded.payload["profile"]["profile"]["relays"][0]["relay_device"],
            1,
        )

    def test_handle_request_rejects_invalid_profile_cache_update(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        response = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PUT_PROFILE,
                payload={
                    "target_gpu": 0,
                    "relay_gpus": [1],
                    "profile": {"direct_h2d_bw_gbps": 0.0},
                },
            )
        )

        self.assertFalse(response.ok)
        self.assertIn("direct_h2d", response.error)

    def test_transfer_reservation_uses_relay_chunk_quota(self) -> None:
        daemon = TurboBusDaemon(
            relay_gpus=[1],
            max_sessions_per_relay=2,
            max_inflight_chunks_per_relay=4,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=4,
        )
        session_id = register.payload["session"]["session_id"]

        first = daemon.reserve_transfer(
            session_id,
            relay_gpu=1,
            chunks=3,
            bytes_=1024,
            direction="h2d",
        )
        self.assertTrue(first.ok)

        blocked = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=2)
        self.assertFalse(blocked.ok)
        self.assertIn("quota", blocked.error)

        reservation_id = first.payload["reservation"]["reservation_id"]
        released = daemon.release_transfer(reservation_id)
        self.assertTrue(released.ok)

        second = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=2)
        self.assertTrue(second.ok)

    def test_close_session_releases_transfer_reservations(self) -> None:
        daemon = TurboBusDaemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=4,
        )
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]
        reserved = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=4)
        self.assertTrue(reserved.ok)

        closed = daemon.close_session(session_id)
        self.assertTrue(closed.ok)

        profile = daemon.describe()
        self.assertEqual(profile.payload["relay_quotas"][1]["active_chunks"], 0)

    def test_transfer_reservation_uses_session_chunk_quota(self) -> None:
        daemon = TurboBusDaemon(
            relay_gpus=[1, 2],
            max_sessions_per_relay=2,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1, 2],
            max_inflight_chunks=4,
        )
        session_id = register.payload["session"]["session_id"]

        first = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=3)
        self.assertTrue(first.ok)

        blocked = daemon.reserve_transfer(session_id, relay_gpu=2, chunks=2)
        self.assertFalse(blocked.ok)
        self.assertIn("session chunk quota", blocked.error)

        reservation_id = first.payload["reservation"]["reservation_id"]
        released = daemon.release_transfer(reservation_id)
        self.assertTrue(released.ok)

        second = daemon.reserve_transfer(session_id, relay_gpu=2, chunks=2)
        self.assertTrue(second.ok)


if __name__ == "__main__":
    unittest.main()
