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

    def test_register_session_normalizes_duplicate_relays(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1], max_sessions_per_relay=1)

        registered = daemon.register_session(target_gpu=0, requested_relays=[1, 1])

        self.assertTrue(registered.ok)
        session_id = registered.payload["session"]["session_id"]
        self.assertEqual(registered.payload["session"]["relay_gpus"], [1])
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["sessions"], [session_id])

    def test_register_session_rejects_invalid_session_chunk_limit(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        response = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=0,
        )

        self.assertFalse(response.ok)
        self.assertIn("max_inflight_chunks", response.error)

    def test_job_and_buffer_registration_are_tracked_and_cleaned_up(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        job = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_JOB,
                payload={
                    "job_id": "job-1",
                    "user_id": "user-1",
                    "session_id": "session-1",
                },
            )
        )
        self.assertTrue(job.ok)
        self.assertEqual(job.payload["job"]["job_id"], "job-1")

        buffer_registration = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_BUFFER,
                payload={
                    "buffer_id": "buffer-1",
                    "job_id": "job-1",
                    "kind": "cpu_pinned",
                    "size_bytes": 4096,
                    "device_index": 0,
                    "pinned": True,
                },
            )
        )
        self.assertTrue(buffer_registration.ok)
        self.assertEqual(buffer_registration.payload["buffer"]["buffer_id"], "buffer-1")

        snapshot = daemon.describe().payload
        self.assertEqual(snapshot["jobs"]["job-1"]["user_id"], "user-1")
        self.assertEqual(snapshot["buffers"]["buffer-1"]["kind"], "cpu_pinned")

        cleanup = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.CLEANUP,
                payload={
                    "target_kind": "job",
                    "target_id": "job-1",
                    "reason": "manual",
                },
            )
        )
        self.assertTrue(cleanup.ok)
        self.assertEqual(cleanup.payload["removed"]["jobs"], 1)
        self.assertEqual(cleanup.payload["removed"]["buffers"], 1)
        self.assertNotIn("job-1", daemon.describe().payload["jobs"])
        self.assertNotIn("buffer-1", daemon.describe().payload["buffers"])

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

    def test_profile_cache_can_be_invalidated_explicitly(self) -> None:
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

        daemon.put_profile(target_gpu=0, relay_gpus=[1], profile=profile, profile_bytes=1234)

        invalidated = daemon.invalidate_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(invalidated.ok)
        self.assertTrue(invalidated.payload["removed"])
        self.assertIsNone(daemon.get_profile(target_gpu=0, relay_gpus=[1]).payload["profile"])

    def test_profile_cache_purges_stale_entries_on_access(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1], profile_max_age_seconds=1.0)
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

        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile=profile,
            profile_bytes=1234,
            updated_at=time.time() - 10.0,
        )

        loaded = daemon.get_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(loaded.ok)
        self.assertIsNone(loaded.payload["profile"])
        self.assertEqual(daemon.describe().payload["profile_cache"], {})

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

    def test_handle_request_rejects_missing_required_fields(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        response = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_SESSION,
                payload={"relay_gpus": [1]},
            )
        )

        self.assertFalse(response.ok)
        self.assertIn("invalid request", response.error)

    def test_wire_message_errors_do_not_mutate_state(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        malformed = daemon.handle_wire_message("{not-json")
        missing_type = daemon.handle_wire_message("{}")
        good = daemon.handle_wire_message(
            '{"request_type":"REGISTER_SESSION","payload":{"target_gpu":0,"relay_gpus":[1]}}'
        )

        self.assertFalse(malformed.ok)
        self.assertFalse(missing_type.ok)
        self.assertTrue(good.ok)
        self.assertEqual(len(daemon.describe().payload["sessions"]), 1)

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

    def test_transfer_reservation_rejects_invalid_payload_values(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]

        negative_bytes = daemon.reserve_transfer(
            session_id,
            relay_gpu=1,
            chunks=1,
            bytes_=-1,
        )
        invalid_direction = daemon.reserve_transfer(
            session_id,
            relay_gpu=1,
            chunks=1,
            direction="sideways",
        )

        self.assertFalse(negative_bytes.ok)
        self.assertIn("bytes", negative_bytes.error)
        self.assertFalse(invalid_direction.ok)
        self.assertIn("direction", invalid_direction.error)
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_stale_session_reap_releases_reservations_and_quota(self) -> None:
        daemon = TurboBusDaemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=4,
            session_timeout_seconds=1.0,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=4,
        )
        session_id = register.payload["session"]["session_id"]
        reserved = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=4)
        self.assertTrue(reserved.ok)

        daemon._sessions[session_id].last_seen = time.time() - 10.0
        expired = daemon.reap_stale_sessions(now=time.time())

        self.assertEqual(expired, [session_id])
        profile = daemon.describe().payload
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        self.assertEqual(profile["relay_quotas"][1]["sessions"], [])
        self.assertEqual(profile["sessions"], {})

        reopened = daemon.register_session(target_gpu=0, requested_relays=[1], max_inflight_chunks=4)
        self.assertTrue(reopened.ok)

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

    def test_plan_transfer_uses_cached_profile_and_reserves_leases(self) -> None:
        daemon = TurboBusDaemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
                "target_device": 0,
                "direct_h2d_bw_gbps": 7.5,
                "direct_d2h_bw_gbps": 6.5,
                "relays": [
                    {
                        "relay_device": 1,
                        "target_device": 0,
                        "h2d_bw_gbps": 7.5,
                        "d2h_bw_gbps": 6.5,
                        "p2p_bw_gbps": 40.0,
                        "effective_bw_gbps": 7.5,
                        "effective_d2h_bw_gbps": 6.5,
                        "p2p_enabled": True,
                    }
                ],
            },
        )

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
        )

        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "pool")
        self.assertEqual(len(planned.payload["leases"]), 1)
        self.assertEqual(len(planned.payload["reservations"]), 1)
        reservation = planned.payload["reservations"][0]
        self.assertEqual(
            reservation["reservation_id"],
            planned.payload["leases"][0]["lease_id"],
        )
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 2)

        released = daemon.release_transfer(reservation["reservation_id"])
        self.assertTrue(released.ok)
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_plan_transfer_falls_back_direct_when_relay_quota_is_unavailable(self) -> None:
        daemon = TurboBusDaemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=1,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
                "target_device": 0,
                "direct_h2d_bw_gbps": 7.5,
                "direct_d2h_bw_gbps": 6.5,
                "relays": [
                    {
                        "relay_device": 1,
                        "target_device": 0,
                        "h2d_bw_gbps": 7.5,
                        "d2h_bw_gbps": 6.5,
                        "p2p_bw_gbps": 40.0,
                        "effective_bw_gbps": 7.5,
                        "effective_d2h_bw_gbps": 6.5,
                        "p2p_enabled": True,
                    }
                ],
            },
        )

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                },
            )
        )

        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "direct")
        self.assertIn("quota", planned.payload["stats"]["fallback_reason"])
        self.assertEqual(planned.payload["leases"], [])
        self.assertEqual(planned.payload["reservations"], [])
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_plan_transfer_rejects_unknown_session(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id="missing",
                payload={"total_bytes": 64, "chunk_bytes": 16},
            )
        )

        self.assertFalse(planned.ok)
        self.assertIn("unknown session", planned.error)


if __name__ == "__main__":
    unittest.main()
