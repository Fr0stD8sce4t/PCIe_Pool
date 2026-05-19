from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

