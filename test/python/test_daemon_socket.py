from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest

from turbobus.daemon import TurboBusDaemonClient
from turbobus.daemon.server import TurboBusDaemon


def send_request(path: str, request: dict) -> dict:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(path)
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        data = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        return json.loads(data.decode("utf-8"))
    finally:
        client.close()


class DaemonSocketTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_socket_session_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(relay_gpus=[1], max_sessions_per_relay=1)
            thread = threading.Thread(
                target=daemon.serve_forever,
                args=(socket_path,),
                daemon=True,
            )
            thread.start()

            for _ in range(100):
                if os.path.exists(socket_path):
                    break
                time.sleep(0.01)
            self.assertTrue(os.path.exists(socket_path))

            register = send_request(
                socket_path,
                {
                    "request_type": "REGISTER_SESSION",
                    "payload": {"target_gpu": 0, "relay_gpus": [1]},
                },
            )
            self.assertTrue(register["ok"])
            session_id = register["payload"]["session"]["session_id"]

            profile = send_request(socket_path, {"request_type": "PROFILE"})
            self.assertTrue(profile["ok"])
            self.assertIn(session_id, profile["payload"]["sessions"])

            closed = send_request(
                socket_path,
                {
                    "request_type": "CLOSE_SESSION",
                    "session_id": session_id,
                },
            )
            self.assertTrue(closed["ok"])

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_reserve_and_release_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(
                relay_gpus=[1],
                max_sessions_per_relay=2,
                max_inflight_chunks_per_relay=4,
            )
            thread = threading.Thread(
                target=daemon.serve_forever,
                args=(socket_path,),
                daemon=True,
            )
            thread.start()

            for _ in range(100):
                if os.path.exists(socket_path):
                    break
                time.sleep(0.01)
            self.assertTrue(os.path.exists(socket_path))

            client = TurboBusDaemonClient(socket_path)
            registered = client.register_session(
                target_gpu=0,
                relay_gpus=[1],
                max_inflight_chunks=4,
            )
            self.assertTrue(registered.ok)
            session_id = registered.payload["session"]["session_id"]
            other_registered = client.register_session(
                target_gpu=2,
                relay_gpus=[1],
                max_inflight_chunks=4,
            )
            self.assertTrue(other_registered.ok)
            other_session_id = other_registered.payload["session"]["session_id"]

            reserved = client.reserve_transfer(
                session_id,
                relay_gpu=1,
                chunks=4,
                bytes_=1024,
                direction="h2d",
            )
            self.assertTrue(reserved.ok)
            reservation_id = reserved.payload["reservation"]["reservation_id"]

            blocked = client.reserve_transfer(session_id, relay_gpu=1, chunks=1)
            self.assertFalse(blocked.ok)

            other_blocked = client.reserve_transfer(other_session_id, relay_gpu=1, chunks=1)
            self.assertFalse(other_blocked.ok)

            released = client.release_transfer(reservation_id)
            self.assertTrue(released.ok)

            second = client.reserve_transfer(other_session_id, relay_gpu=1, chunks=1)
            self.assertTrue(second.ok)

            closed = client.close_session(session_id)
            self.assertTrue(closed.ok)
            other_closed = client.close_session(other_session_id)
            self.assertTrue(other_closed.ok)


if __name__ == "__main__":
    unittest.main()
