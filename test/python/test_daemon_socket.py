from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest

from turbobus.daemon import TurboBusDaemonClient
from turbobus.daemon.protocol import WorkerTransferAuthorizationRequest
from turbobus.daemon.server import TurboBusDaemon
from turbobus.daemon.topology import (
    DaemonResourceInventory,
    GpuInventoryRecord,
    StaticTopologyProvider,
)
from turbobus.transfer import TransferRequest


def send_request(path: str, request: dict) -> dict:
    return send_raw_request(path, (json.dumps(request) + "\n").encode("utf-8"))


def send_raw_request(path: str, request: bytes) -> dict:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(path)
        client.sendall(request)
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
    def test_invalid_socket_request_returns_error_and_keeps_daemon_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(relay_gpus=[1])
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

            invalid = send_raw_request(socket_path, b"{not-json\n")
            self.assertFalse(invalid["ok"])
            self.assertIn("invalid request", invalid["error"])

            profile = send_request(socket_path, {"request_type": "PROFILE"})
            self.assertTrue(profile["ok"])

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

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_get_and_put_profile_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(relay_gpus=[1])
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
            missing = client.get_profile(target_gpu=0, relay_gpus=[1])
            self.assertTrue(missing.ok)
            self.assertIsNone(missing.payload["profile"])

            stored = client.put_profile(
                target_gpu=0,
                relay_gpus=[1],
                profile={
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
                },
                profile_bytes=4096,
            )
            self.assertTrue(stored.ok)

            loaded = client.get_profile(target_gpu=0, relay_gpus=[1])
            self.assertTrue(loaded.ok)
            self.assertEqual(loaded.payload["profile"]["profile_bytes"], 4096)

            invalidated = client.invalidate_profile(target_gpu=0, relay_gpus=[1])
            self.assertTrue(invalidated.ok)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_get_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(
                relay_gpus=[1],
                topology_provider=StaticTopologyProvider(
                    DaemonResourceInventory(
                        gpus=(
                            GpuInventoryRecord(
                                device_id=1,
                                backend="cuda",
                                vendor="nvidia",
                                role="relay",
                            ),
                        ),
                        source="test",
                        discovered_at=1.0,
                    )
                ),
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
            inventory = client.get_inventory()

            self.assertTrue(inventory.ok)
            self.assertEqual(inventory.payload["inventory"]["source"], "test")
            self.assertEqual(
                inventory.payload["inventory"]["gpus"][0]["device_id"],
                1,
            )

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_plan_transfer_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(
                relay_gpus=[1],
                max_sessions_per_relay=1,
                max_inflight_chunks_per_relay=8,
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
                max_inflight_chunks=8,
            )
            self.assertTrue(registered.ok)
            session_id = registered.payload["session"]["session_id"]
            job = client.register_job(job_id="job-1", session_id=session_id)
            self.assertTrue(job.ok)
            cpu_buffer = client.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            )
            gpu_buffer = client.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
            )
            self.assertTrue(cpu_buffer.ok)
            self.assertTrue(gpu_buffer.ok)
            stored = client.put_profile(
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
            self.assertTrue(stored.ok)

            planned = client.plan_transfer_request(
                session_id=session_id,
                request=TransferRequest(
                    total_bytes=64,
                    chunk_bytes=16,
                    mode="pool",
                    direction="h2d",
                    job_id="job-1",
                    metadata={"buffer_ids": ["cpu-buffer", "gpu-buffer"]},
                ),
            )

            self.assertTrue(planned.ok)
            self.assertEqual(planned.payload["stats"]["resolved_mode"], "pool")
            transfer_id = planned.payload["transfer_id"]
            reservation_id = planned.payload["reservations"][0]["reservation_id"]
            lease_token = planned.payload["lease_tokens"][0]

            submitted = client.transfer_status(transfer_id)
            self.assertTrue(submitted.ok)
            self.assertEqual(submitted.payload["status"]["state"], "submitted")

            validated = client.validate_lease(
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                relay_gpu=1,
                job_id="job-1",
                buffer_ids=["cpu-buffer", "gpu-buffer"],
            )
            self.assertTrue(validated.ok)
            authorized = client.authorize_worker_transfer(
                WorkerTransferAuthorizationRequest(
                    transfer_id=transfer_id,
                    lease_id=lease_token["lease_id"],
                    token=lease_token["token"],
                    session_id=session_id,
                    job_id="job-1",
                    src_buffer_id="cpu-buffer",
                    dst_buffer_id="gpu-buffer",
                    direction="h2d",
                    relay_gpu=1,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                )
            )
            self.assertTrue(authorized.ok)
            self.assertEqual(
                authorized.payload["authorization"]["src_buffer"]["buffer_id"],
                "cpu-buffer",
            )

            released = client.release_transfer(reservation_id)
            self.assertTrue(released.ok)

            invalidated = client.validate_lease(
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                relay_gpu=1,
            )
            self.assertFalse(invalidated.ok)

            completed = client.transfer_status(transfer_id)
            self.assertTrue(completed.ok)
            self.assertEqual(completed.payload["status"]["state"], "complete")
            self.assertEqual(completed.payload["status"]["bytes_completed"], 64)


if __name__ == "__main__":
    unittest.main()
