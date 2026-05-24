from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest

from turbobus.daemon import TurboBusDaemonClient
from turbobus.daemon.protocol import (
    DaemonResponse,
    RequestType,
    WorkerTransferAuthorizationRequest,
)
from turbobus.daemon.server import TurboBusDaemon
from turbobus.topology import (
    DaemonResourceInventory,
    GpuInventoryRecord,
)
from test.python.fixtures.topology import (
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


class RecordingDaemonClient(TurboBusDaemonClient):
    def __init__(self) -> None:
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return DaemonResponse(ok=True, payload={"sessions": {}})


class DaemonSocketTest(unittest.TestCase):
    def test_client_describe_uses_profile_request(self) -> None:
        client = RecordingDaemonClient()

        response = client.describe()

        self.assertTrue(response.ok)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].request_type, RequestType.PROFILE)

    def test_client_cleanup_uses_cleanup_request(self) -> None:
        client = RecordingDaemonClient()

        response = client.cleanup(
            target_kind="session",
            target_id="session-1",
            reason="test",
            force=True,
        )

        self.assertTrue(response.ok)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].request_type, RequestType.CLEANUP)
        self.assertEqual(
            client.requests[0].payload,
            {
                "target_kind": "session",
                "target_id": "session-1",
                "reason": "test",
                "force": True,
            },
        )

    def test_client_discover_relays_uses_discover_request(self) -> None:
        client = RecordingDaemonClient()

        response = client.discover_relays(target_gpu=0, relay_gpus=[1, 2])

        self.assertTrue(response.ok)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].request_type, RequestType.DISCOVER_RELAYS)
        self.assertEqual(
            client.requests[0].payload,
            {
                "target_gpu": 0,
                "relay_gpus": [1, 2],
            },
        )

    def test_client_reap_expired_leases_uses_reap_request(self) -> None:
        client = RecordingDaemonClient()

        response = client.reap_expired_leases(now=12.5)

        self.assertTrue(response.ok)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].request_type, RequestType.REAP_EXPIRED_LEASES)
        self.assertEqual(client.requests[0].payload, {"now": 12.5})

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_socket_register_job_rejects_unknown_session(self) -> None:
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

            client = TurboBusDaemonClient(socket_path)
            registered = client.register_job(
                job_id="job-1",
                session_id="missing-session",
            )

            self.assertFalse(registered.ok)
            self.assertIn("unknown session", registered.error)
            self.assertEqual(daemon.describe().payload["jobs"], {})

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
    def test_client_reap_expired_leases_round_trip_clears_relay_discovery_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(
                relay_gpus=[1],
                max_sessions_per_relay=1,
                max_inflight_chunks_per_relay=2,
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
            client.register_job(job_id="job-1", session_id=session_id)
            client.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            )
            client.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
            )
            client.put_profile(
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
            lease_token = planned.payload["lease_tokens"][0]

            reaped = client.reap_expired_leases(now=lease_token["expires_at"] + 1.0)
            self.assertTrue(reaped.ok)
            self.assertEqual(reaped.payload["expired_lease_ids"], [lease_token["lease_id"]])
            self.assertEqual(reaped.payload["expired_count"], 1)

            discovered = client.discover_relays(target_gpu=0, relay_gpus=[1])
            self.assertTrue(discovered.ok)
            relay = discovered.payload["relay_discovery"]
            self.assertEqual(relay["summary"]["active_reservation_count"], 0)
            self.assertEqual(relay["summary"]["active_lease_count"], 0)
            self.assertEqual(relay["relays"][0]["reservations"], [])
            self.assertEqual(relay["relays"][0]["leases"], [])
            self.assertEqual(relay["relays"][0]["quota"]["available_chunks"], 2)

            fallback_session = client.register_session(
                target_gpu=2,
                relay_gpus=[1],
                max_inflight_chunks=8,
            )
            self.assertTrue(fallback_session.ok)
            fallback_planned = client.plan_transfer_request(
                session_id=fallback_session.payload["session"]["session_id"],
                request=TransferRequest(
                    total_bytes=64,
                    chunk_bytes=16,
                    mode="pool",
                    direction="h2d",
                    job_id="job-2",
                ),
            )
            self.assertTrue(fallback_planned.ok)
            self.assertEqual(fallback_planned.payload["stats"]["resolved_mode"], "direct")
            self.assertEqual(fallback_planned.payload["reservations"], [])

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_cleanup_reports_control_plane_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(
                relay_gpus=[1],
                max_sessions_per_relay=1,
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
            reserved = client.reserve_transfer(
                session_id=session_id,
                relay_gpu=1,
                chunks=4,
            )
            self.assertTrue(reserved.ok)
            reservation_id = reserved.payload["reservation"]["reservation_id"]

            cleanup = client.cleanup(
                target_kind="session",
                target_id=session_id,
                reason="client_requested",
                force=True,
            )
            self.assertTrue(cleanup.ok)
            self.assertEqual(cleanup.payload["removed"]["sessions"], 1)
            profile = client.describe()

            self.assertIn(cleanup.payload["cleanup"], profile.payload["cleanup_events"])
            self.assertIn(
                {
                    "target_kind": "session",
                    "target_id": session_id,
                    "reason": "client_requested",
                    "force": True,
                },
                profile.payload["system_cleanup_events"],
            )
            self.assertIn(
                {
                    "target_kind": "reservation",
                    "target_id": reservation_id,
                    "reason": "client_requested",
                    "force": True,
                },
                profile.payload["system_cleanup_events"],
            )

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_describe_reports_cleanup_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "turbobusd.sock")
            daemon = TurboBusDaemon(
                relay_gpus=[1],
                max_sessions_per_relay=1,
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
            reserved = client.reserve_transfer(
                session_id=session_id,
                relay_gpu=1,
                chunks=4,
            )
            self.assertTrue(reserved.ok)
            reservation_id = reserved.payload["reservation"]["reservation_id"]

            closed = client.close_session(session_id)
            self.assertTrue(closed.ok)
            profile = client.describe()

            self.assertTrue(profile.ok)
            self.assertIn(
                {
                    "target_kind": "session",
                    "target_id": session_id,
                    "reason": "session_closed",
                    "force": True,
                },
                profile.payload["system_cleanup_events"],
            )
            self.assertIn(
                {
                    "target_kind": "reservation",
                    "target_id": reservation_id,
                    "reason": "session_closed",
                    "force": True,
                },
                profile.payload["system_cleanup_events"],
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
                )
            )
            self.assertTrue(authorized.ok)
            self.assertEqual(
                authorized.payload["authorization"]["src_buffer"]["buffer_id"],
                "cpu-buffer",
            )
            self.assertEqual(
                authorized.payload["authorization"]["plan"],
                planned.payload["plan"],
            )

            reported = client.transfer_status(
                transfer_id,
                state="complete",
                bytes_completed=64,
            )
            self.assertTrue(reported.ok)

            invalidated = client.validate_lease(
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                relay_gpu=1,
            )
            self.assertFalse(invalidated.ok)
            self.assertIn("transfer is terminal", invalidated.error)

            released = client.release_transfer(reservation_id)
            self.assertTrue(released.ok)

            completed = client.transfer_status(transfer_id)
            self.assertTrue(completed.ok)
            self.assertEqual(completed.payload["status"]["state"], "complete")
            self.assertEqual(completed.payload["status"]["bytes_completed"], 64)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_client_plan_transfer_round_trip_preserves_range_offsets(self) -> None:
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
            self.assertTrue(client.register_job(job_id="job-1", session_id=session_id).ok)
            self.assertTrue(
                client.register_buffer(
                    buffer_id="cpu-buffer",
                    job_id="job-1",
                    kind="cpu_pinned",
                    size_bytes=64,
                    pinned=True,
                ).ok
            )
            self.assertTrue(
                client.register_buffer(
                    buffer_id="gpu-buffer",
                    job_id="job-1",
                    kind="gpu",
                    size_bytes=64,
                    device_index=0,
                ).ok
            )
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
                request=TransferRequest.from_ranges(
                    [{"src_offset": 8, "dst_offset": 24, "bytes": 16}],
                    chunk_bytes=8,
                    mode="relay",
                    direction="h2d",
                    job_id="job-1",
                    metadata={"buffer_ids": ["cpu-buffer", "gpu-buffer"]},
                ),
            )

            expected_ranges = (
                {"src_offset": 8, "dst_offset": 24, "bytes": 8},
                {"src_offset": 16, "dst_offset": 32, "bytes": 8},
            )
            self.assertTrue(planned.ok)
            self.assertEqual(planned.payload["stats"]["resolved_mode"], "relay")
            self.assertEqual(
                tuple(
                    chunk
                    for assignment in planned.payload["plan"]["assignments"]
                    for chunk in assignment["chunks"]
                ),
                expected_ranges,
            )

            lease_token = planned.payload["lease_tokens"][0]
            authorized = client.authorize_worker_transfer(
                WorkerTransferAuthorizationRequest(
                    transfer_id=planned.payload["transfer_id"],
                    lease_id=lease_token["lease_id"],
                    token=lease_token["token"],
                    session_id=session_id,
                    job_id="job-1",
                    src_buffer_id="cpu-buffer",
                    dst_buffer_id="gpu-buffer",
                    direction="h2d",
                    relay_gpu=1,
                )
            )

            self.assertTrue(authorized.ok)
            self.assertEqual(
                tuple(authorized.payload["authorization"]["ranges"]),
                expected_ranges,
            )


if __name__ == "__main__":
    unittest.main()
