from __future__ import annotations

import json
import os
import socket
import threading
import uuid
from dataclasses import asdict
from typing import Iterable

from .protocol import DaemonRequest, DaemonResponse, RelayQuota, RequestType, Session


class TurboBusDaemon:
    """Minimal resource-control daemon.

    The first version deliberately does not move GPU pointers across processes.
    It owns session and relay quota state; client processes still execute CUDA
    transfers locally after obtaining a session.
    """

    def __init__(
        self,
        relay_gpus: Iterable[int],
        max_sessions_per_relay: int = 1,
        max_inflight_chunks_per_relay: int = 8,
    ) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._relay_quotas = {
            int(gpu): RelayQuota(
                relay_gpu=int(gpu),
                max_sessions=max_sessions_per_relay,
                max_inflight_chunks=max_inflight_chunks_per_relay,
            )
            for gpu in relay_gpus
        }

    def register_session(
        self,
        target_gpu: int,
        requested_relays: Iterable[int],
        max_inflight_chunks: int = 8,
    ) -> DaemonResponse:
        relays = [int(gpu) for gpu in requested_relays]
        with self._lock:
            unavailable = [
                gpu
                for gpu in relays
                if gpu not in self._relay_quotas or not self._relay_quotas[gpu].can_attach()
            ]
            if unavailable:
                return DaemonResponse(
                    ok=False,
                    error=f"relay GPUs are unavailable: {unavailable}",
                )

            session_id = str(uuid.uuid4())
            session = Session(
                session_id=session_id,
                target_gpu=int(target_gpu),
                relay_gpus=relays,
                max_inflight_chunks=int(max_inflight_chunks),
            )
            self._sessions[session_id] = session
            for gpu in relays:
                self._relay_quotas[gpu].sessions.add(session_id)
            return DaemonResponse(ok=True, payload={"session": asdict(session)})

    def close_session(self, session_id: str) -> DaemonResponse:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return DaemonResponse(ok=False, error="unknown session")
            for gpu in session.relay_gpus:
                quota = self._relay_quotas.get(gpu)
                if quota is not None:
                    quota.sessions.discard(session_id)
            session.active = False
            return DaemonResponse(ok=True, payload={"session_id": session_id})

    def describe(self) -> DaemonResponse:
        with self._lock:
            return DaemonResponse(
                ok=True,
                payload={
                    "sessions": {key: asdict(value) for key, value in self._sessions.items()},
                    "relay_quotas": {
                        key: {
                            "relay_gpu": quota.relay_gpu,
                            "max_sessions": quota.max_sessions,
                            "max_inflight_chunks": quota.max_inflight_chunks,
                            "sessions": sorted(quota.sessions),
                        }
                        for key, quota in self._relay_quotas.items()
                    },
                },
            )

    def handle_request(self, request: DaemonRequest) -> DaemonResponse:
        if request.request_type == RequestType.REGISTER_SESSION:
            payload = request.payload
            return self.register_session(
                target_gpu=int(payload["target_gpu"]),
                requested_relays=payload.get("relay_gpus", []),
                max_inflight_chunks=int(payload.get("max_inflight_chunks", 8)),
            )
        if request.request_type == RequestType.CLOSE_SESSION:
            if request.session_id is None:
                return DaemonResponse(ok=False, error="session_id is required")
            return self.close_session(request.session_id)
        if request.request_type == RequestType.PROFILE:
            return self.describe()
        return DaemonResponse(ok=False, error=f"unsupported request: {request.request_type}")

    def serve_forever(self, socket_path: str) -> None:
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen()

        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    data = b""
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                        if b"\n" in data:
                            break

                    if not data:
                        continue

                    line, _, _ = data.partition(b"\n")
                    request_data = json.loads(line.decode("utf-8"))
                    request = DaemonRequest(
                        request_type=RequestType(request_data["request_type"]),
                        session_id=request_data.get("session_id"),
                        payload=request_data.get("payload", {}),
                    )
                    response = self.handle_request(request)
                    conn.sendall((json.dumps(asdict(response)) + "\n").encode("utf-8"))
        finally:
            server.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)


def socket_path_for_user(base_dir: str = "/tmp") -> str:
    return f"{base_dir.rstrip('/')}/turbobusd.sock"


def reserve_socket(path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen()
    return sock
