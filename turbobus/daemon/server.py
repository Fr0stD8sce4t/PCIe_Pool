from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from dataclasses import asdict
from typing import Iterable

from .protocol import (
    DaemonRequest,
    DaemonResponse,
    RelayQuota,
    RequestType,
    Session,
    TransferReservation,
)


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
        self._reservations: dict[str, TransferReservation] = {}
        self._profile_cache: dict[str, dict] = {}
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
            for reservation_id, reservation in list(self._reservations.items()):
                if reservation.session_id == session_id:
                    self._release_reservation_locked(reservation_id)
            for gpu in session.relay_gpus:
                quota = self._relay_quotas.get(gpu)
                if quota is not None:
                    quota.sessions.discard(session_id)
            session.active = False
            return DaemonResponse(ok=True, payload={"session_id": session_id})

    def reserve_transfer(
        self,
        session_id: str,
        relay_gpu: int,
        chunks: int,
        bytes_: int = 0,
        direction: str = "unknown",
    ) -> DaemonResponse:
        chunks = int(chunks)
        relay_gpu = int(relay_gpu)
        if chunks <= 0:
            return DaemonResponse(ok=False, error="chunks must be positive")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                return DaemonResponse(ok=False, error="unknown session")
            if relay_gpu not in session.relay_gpus:
                return DaemonResponse(ok=False, error="relay GPU is not assigned to this session")
            if chunks > session.max_inflight_chunks:
                return DaemonResponse(ok=False, error="reservation exceeds session chunk limit")
            if session.active_chunks + chunks > session.max_inflight_chunks:
                return DaemonResponse(ok=False, error="session chunk quota is unavailable")
            quota = self._relay_quotas.get(relay_gpu)
            if quota is None or not quota.can_reserve(chunks):
                return DaemonResponse(ok=False, error="relay chunk quota is unavailable")

            reservation = TransferReservation(
                reservation_id=str(uuid.uuid4()),
                session_id=session_id,
                relay_gpu=relay_gpu,
                chunks=chunks,
                bytes=int(bytes_),
                direction=str(direction),
            )
            self._reservations[reservation.reservation_id] = reservation
            session.active_chunks += chunks
            quota.active_chunks += chunks
            return DaemonResponse(ok=True, payload={"reservation": asdict(reservation)})

    def release_transfer(self, reservation_id: str) -> DaemonResponse:
        with self._lock:
            reservation = self._release_reservation_locked(reservation_id)
            if reservation is None:
                return DaemonResponse(ok=False, error="unknown reservation")
            return DaemonResponse(ok=True, payload={"reservation_id": reservation_id})

    def get_profile(self, target_gpu: int, relay_gpus: Iterable[int]) -> DaemonResponse:
        key = self._profile_key(target_gpu, relay_gpus)
        with self._lock:
            entry = self._profile_cache.get(key)
            return DaemonResponse(ok=True, payload={"profile": dict(entry) if entry else None})

    def put_profile(
        self,
        target_gpu: int,
        relay_gpus: Iterable[int],
        profile: dict,
        profile_bytes: int = 0,
        updated_at: float | None = None,
    ) -> DaemonResponse:
        target = int(target_gpu)
        relays = [int(gpu) for gpu in relay_gpus]
        normalized = self._normalize_profile(profile, target)
        entry = {
            "target_gpu": target,
            "relay_gpus": relays,
            "profile_bytes": int(profile_bytes),
            "updated_at": float(time.time() if updated_at is None else updated_at),
            "profile": normalized,
        }
        key = self._profile_key(target, relays)
        with self._lock:
            self._profile_cache[key] = entry
        return DaemonResponse(ok=True, payload={"profile": dict(entry)})

    def _release_reservation_locked(self, reservation_id: str) -> TransferReservation | None:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return None
        session = self._sessions.get(reservation.session_id)
        if session is not None:
            session.active_chunks = max(0, session.active_chunks - reservation.chunks)
        quota = self._relay_quotas.get(reservation.relay_gpu)
        if quota is not None:
            quota.active_chunks = max(0, quota.active_chunks - reservation.chunks)
        return reservation

    def describe(self) -> DaemonResponse:
        with self._lock:
            return DaemonResponse(
                ok=True,
                payload={
                    "sessions": {key: asdict(value) for key, value in self._sessions.items()},
                    "reservations": {
                        key: asdict(value) for key, value in self._reservations.items()
                    },
                    "relay_quotas": {
                        key: {
                            "relay_gpu": quota.relay_gpu,
                            "max_sessions": quota.max_sessions,
                            "max_inflight_chunks": quota.max_inflight_chunks,
                            "active_chunks": quota.active_chunks,
                            "sessions": sorted(quota.sessions),
                        }
                        for key, quota in self._relay_quotas.items()
                    },
                    "profile_cache": {
                        key: dict(value) for key, value in self._profile_cache.items()
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
        if request.request_type == RequestType.RESERVE_TRANSFER:
            if request.session_id is None:
                return DaemonResponse(ok=False, error="session_id is required")
            payload = request.payload
            return self.reserve_transfer(
                session_id=request.session_id,
                relay_gpu=int(payload["relay_gpu"]),
                chunks=int(payload.get("chunks", 1)),
                bytes_=int(payload.get("bytes", 0)),
                direction=str(payload.get("direction", "unknown")),
            )
        if request.request_type == RequestType.RELEASE_TRANSFER:
            payload = request.payload
            reservation_id = payload.get("reservation_id")
            if reservation_id is None:
                return DaemonResponse(ok=False, error="reservation_id is required")
            return self.release_transfer(str(reservation_id))
        if request.request_type == RequestType.GET_PROFILE:
            payload = request.payload
            return self.get_profile(
                target_gpu=int(payload["target_gpu"]),
                relay_gpus=payload.get("relay_gpus", []),
            )
        if request.request_type == RequestType.PUT_PROFILE:
            payload = request.payload
            try:
                return self.put_profile(
                    target_gpu=int(payload["target_gpu"]),
                    relay_gpus=payload.get("relay_gpus", []),
                    profile=payload.get("profile", {}),
                    profile_bytes=int(payload.get("profile_bytes", 0)),
                    updated_at=payload.get("updated_at"),
                )
            except Exception as exc:
                return DaemonResponse(ok=False, error=str(exc))
        if request.request_type == RequestType.PROFILE:
            return self.describe()
        return DaemonResponse(ok=False, error=f"unsupported request: {request.request_type}")

    @staticmethod
    def _profile_key(target_gpu: int, relay_gpus: Iterable[int]) -> str:
        relays = ",".join(str(gpu) for gpu in sorted(int(gpu) for gpu in relay_gpus))
        return f"target={int(target_gpu)};relays={relays}"

    @staticmethod
    def _normalize_profile(profile: dict, target_gpu: int) -> dict:
        if not isinstance(profile, dict):
            raise ValueError("profile must be a dict")
        direct_h2d = float(profile.get("direct_h2d_bw_gbps", 0.0) or 0.0)
        direct_d2h = float(profile.get("direct_d2h_bw_gbps", 0.0) or 0.0)
        if direct_h2d <= 0.0:
            raise ValueError("profile direct_h2d_bw_gbps must be positive")
        relays = []
        for relay in profile.get("relays", []) or []:
            if not isinstance(relay, dict):
                raise ValueError("profile relays must be dicts")
            relays.append(
                {
                    "relay_device": int(relay["relay_device"]),
                    "target_device": int(relay.get("target_device", target_gpu)),
                    "h2d_bw_gbps": float(relay.get("h2d_bw_gbps", 0.0) or 0.0),
                    "d2h_bw_gbps": float(relay.get("d2h_bw_gbps", 0.0) or 0.0),
                    "p2p_bw_gbps": float(relay.get("p2p_bw_gbps", 0.0) or 0.0),
                    "effective_bw_gbps": float(relay.get("effective_bw_gbps", 0.0) or 0.0),
                    "effective_d2h_bw_gbps": float(
                        relay.get("effective_d2h_bw_gbps", 0.0) or 0.0
                    ),
                    "p2p_enabled": bool(relay.get("p2p_enabled", False)),
                }
            )
        return {
            "target_device": int(profile.get("target_device", target_gpu)),
            "direct_h2d_bw_gbps": direct_h2d,
            "direct_d2h_bw_gbps": direct_d2h,
            "relays": relays,
        }

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
