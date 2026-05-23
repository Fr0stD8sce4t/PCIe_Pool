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
    BufferRegistration,
    CleanupRequest,
    DaemonRequest,
    DaemonResponse,
    JobIdentity,
    RelayQuota,
    RequestType,
    Session,
    TransferReservation,
    TransferStatus,
    TransferStatusState,
)
from .scheduler import DaemonScheduler, SchedulerDecision


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
        session_timeout_seconds: float = 0.0,
        profile_max_age_seconds: float = 0.0,
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobIdentity] = {}
        self._buffers: dict[str, BufferRegistration] = {}
        self._sessions: dict[str, Session] = {}
        self._reservations: dict[str, TransferReservation] = {}
        self._reservation_transfers: dict[str, str] = {}
        self._transfer_statuses: dict[str, TransferStatus] = {}
        self._cleanup_events: list[CleanupRequest] = []
        self._profile_cache: dict[str, dict] = {}
        self._scheduler = DaemonScheduler()
        self._session_timeout_seconds = max(0.0, float(session_timeout_seconds))
        self._profile_max_age_seconds = max(0.0, float(profile_max_age_seconds))
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
        relays = self._normalize_relays(requested_relays)
        max_inflight = int(max_inflight_chunks)
        if max_inflight <= 0:
            return DaemonResponse(ok=False, error="max_inflight_chunks must be positive")
        now = time.time()
        with self._lock:
            self._reap_stale_sessions_locked(now)
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
                max_inflight_chunks=max_inflight,
                created_at=now,
                last_seen=now,
            )
            self._sessions[session_id] = session
            for gpu in relays:
                self._relay_quotas[gpu].sessions.add(session_id)
            return DaemonResponse(ok=True, payload={"session": asdict(session)})

    def register_job(
        self,
        job_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        container_id: str | None = None,
        process_id: int | None = None,
    ) -> DaemonResponse:
        job = JobIdentity(
            job_id=job_id,
            user_id=user_id,
            session_id=session_id,
            container_id=container_id,
            process_id=process_id,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            return DaemonResponse(ok=True, payload={"job": asdict(job)})

    def register_buffer(
        self,
        buffer_id: str,
        job_id: str,
        kind: str,
        size_bytes: int,
        device_index: int | None = None,
        address: int | None = None,
        pinned: bool = False,
    ) -> DaemonResponse:
        buffer = BufferRegistration(
            buffer_id=buffer_id,
            job_id=job_id,
            kind=kind,
            size_bytes=size_bytes,
            device_index=device_index,
            address=address,
            pinned=pinned,
        )
        with self._lock:
            if buffer.job_id not in self._jobs:
                return DaemonResponse(ok=False, error="unknown job")
            self._buffers[buffer.buffer_id] = buffer
            return DaemonResponse(ok=True, payload={"buffer": asdict(buffer)})

    def cleanup(
        self,
        target_kind: str,
        target_id: str,
        reason: str,
        force: bool = False,
    ) -> DaemonResponse:
        cleanup = CleanupRequest(
            target_kind=target_kind,
            target_id=target_id,
            reason=reason,
            force=force,
        )
        with self._lock:
            removed: dict[str, object] = {"jobs": 0, "buffers": 0, "sessions": 0, "reservations": 0}
            if cleanup.target_kind == "job":
                if cleanup.target_id not in self._jobs:
                    return DaemonResponse(ok=False, error="unknown job")
                self._jobs.pop(cleanup.target_id, None)
                removed["jobs"] = 1
                for buffer_id, buffer in list(self._buffers.items()):
                    if buffer.job_id == cleanup.target_id:
                        self._buffers.pop(buffer_id, None)
                        removed["buffers"] = int(removed["buffers"]) + 1
            elif cleanup.target_kind == "buffer":
                buffer = self._buffers.pop(cleanup.target_id, None)
                if buffer is None:
                    return DaemonResponse(ok=False, error="unknown buffer")
                removed["buffers"] = 1
            elif cleanup.target_kind == "session":
                session = self._close_session_locked(cleanup.target_id)
                if session is None:
                    return DaemonResponse(ok=False, error="unknown session")
                removed["sessions"] = 1
            elif cleanup.target_kind == "reservation":
                reservation = self._release_reservation_locked(
                    cleanup.target_id,
                    final_state=TransferStatusState.CANCELED,
                )
                if reservation is None:
                    return DaemonResponse(ok=False, error="unknown reservation")
                removed["reservations"] = 1
            else:
                return DaemonResponse(ok=False, error="unsupported cleanup target")
            self._cleanup_events.append(cleanup)
            return DaemonResponse(
                ok=True,
                payload={"cleanup": asdict(cleanup), "removed": removed},
            )

    def close_session(self, session_id: str) -> DaemonResponse:
        with self._lock:
            self._reap_stale_sessions_locked(time.time())
            session = self._close_session_locked(session_id)
            if session is None:
                return DaemonResponse(ok=False, error="unknown session")
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
        bytes_count = int(bytes_)
        direction_value = str(direction).lower()
        if chunks <= 0:
            return DaemonResponse(ok=False, error="chunks must be positive")
        if bytes_count < 0:
            return DaemonResponse(ok=False, error="bytes must be non-negative")
        if direction_value not in {"h2d", "d2h", "unknown"}:
            return DaemonResponse(ok=False, error="direction must be h2d, d2h, or unknown")
        now = time.time()
        with self._lock:
            self._reap_stale_sessions_locked(now)
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
                bytes=bytes_count,
                direction=direction_value,
            )
            self._reservations[reservation.reservation_id] = reservation
            session.active_chunks += chunks
            quota.active_chunks += chunks
            self._touch_session_locked(session_id, now)
            return DaemonResponse(ok=True, payload={"reservation": asdict(reservation)})

    def release_transfer(self, reservation_id: str) -> DaemonResponse:
        with self._lock:
            reservation = self._release_reservation_locked(reservation_id)
            if reservation is None:
                return DaemonResponse(ok=False, error="unknown reservation")
            return DaemonResponse(ok=True, payload={"reservation_id": reservation_id})

    def transfer_status(
        self,
        transfer_id: str,
        state: str | None = None,
        bytes_completed: int | None = None,
        error: str | None = None,
    ) -> DaemonResponse:
        with self._lock:
            status = self._transfer_statuses.get(str(transfer_id))
            if status is None:
                return DaemonResponse(ok=False, error="unknown transfer")
            if state is None and bytes_completed is None and error is None:
                return DaemonResponse(ok=True, payload={"status": asdict(status)})
            updated = TransferStatus(
                transfer_id=status.transfer_id,
                job_id=status.job_id,
                state=status.state if state is None else TransferStatusState(state),
                bytes_total=status.bytes_total,
                bytes_completed=(
                    status.bytes_completed
                    if bytes_completed is None
                    else int(bytes_completed)
                ),
                session_id=status.session_id,
                error=status.error if error is None else error,
            )
            self._transfer_statuses[updated.transfer_id] = updated
            return DaemonResponse(ok=True, payload={"status": asdict(updated)})

    def plan_transfer(
        self,
        session_id: str,
        total_bytes: int,
        chunk_bytes: int,
        mode: str = "pool",
        direction: str = "h2d",
        job_id: str | None = None,
    ) -> DaemonResponse:
        now = time.time()
        with self._lock:
            self._reap_stale_sessions_locked(now)
            self._purge_stale_profiles_locked(now)
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                return DaemonResponse(ok=False, error="unknown session")

            profile_entry = self._profile_cache.get(
                self._profile_key(session.target_gpu, session.relay_gpus)
            )
            decision = self._scheduler.plan_transfer(
                session=session,
                profile_entry=profile_entry,
                relay_quotas=self._relay_quotas,
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                mode=mode,
                direction=direction,
                now=now,
                job_id=job_id,
            )
            transfer_id = str(uuid.uuid4())
            reservations = self._commit_scheduler_leases_locked(
                session,
                decision,
                transfer_id=transfer_id,
            )
            status = TransferStatus(
                transfer_id=transfer_id,
                job_id=str(job_id or session.session_id),
                state=TransferStatusState.SUBMITTED,
                bytes_total=int(total_bytes),
                bytes_completed=0,
                session_id=session.session_id,
            )
            self._transfer_statuses[transfer_id] = status
            self._touch_session_locked(session.session_id, now)
            payload = decision.as_dict()
            payload["transfer_id"] = transfer_id
            payload["transfer_status"] = asdict(status)
            payload["reservations"] = [asdict(reservation) for reservation in reservations]
            return DaemonResponse(ok=True, payload=payload)

    def get_profile(self, target_gpu: int, relay_gpus: Iterable[int]) -> DaemonResponse:
        key = self._profile_key(target_gpu, relay_gpus)
        with self._lock:
            self._purge_stale_profiles_locked(time.time())
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
        relays = self._normalize_relays(relay_gpus)
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
            self._purge_stale_profiles_locked(time.time())
            self._profile_cache[key] = entry
        return DaemonResponse(ok=True, payload={"profile": dict(entry)})

    def invalidate_profile(self, target_gpu: int, relay_gpus: Iterable[int]) -> DaemonResponse:
        key = self._profile_key(target_gpu, relay_gpus)
        with self._lock:
            removed = self._profile_cache.pop(key, None)
            return DaemonResponse(
                ok=True,
                payload={
                    "profile_key": key,
                    "removed": removed is not None,
                },
            )

    def reap_stale_sessions(self, now: float | None = None) -> list[str]:
        with self._lock:
            return self._reap_stale_sessions_locked(time.time() if now is None else float(now))

    def _release_reservation_locked(
        self,
        reservation_id: str,
        final_state: TransferStatusState = TransferStatusState.COMPLETE,
    ) -> TransferReservation | None:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return None
        transfer_id = self._reservation_transfers.pop(reservation_id, None)
        session = self._sessions.get(reservation.session_id)
        if session is not None:
            session.active_chunks = max(0, session.active_chunks - reservation.chunks)
        quota = self._relay_quotas.get(reservation.relay_gpu)
        if quota is not None:
            quota.active_chunks = max(0, quota.active_chunks - reservation.chunks)
        if transfer_id is not None:
            self._mark_transfer_terminal_if_unblocked_locked(transfer_id, final_state)
        return reservation

    def _commit_scheduler_leases_locked(
        self,
        session: Session,
        decision: SchedulerDecision,
        transfer_id: str | None = None,
    ) -> list[TransferReservation]:
        reservations: list[TransferReservation] = []
        for lease in decision.leases:
            reservation = TransferReservation(
                reservation_id=lease.lease_id,
                session_id=lease.session_id,
                relay_gpu=lease.relay_device,
                chunks=lease.chunk_limit,
                bytes=lease.bytes_limit,
                direction=lease.direction,
            )
            self._reservations[reservation.reservation_id] = reservation
            session.active_chunks += reservation.chunks
            quota = self._relay_quotas.get(reservation.relay_gpu)
            if quota is not None:
                quota.active_chunks += reservation.chunks
            if transfer_id is not None:
                self._reservation_transfers[reservation.reservation_id] = transfer_id
            reservations.append(reservation)
        return reservations

    def _mark_transfer_terminal_if_unblocked_locked(
        self,
        transfer_id: str,
        final_state: TransferStatusState,
    ) -> None:
        if any(value == transfer_id for value in self._reservation_transfers.values()):
            return
        status = self._transfer_statuses.get(transfer_id)
        if status is None or status.state in {
            TransferStatusState.COMPLETE,
            TransferStatusState.FAILED,
            TransferStatusState.CANCELED,
        }:
            return
        completed = status.bytes_total if final_state is TransferStatusState.COMPLETE else status.bytes_completed
        self._transfer_statuses[transfer_id] = TransferStatus(
            transfer_id=status.transfer_id,
            job_id=status.job_id,
            state=final_state,
            bytes_total=status.bytes_total,
            bytes_completed=completed,
            session_id=status.session_id,
            error=status.error,
        )

    def _close_session_locked(self, session_id: str) -> Session | None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.active = False
        session.closed_at = time.time()
        for reservation_id, reservation in list(self._reservations.items()):
            if reservation.session_id == session_id:
                self._release_reservation_locked(
                    reservation_id,
                    final_state=TransferStatusState.CANCELED,
                )
        for gpu in session.relay_gpus:
            quota = self._relay_quotas.get(gpu)
            if quota is not None:
                quota.sessions.discard(session_id)
        return session

    def _touch_session_locked(self, session_id: str, now: float | None = None) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.last_seen = time.time() if now is None else float(now)

    def _reap_stale_sessions_locked(self, now: float) -> list[str]:
        if self._session_timeout_seconds <= 0.0:
            return []
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.active and session.last_seen > 0.0 and now - session.last_seen > self._session_timeout_seconds
        ]
        for session_id in expired:
            self._close_session_locked(session_id)
        return expired

    def _purge_stale_profiles_locked(self, now: float) -> list[str]:
        if self._profile_max_age_seconds <= 0.0:
            return []
        expired = [
            key
            for key, entry in self._profile_cache.items()
            if now - float(entry.get("updated_at", 0.0) or 0.0) > self._profile_max_age_seconds
        ]
        for key in expired:
            self._profile_cache.pop(key, None)
        return expired

    def describe(self) -> DaemonResponse:
        with self._lock:
            now = time.time()
            self._reap_stale_sessions_locked(now)
            self._purge_stale_profiles_locked(now)
            return DaemonResponse(
                ok=True,
                payload={
                    "jobs": {key: asdict(value) for key, value in self._jobs.items()},
                    "buffers": {key: asdict(value) for key, value in self._buffers.items()},
                    "sessions": {key: asdict(value) for key, value in self._sessions.items()},
                    "reservations": {
                        key: asdict(value) for key, value in self._reservations.items()
                    },
                    "transfer_statuses": {
                        key: asdict(value) for key, value in self._transfer_statuses.items()
                    },
                    "cleanup_events": [asdict(item) for item in self._cleanup_events],
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
        try:
            return self._handle_request(request)
        except (KeyError, TypeError, ValueError) as exc:
            return DaemonResponse(ok=False, error=f"invalid request: {exc}")

    def _handle_request(self, request: DaemonRequest) -> DaemonResponse:
        if request.request_type == RequestType.REGISTER_JOB:
            payload = request.payload
            return self.register_job(
                job_id=str(payload["job_id"]),
                user_id=payload.get("user_id"),
                session_id=payload.get("session_id"),
                container_id=payload.get("container_id"),
                process_id=payload.get("process_id"),
            )
        if request.request_type == RequestType.REGISTER_BUFFER:
            payload = request.payload
            return self.register_buffer(
                buffer_id=str(payload["buffer_id"]),
                job_id=str(payload["job_id"]),
                kind=str(payload.get("kind", "cpu_pinned")),
                size_bytes=int(payload.get("size_bytes", 0)),
                device_index=payload.get("device_index"),
                address=payload.get("address"),
                pinned=bool(payload.get("pinned", False)),
            )
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
        if request.request_type == RequestType.PLAN_TRANSFER:
            if request.session_id is None:
                return DaemonResponse(ok=False, error="session_id is required")
            payload = request.payload
            total_bytes = int(payload.get("total_bytes", payload.get("bytes", 0)))
            return self.plan_transfer(
                session_id=request.session_id,
                total_bytes=total_bytes,
                chunk_bytes=int(payload.get("chunk_bytes", 16 * 1024 * 1024)),
                mode=str(payload.get("mode", "pool")),
                direction=str(payload.get("direction", "h2d")),
                job_id=str(payload["job_id"]) if "job_id" in payload else None,
            )
        if request.request_type == RequestType.RELEASE_TRANSFER:
            payload = request.payload
            reservation_id = payload.get("reservation_id")
            if reservation_id is None:
                return DaemonResponse(ok=False, error="reservation_id is required")
            return self.release_transfer(str(reservation_id))
        if request.request_type == RequestType.TRANSFER_STATUS:
            payload = request.payload
            return self.transfer_status(
                transfer_id=str(payload["transfer_id"]),
                state=payload.get("state"),
                bytes_completed=payload.get("bytes_completed"),
                error=payload.get("error"),
            )
        if request.request_type == RequestType.CLEANUP:
            payload = request.payload
            return self.cleanup(
                target_kind=str(payload["target_kind"]),
                target_id=str(payload["target_id"]),
                reason=str(payload.get("reason", "manual")),
                force=bool(payload.get("force", False)),
            )
        if request.request_type == RequestType.INVALIDATE_PROFILE:
            payload = request.payload
            return self.invalidate_profile(
                target_gpu=int(payload["target_gpu"]),
                relay_gpus=payload.get("relay_gpus", []),
            )
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

    def handle_wire_message(self, data: bytes | str) -> DaemonResponse:
        try:
            text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            request_data = json.loads(text)
            if not isinstance(request_data, dict):
                raise ValueError("request must be a JSON object")
            payload = request_data.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            request = DaemonRequest(
                request_type=RequestType(request_data["request_type"]),
                session_id=request_data.get("session_id"),
                payload=payload,
            )
        except Exception as exc:
            return DaemonResponse(ok=False, error=f"invalid request: {exc}")
        return self.handle_request(request)

    @staticmethod
    def _profile_key(target_gpu: int, relay_gpus: Iterable[int]) -> str:
        relays = ",".join(str(gpu) for gpu in TurboBusDaemon._normalize_relays(relay_gpus))
        return f"target={int(target_gpu)};relays={relays}"

    @staticmethod
    def _normalize_relays(relay_gpus: Iterable[int]) -> list[int]:
        return sorted({int(gpu) for gpu in relay_gpus})

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
                    response = self.handle_wire_message(line)
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
