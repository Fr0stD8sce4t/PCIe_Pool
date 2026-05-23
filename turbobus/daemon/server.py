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
    LeaseToken,
    RelayQuota,
    RequestType,
    Session,
    TransferReservation,
    TransferStatus,
    TransferStatusState,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)
from .scheduler import DaemonScheduler, SchedulerDecision
from .topology import StaticTopologyProvider, TopologyProvider


_TERMINAL_TRANSFER_STATES = {
    TransferStatusState.COMPLETE,
    TransferStatusState.FAILED,
    TransferStatusState.CANCELED,
}


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
        topology_provider: TopologyProvider | None = None,
    ) -> None:
        relays = tuple(self._normalize_relays(relay_gpus))
        self._lock = threading.Lock()
        self._jobs: dict[str, JobIdentity] = {}
        self._buffers: dict[str, BufferRegistration] = {}
        self._sessions: dict[str, Session] = {}
        self._reservations: dict[str, TransferReservation] = {}
        self._reservation_transfers: dict[str, str] = {}
        self._transfer_plans: dict[str, dict[str, object]] = {}
        self._lease_tokens: dict[str, LeaseToken] = {}
        self._transfer_statuses: dict[str, TransferStatus] = {}
        self._cleanup_events: list[CleanupRequest] = []
        self._system_cleanup_events: list[CleanupRequest] = []
        self._profile_cache: dict[str, dict] = {}
        self._scheduler = DaemonScheduler()
        self._topology_provider = topology_provider or StaticTopologyProvider.from_relay_gpus(
            relays
        )
        self._session_timeout_seconds = max(0.0, float(session_timeout_seconds))
        self._profile_max_age_seconds = max(0.0, float(profile_max_age_seconds))
        self._relay_quotas = {
            int(gpu): RelayQuota(
                relay_gpu=int(gpu),
                max_sessions=max_sessions_per_relay,
                max_inflight_chunks=max_inflight_chunks_per_relay,
            )
            for gpu in relays
        }

    def get_inventory(self) -> DaemonResponse:
        inventory = self._topology_provider.snapshot()
        return DaemonResponse(ok=True, payload={"inventory": inventory.as_dict()})

    def discover_relays(
        self,
        target_gpu: int | None = None,
        requested_relays: Iterable[int] | None = None,
    ) -> DaemonResponse:
        now = time.time()
        target = None if target_gpu is None else int(target_gpu)
        with self._lock:
            self._reap_stale_sessions_locked(now)
            self._reap_expired_leases_locked(now)
            inventory = self._topology_provider.snapshot()
            candidates = (
                tuple(sorted(self._relay_quotas))
                if requested_relays is None
                else tuple(self._normalize_relays(requested_relays))
            )
            return DaemonResponse(
                ok=True,
                payload={
                    "relay_discovery": self._relay_discovery_snapshot_locked(
                        inventory=inventory,
                        target_gpu=target,
                        requested_relays=candidates,
                    )
                },
            )

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
            self._reap_expired_leases_locked(now)
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
        handle_type: str = "registered_buffer",
        metadata: dict[str, object] | None = None,
    ) -> DaemonResponse:
        buffer = BufferRegistration(
            buffer_id=buffer_id,
            job_id=job_id,
            kind=kind,
            size_bytes=size_bytes,
            device_index=device_index,
            address=address,
            pinned=pinned,
            handle_type=handle_type,
            metadata={} if metadata is None else metadata,
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
                session = self._close_session_locked(
                    cleanup.target_id,
                    reason=cleanup.reason,
                )
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
            session = self._close_session_locked(session_id, reason="session_closed")
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
            self._reap_expired_leases_locked(now)
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
            lease_token = self._issue_lease_token_locked(
                lease_id=reservation.reservation_id,
                session_id=session_id,
                relay_gpu=relay_gpu,
                now=now,
            )
            session.active_chunks += chunks
            quota.active_chunks += chunks
            self._touch_session_locked(session_id, now)
            return DaemonResponse(
                ok=True,
                payload={
                    "reservation": asdict(reservation),
                    "lease_token": asdict(lease_token),
                },
            )

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
            try:
                requested_state = (
                    status.state if state is None else TransferStatusState(state)
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            if status.state in _TERMINAL_TRANSFER_STATES:
                if (
                    requested_state == status.state
                    and _status_bytes_match(status, bytes_completed)
                    and (error is None or error == status.error)
                ):
                    return DaemonResponse(ok=True, payload={"status": asdict(status)})
                return DaemonResponse(
                    ok=False,
                    error="terminal transfer status cannot be updated",
                )
            try:
                updated = TransferStatus(
                    transfer_id=status.transfer_id,
                    job_id=status.job_id,
                    state=requested_state,
                    bytes_total=status.bytes_total,
                    bytes_completed=(
                        status.bytes_completed
                        if bytes_completed is None
                        else int(bytes_completed)
                    ),
                    session_id=status.session_id,
                    error=status.error if error is None else error,
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            self._transfer_statuses[updated.transfer_id] = updated
            return DaemonResponse(ok=True, payload={"status": asdict(updated)})

    def validate_lease(
        self,
        lease_id: str,
        token: str,
        session_id: str | None = None,
        relay_gpu: int | None = None,
        job_id: str | None = None,
        buffer_ids: Iterable[str] | None = None,
        now: float | None = None,
    ) -> DaemonResponse:
        checked_at = time.time() if now is None else float(now)
        with self._lock:
            self._reap_stale_sessions_locked(checked_at)
            lease = self._lease_tokens.get(str(lease_id))
            if lease is None:
                return DaemonResponse(ok=False, error="unknown lease")
            if lease.token != str(token):
                return DaemonResponse(ok=False, error="invalid lease token")
            if session_id is not None and lease.session_id != str(session_id):
                return DaemonResponse(ok=False, error="lease session mismatch")
            if relay_gpu is not None and lease.relay_gpu != int(relay_gpu):
                return DaemonResponse(ok=False, error="lease relay mismatch")
            if job_id is not None and lease.job_id != str(job_id):
                return DaemonResponse(ok=False, error="lease job mismatch")
            if buffer_ids is not None:
                requested_buffers = tuple(str(buffer_id) for buffer_id in buffer_ids)
                for buffer_id in requested_buffers:
                    if buffer_id not in lease.buffer_ids:
                        return DaemonResponse(ok=False, error="lease buffer mismatch")
                    buffer = self._buffers.get(buffer_id)
                    if buffer is None:
                        return DaemonResponse(ok=False, error="unknown buffer")
                    if lease.job_id is not None and buffer.job_id != lease.job_id:
                        return DaemonResponse(ok=False, error="lease buffer owner mismatch")
            if lease.expires_at and checked_at > lease.expires_at:
                self._release_expired_lease_locked(lease.lease_id)
                return DaemonResponse(ok=False, error="lease expired")
            if lease.lease_id not in self._reservations:
                return DaemonResponse(ok=False, error="lease is not active")
            return DaemonResponse(ok=True, payload={"lease_token": asdict(lease)})

    def authorize_worker_transfer(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> DaemonResponse:
        now = time.time()
        with self._lock:
            self._reap_stale_sessions_locked(now)
            status = self._transfer_statuses.get(request.transfer_id)
            if status is None:
                return DaemonResponse(ok=False, error="unknown transfer")
            if status.session_id != request.session_id:
                return DaemonResponse(ok=False, error="transfer session mismatch")
            if status.job_id != request.job_id:
                return DaemonResponse(ok=False, error="transfer job mismatch")
            lease = self._lease_tokens.get(request.lease_id)
            if lease is None:
                return DaemonResponse(ok=False, error="unknown lease")
            if lease.token != request.token:
                return DaemonResponse(ok=False, error="invalid lease token")
            if lease.session_id != request.session_id:
                return DaemonResponse(ok=False, error="lease session mismatch")
            if lease.job_id != request.job_id:
                return DaemonResponse(ok=False, error="lease job mismatch")
            if request.relay_gpu is not None and lease.relay_gpu != request.relay_gpu:
                return DaemonResponse(ok=False, error="lease relay mismatch")
            if lease.expires_at and now > lease.expires_at:
                self._release_expired_lease_locked(lease.lease_id)
                return DaemonResponse(ok=False, error="lease expired")
            if lease.lease_id not in self._reservations:
                return DaemonResponse(ok=False, error="lease is not active")
            reservation = self._reservations[lease.lease_id]
            if reservation.direction not in {"unknown", request.direction}:
                return DaemonResponse(ok=False, error="reservation direction mismatch")
            plan = self._transfer_plans.get(request.transfer_id)
            if plan is None:
                return DaemonResponse(ok=False, error="transfer plan is unavailable")
            try:
                authorized_ranges = _relay_ranges_from_plan(
                    plan,
                    relay_gpu=lease.relay_gpu,
                    direction=request.direction,
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            if request.ranges and request.ranges != authorized_ranges:
                return DaemonResponse(ok=False, error="worker ranges do not match daemon plan")
            requested_bytes = sum(item["bytes"] for item in authorized_ranges)
            if requested_bytes > reservation.bytes:
                return DaemonResponse(ok=False, error="authorization exceeds reservation bytes")
            required_buffers = (request.src_buffer_id, request.dst_buffer_id)
            for buffer_id in required_buffers:
                if buffer_id not in lease.buffer_ids:
                    return DaemonResponse(ok=False, error="lease buffer mismatch")
            src_buffer = self._buffers.get(request.src_buffer_id)
            dst_buffer = self._buffers.get(request.dst_buffer_id)
            if src_buffer is None or dst_buffer is None:
                return DaemonResponse(ok=False, error="unknown buffer")
            authorization = WorkerTransferAuthorization(
                transfer_id=request.transfer_id,
                lease_id=request.lease_id,
                session_id=request.session_id,
                job_id=request.job_id,
                src_buffer=src_buffer,
                dst_buffer=dst_buffer,
                direction=request.direction,
                ranges=authorized_ranges,
                relay_gpu=lease.relay_gpu,
                plan=plan,
            )
            return DaemonResponse(
                ok=True,
                payload={"authorization": asdict(authorization)},
            )

    def plan_transfer(
        self,
        session_id: str,
        total_bytes: int,
        chunk_bytes: int,
        mode: str = "pool",
        direction: str = "h2d",
        job_id: str | None = None,
        buffer_ids: Iterable[str] | None = None,
        ranges: Iterable[dict[str, int]] | None = None,
    ) -> DaemonResponse:
        now = time.time()
        try:
            normalized_ranges = _normalize_transfer_ranges(ranges)
            if normalized_ranges is not None:
                range_bytes = sum(item["bytes"] for item in normalized_ranges)
                if range_bytes != int(total_bytes):
                    return DaemonResponse(
                        ok=False,
                        error="range bytes must match total_bytes",
                    )
        except (KeyError, TypeError, ValueError) as exc:
            return DaemonResponse(ok=False, error=str(exc))
        with self._lock:
            self._reap_stale_sessions_locked(now)
            self._reap_expired_leases_locked(now)
            self._purge_stale_profiles_locked(now)
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                return DaemonResponse(ok=False, error="unknown session")
            buffer_ids_tuple = self._validate_transfer_buffers_locked(
                buffer_ids,
                job_id=job_id,
                session_id=session.session_id,
            )
            relay_eligibility = self._relay_eligibility_for_session_locked(session)
            planning_relays = tuple(
                item["relay_gpu"] for item in relay_eligibility["eligible_relays"]
            )

            profile_entry = self._profile_cache.get(
                self._profile_key(session.target_gpu, planning_relays)
            )
            if profile_entry is None and planning_relays != tuple(session.relay_gpus):
                profile_entry = self._profile_cache.get(
                    self._profile_key(session.target_gpu, session.relay_gpus)
                )
            planning_session = (
                session
                if planning_relays == tuple(session.relay_gpus)
                else Session(
                    session_id=session.session_id,
                    target_gpu=session.target_gpu,
                    relay_gpus=list(planning_relays),
                    max_inflight_chunks=session.max_inflight_chunks,
                    active_chunks=session.active_chunks,
                    active=session.active,
                    created_at=session.created_at,
                    last_seen=session.last_seen,
                    closed_at=session.closed_at,
                )
            )
            decision = self._scheduler.plan_transfer(
                session=planning_session,
                profile_entry=profile_entry,
                relay_quotas=self._relay_quotas,
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                ranges=normalized_ranges,
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
                buffer_ids=buffer_ids_tuple,
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
            self._transfer_plans[transfer_id] = decision.plan.as_dict()
            self._touch_session_locked(session.session_id, now)
            payload = decision.as_dict()
            payload["transfer_id"] = transfer_id
            payload["transfer_status"] = asdict(status)
            payload["planning"] = {
                "target_gpu": session.target_gpu,
                "profile_key": self._profile_key(session.target_gpu, planning_relays),
                "relay_eligibility": relay_eligibility,
            }
            payload["reservations"] = [asdict(reservation) for reservation in reservations]
            payload["lease_tokens"] = [
                asdict(self._lease_tokens[reservation.reservation_id])
                for reservation in reservations
                if reservation.reservation_id in self._lease_tokens
            ]
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

    def reap_expired_leases(self, now: float | None = None) -> list[str]:
        with self._lock:
            return self._reap_expired_leases_locked(
                time.time() if now is None else float(now)
            )

    def _release_reservation_locked(
        self,
        reservation_id: str,
        final_state: TransferStatusState = TransferStatusState.COMPLETE,
        cleanup_reason: str | None = None,
    ) -> TransferReservation | None:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is None:
            return None
        self._lease_tokens.pop(reservation_id, None)
        transfer_id = self._reservation_transfers.pop(reservation_id, None)
        session = self._sessions.get(reservation.session_id)
        if session is not None:
            session.active_chunks = max(0, session.active_chunks - reservation.chunks)
        quota = self._relay_quotas.get(reservation.relay_gpu)
        if quota is not None:
            quota.active_chunks = max(0, quota.active_chunks - reservation.chunks)
        if transfer_id is not None:
            self._mark_transfer_terminal_if_unblocked_locked(transfer_id, final_state)
        if final_state is TransferStatusState.CANCELED and cleanup_reason is not None:
            self._system_cleanup_events.append(
                CleanupRequest(
                    target_kind="reservation",
                    target_id=reservation_id,
                    reason=cleanup_reason,
                    force=True,
                )
            )
        return reservation

    def _commit_scheduler_leases_locked(
        self,
        session: Session,
        decision: SchedulerDecision,
        transfer_id: str | None = None,
        buffer_ids: tuple[str, ...] = (),
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
            self._lease_tokens[reservation.reservation_id] = LeaseToken(
                lease_id=reservation.reservation_id,
                session_id=reservation.session_id,
                relay_gpu=reservation.relay_gpu,
                token=str(uuid.uuid4()),
                buffer_ids=buffer_ids,
                job_id=lease.job_id,
                issued_at=lease.granted_at,
                expires_at=lease.expires_at,
            )
            session.active_chunks += reservation.chunks
            quota = self._relay_quotas.get(reservation.relay_gpu)
            if quota is not None:
                quota.active_chunks += reservation.chunks
            if transfer_id is not None:
                self._reservation_transfers[reservation.reservation_id] = transfer_id
            reservations.append(reservation)
        return reservations

    def _issue_lease_token_locked(
        self,
        lease_id: str,
        session_id: str,
        relay_gpu: int,
        now: float,
        job_id: str | None = None,
        expires_at: float = 0.0,
    ) -> LeaseToken:
        lease_token = LeaseToken(
            lease_id=lease_id,
            session_id=session_id,
            relay_gpu=relay_gpu,
            token=str(uuid.uuid4()),
            job_id=job_id,
            issued_at=float(now),
            expires_at=float(expires_at),
        )
        self._lease_tokens[lease_token.lease_id] = lease_token
        return lease_token

    def _validate_transfer_buffers_locked(
        self,
        buffer_ids: Iterable[str] | None,
        job_id: str | None,
        session_id: str,
    ) -> tuple[str, ...]:
        if buffer_ids is None:
            return ()
        normalized = tuple(str(buffer_id) for buffer_id in buffer_ids)
        if not normalized:
            return ()
        if any(not buffer_id.strip() for buffer_id in normalized):
            raise ValueError("buffer_ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("buffer_ids must be unique")
        owner_job_id = job_id
        if owner_job_id is None:
            for job in self._jobs.values():
                if job.session_id == session_id:
                    owner_job_id = job.job_id
                    break
        if owner_job_id is None:
            raise ValueError("job_id is required when buffer_ids are provided")
        job = self._jobs.get(str(owner_job_id))
        if job is None:
            raise ValueError("unknown job")
        if job.session_id is not None and job.session_id != session_id:
            raise ValueError("job session does not match transfer session")
        for buffer_id in normalized:
            buffer = self._buffers.get(buffer_id)
            if buffer is None:
                raise ValueError(f"unknown buffer: {buffer_id}")
            if buffer.job_id != str(owner_job_id):
                raise ValueError("buffer owner does not match job")
        return normalized

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

    def _close_session_locked(
        self,
        session_id: str,
        reason: str = "session_closed",
    ) -> Session | None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.active = False
        session.closed_at = time.time()
        self._system_cleanup_events.append(
            CleanupRequest(
                target_kind="session",
                target_id=session_id,
                reason=reason,
                force=True,
            )
        )
        for reservation_id, reservation in list(self._reservations.items()):
            if reservation.session_id == session_id:
                self._release_reservation_locked(
                    reservation_id,
                    final_state=TransferStatusState.CANCELED,
                    cleanup_reason=reason,
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
            self._close_session_locked(session_id, reason="stale_session_timeout")
        return expired

    def _reap_expired_leases_locked(self, now: float) -> list[str]:
        expired = [
            lease_id
            for lease_id, lease in self._lease_tokens.items()
            if lease.expires_at and float(now) > lease.expires_at
        ]
        for lease_id in expired:
            self._release_expired_lease_locked(lease_id)
        return expired

    def _release_expired_lease_locked(self, lease_id: str) -> TransferReservation | None:
        reservation = self._release_reservation_locked(
            lease_id,
            final_state=TransferStatusState.CANCELED,
            cleanup_reason="lease_expired",
        )
        if reservation is None:
            self._lease_tokens.pop(lease_id, None)
        return reservation

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
            self._reap_expired_leases_locked(now)
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
                    "system_cleanup_events": [
                        asdict(item) for item in self._system_cleanup_events
                    ],
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
                handle_type=str(payload.get("handle_type", "registered_buffer")),
                metadata=payload.get("metadata") or {},
            )
        if request.request_type == RequestType.GET_INVENTORY:
            return self.get_inventory()
        if request.request_type == RequestType.DISCOVER_RELAYS:
            payload = request.payload
            target_gpu = payload.get("target_gpu")
            return self.discover_relays(
                target_gpu=None if target_gpu is None else int(target_gpu),
                requested_relays=payload.get("relay_gpus"),
            )
        if request.request_type == RequestType.REAP_EXPIRED_LEASES:
            payload = request.payload
            expired = self.reap_expired_leases(now=payload.get("now"))
            return DaemonResponse(
                ok=True,
                payload={
                    "expired_lease_ids": expired,
                    "expired_count": len(expired),
                },
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
                buffer_ids=payload.get("buffer_ids"),
                ranges=payload.get("ranges"),
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
        if request.request_type == RequestType.VALIDATE_LEASE:
            payload = request.payload
            return self.validate_lease(
                lease_id=str(payload["lease_id"]),
                token=str(payload["token"]),
                session_id=payload.get("session_id"),
                relay_gpu=payload.get("relay_gpu"),
                job_id=payload.get("job_id"),
                buffer_ids=payload.get("buffer_ids"),
            )
        if request.request_type == RequestType.AUTHORIZE_WORKER_TRANSFER:
            return self.authorize_worker_transfer(
                WorkerTransferAuthorizationRequest(**request.payload)
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

    def _eligible_relays_for_session_locked(self, session: Session) -> tuple[int, ...]:
        relay_eligibility = self._relay_eligibility_for_session_locked(session)
        return tuple(item["relay_gpu"] for item in relay_eligibility["eligible_relays"])

    def _relay_eligibility_for_session_locked(self, session: Session) -> dict[str, object]:
        return self._relay_eligibility_for_target_locked(
            target_gpu=session.target_gpu,
            requested_relays=session.relay_gpus,
        )

    def _relay_eligibility_for_target_locked(
        self,
        target_gpu: int,
        requested_relays: Iterable[int],
        inventory=None,
    ) -> dict[str, object]:
        if inventory is None:
            inventory = self._topology_provider.snapshot()
        relay_eligibility = inventory.relay_eligibility(
            target_device=int(target_gpu),
            requested_relays=requested_relays,
        )
        eligible_relays = []
        filtered_relays = list(relay_eligibility["filtered_relays"])
        for item in relay_eligibility["eligible_relays"]:
            relay_gpu = int(item["relay_gpu"])
            if relay_gpu in self._relay_quotas:
                eligible_relays.append({"relay_gpu": relay_gpu, "reason": "eligible"})
            else:
                filtered_relays.append(
                    {"relay_gpu": relay_gpu, "reason": "relay not configured"}
                )
        return {
            **relay_eligibility,
            "eligible_relays": eligible_relays,
            "filtered_relays": filtered_relays,
        }

    def _relay_discovery_snapshot_locked(
        self,
        *,
        inventory,
        target_gpu: int | None,
        requested_relays: Iterable[int],
    ) -> dict[str, object]:
        candidates = tuple(self._normalize_relays(requested_relays))
        if target_gpu is None:
            relay_eligibility = {
                "requested_relays": list(candidates),
                "eligible_relays": [],
                "filtered_relays": [],
                "inventory_source": inventory.source,
                "inventory_discovered_at": inventory.discovered_at,
            }
            eligibility_by_relay = {
                relay_gpu: {
                    "eligible": None,
                    "reason": "target_gpu not provided",
                }
                for relay_gpu in candidates
            }
        else:
            relay_eligibility = self._relay_eligibility_for_target_locked(
                target_gpu=target_gpu,
                requested_relays=candidates,
                inventory=inventory,
            )
            eligibility_by_relay = {}
            for item in relay_eligibility["eligible_relays"]:
                eligibility_by_relay[int(item["relay_gpu"])] = {
                    "eligible": True,
                    "reason": str(item.get("reason", "eligible")),
                }
            for item in relay_eligibility["filtered_relays"]:
                eligibility_by_relay[int(item["relay_gpu"])] = {
                    "eligible": False,
                    "reason": str(item.get("reason", "filtered")),
                }

        relay_records = [
            self._relay_discovery_record_locked(
                relay_gpu=relay_gpu,
                inventory=inventory,
                target_gpu=target_gpu,
                eligibility=eligibility_by_relay.get(
                    relay_gpu,
                    {"eligible": False, "reason": "not requested"},
                ),
            )
            for relay_gpu in candidates
        ]
        return {
            "target_gpu": target_gpu,
            "requested_relays": list(candidates),
            "inventory_source": inventory.source,
            "inventory_discovered_at": inventory.discovered_at,
            "relay_eligibility": relay_eligibility,
            "relays": relay_records,
            "summary": {
                "relay_count": len(relay_records),
                "configured_relay_count": sum(
                    1 for item in relay_records if item["configured"]
                ),
                "eligible_relay_count": sum(
                    1
                    for item in relay_records
                    if item["eligibility"]["eligible"] is True
                ),
                "active_session_count": sum(
                    int(item["quota"]["active_sessions"])
                    for item in relay_records
                    if item["quota"] is not None
                ),
                "active_reservation_count": sum(
                    len(item["reservations"]) for item in relay_records
                ),
                "active_lease_count": sum(len(item["leases"]) for item in relay_records),
            },
        }

    def _relay_discovery_record_locked(
        self,
        *,
        relay_gpu: int,
        inventory,
        target_gpu: int | None,
        eligibility: dict[str, object],
    ) -> dict[str, object]:
        quota = self._relay_quotas.get(relay_gpu)
        return {
            "relay_gpu": relay_gpu,
            "configured": quota is not None,
            "eligibility": {
                "target_gpu": target_gpu,
                "eligible": eligibility["eligible"],
                "reason": eligibility["reason"],
            },
            "inventory": self._relay_inventory_record(inventory, relay_gpu, target_gpu),
            "quota": (
                None
                if quota is None
                else {
                    "relay_gpu": quota.relay_gpu,
                    "max_sessions": quota.max_sessions,
                    "active_sessions": len(quota.sessions),
                    "available_sessions": max(
                        0,
                        quota.max_sessions - len(quota.sessions),
                    ),
                    "max_inflight_chunks": quota.max_inflight_chunks,
                    "active_chunks": quota.active_chunks,
                    "available_chunks": max(
                        0,
                        quota.max_inflight_chunks - quota.active_chunks,
                    ),
                }
            ),
            "sessions": self._relay_session_records_locked(relay_gpu),
            "reservations": self._relay_reservation_records_locked(relay_gpu),
            "leases": self._relay_lease_records_locked(relay_gpu),
        }

    def _relay_inventory_record(
        self,
        inventory,
        relay_gpu: int,
        target_gpu: int | None,
    ) -> dict[str, object]:
        relay = int(relay_gpu)
        target = None if target_gpu is None else int(target_gpu)
        fabric_links = []
        for link in inventory.fabric_links:
            touches_relay = link.src_device_id == relay or link.dst_device_id == relay
            touches_target = (
                target is None
                or (link.src_device_id == relay and link.dst_device_id == target)
                or (
                    link.bidirectional
                    and link.src_device_id == target
                    and link.dst_device_id == relay
                )
            )
            if touches_relay and touches_target:
                fabric_links.append(asdict(link))
        return {
            "gpus": [
                asdict(gpu) for gpu in inventory.gpus if gpu.device_id == relay
            ],
            "pcie_paths": [
                asdict(path)
                for path in inventory.pcie_paths
                if path.device_id == relay
            ],
            "fabric_links": fabric_links,
        }

    def _relay_session_records_locked(self, relay_gpu: int) -> list[dict[str, object]]:
        quota = self._relay_quotas.get(relay_gpu)
        if quota is None:
            return []
        records = []
        for session_id in sorted(quota.sessions):
            session = self._sessions.get(session_id)
            if session is None:
                continue
            records.append(
                {
                    "session_id": session.session_id,
                    "target_gpu": session.target_gpu,
                    "active": session.active,
                    "active_chunks": session.active_chunks,
                    "max_inflight_chunks": session.max_inflight_chunks,
                    "job_ids": sorted(
                        job.job_id
                        for job in self._jobs.values()
                        if job.session_id == session.session_id
                    ),
                }
            )
        return records

    def _relay_reservation_records_locked(
        self,
        relay_gpu: int,
    ) -> list[dict[str, object]]:
        records = []
        for reservation_id, reservation in sorted(self._reservations.items()):
            if reservation.relay_gpu != relay_gpu:
                continue
            lease = self._lease_tokens.get(reservation_id)
            record = asdict(reservation)
            record["transfer_id"] = self._reservation_transfers.get(reservation_id)
            record["job_id"] = None if lease is None else lease.job_id
            records.append(record)
        return records

    def _relay_lease_records_locked(self, relay_gpu: int) -> list[dict[str, object]]:
        records = []
        for lease_id, lease in sorted(self._lease_tokens.items()):
            if lease.relay_gpu != relay_gpu:
                continue
            if lease_id not in self._reservations:
                continue
            records.append(
                {
                    "lease_id": lease.lease_id,
                    "session_id": lease.session_id,
                    "relay_gpu": lease.relay_gpu,
                    "job_id": lease.job_id,
                    "buffer_ids": lease.buffer_ids,
                    "issued_at": lease.issued_at,
                    "expires_at": lease.expires_at,
                    "transfer_id": self._reservation_transfers.get(lease_id),
                }
            )
        return records

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


def _relay_ranges_from_plan(
    plan: dict[str, object],
    *,
    relay_gpu: int,
    direction: str,
) -> tuple[dict[str, int], ...]:
    if not isinstance(plan, dict):
        raise ValueError("transfer plan is unavailable")
    ranges: list[dict[str, int]] = []
    relay = int(relay_gpu)
    requested_direction = str(direction).lower()
    for assignment in plan.get("assignments", ()) or ():
        if not isinstance(assignment, dict):
            raise ValueError("transfer plan assignment must be an object")
        path = assignment.get("path")
        if not isinstance(path, dict):
            raise ValueError("transfer plan assignment path must be an object")
        if str(path.get("kind", "")).lower() != "relay":
            continue
        if str(path.get("direction", "")).lower() != requested_direction:
            continue
        if int(path.get("relay_device", -1)) != relay:
            continue
        for chunk in assignment.get("chunks", ()) or ():
            if not isinstance(chunk, dict):
                raise ValueError("transfer plan chunk must be an object")
            ranges.append(
                {
                    "src_offset": int(chunk["src_offset"]),
                    "dst_offset": int(chunk["dst_offset"]),
                    "bytes": int(chunk["bytes"]),
                }
            )
    if not ranges:
        raise ValueError("daemon plan has no authorized relay chunks")
    return tuple(ranges)


def _normalize_transfer_ranges(
    ranges: Iterable[dict[str, int]] | None,
) -> tuple[dict[str, int], ...] | None:
    if ranges is None:
        return None
    normalized: list[dict[str, int]] = []
    for item in ranges:
        if not isinstance(item, dict):
            raise ValueError("transfer ranges must be objects")
        src_offset = int(item["src_offset"])
        dst_offset = int(item["dst_offset"])
        bytes_count = int(item["bytes"])
        if src_offset < 0 or dst_offset < 0:
            raise ValueError("range offsets must be non-negative")
        if bytes_count <= 0:
            raise ValueError("range bytes must be positive")
        normalized.append(
            {
                "src_offset": src_offset,
                "dst_offset": dst_offset,
                "bytes": bytes_count,
            }
        )
    return tuple(normalized)


def _status_bytes_match(
    status: TransferStatus,
    bytes_completed: int | None,
) -> bool:
    if bytes_completed is None:
        return True
    try:
        return int(bytes_completed) == status.bytes_completed
    except (TypeError, ValueError):
        return False
