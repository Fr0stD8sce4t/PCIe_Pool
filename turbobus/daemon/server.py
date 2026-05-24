from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
import uuid
from dataclasses import asdict
from typing import Iterable, Mapping

from .protocol import (
    BufferRegistration,
    CleanupRequest,
    DaemonRequest,
    DaemonResponse,
    JobIdentity,
    LeaseToken,
    PeerIdentity,
    RelayQuota,
    RequestType,
    Session,
    TransferReservation,
    TransferStatus,
    TransferStatusState,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)
from ..schema import ExecutionTicket, TransferIntent, TransferReceipt
from ..topology import TopologyProvider
from ..scheduler import (
    DaemonScheduler,
    SchedulingDecision,
    scheduling_decision_leases,
    scheduling_decision_stats,
)


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
        self._job_peer_identities: dict[str, PeerIdentity] = {}
        self._session_peer_identities: dict[str, PeerIdentity] = {}
        self._buffers: dict[str, BufferRegistration] = {}
        self._sessions: dict[str, Session] = {}
        self._reservations: dict[str, TransferReservation] = {}
        self._reservation_transfers: dict[str, str] = {}
        self._transfer_intents: dict[str, TransferIntent] = {}
        self._intent_transfers: dict[str, str] = {}
        self._transfer_queue: list[str] = []
        self._transfer_queue_records: dict[str, dict[str, object]] = {}
        self._runtime_state_version = 0
        self._transfer_plans: dict[str, dict[str, object]] = {}
        self._scheduling_decisions: dict[str, SchedulingDecision] = {}
        self._execution_tickets: dict[str, ExecutionTicket] = {}
        self._transfer_tickets: dict[str, str] = {}
        self._lease_tokens: dict[str, LeaseToken] = {}
        self._transfer_statuses: dict[str, TransferStatus] = {}
        self._staging_records: dict[str, dict[str, object]] = {}
        self._audit_records: list[dict[str, object]] = []
        self._connection_scoped_sessions: set[str] = set()
        self._connection_scoped_session_connections: dict[str, str] = {}
        self._cleanup_events: list[CleanupRequest] = []
        self._system_cleanup_events: list[CleanupRequest] = []
        self._profile_cache: dict[str, dict] = {}
        self._scheduler = DaemonScheduler()
        self._topology_provider = topology_provider
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
        if self._topology_provider is None:
            return _topology_unavailable_response()
        inventory = self._topology_provider.snapshot()
        return DaemonResponse(
            ok=True,
            payload={
                "inventory": inventory.as_dict(),
                "topology_snapshot": asdict(inventory.to_topology_snapshot()),
            },
        )

    def invalidate_topology(self) -> DaemonResponse:
        if self._topology_provider is None:
            return _topology_unavailable_response()
        invalidate = getattr(self._topology_provider, "invalidate", None)
        if not callable(invalidate):
            return DaemonResponse(
                ok=False,
                error="topology provider does not support invalidation",
            )
        try:
            invalidate()
        except NotImplementedError:
            return DaemonResponse(
                ok=False,
                error="topology provider does not support invalidation",
            )
        inventory = self._topology_provider.snapshot()
        return DaemonResponse(
            ok=True,
            payload={
                "topology_snapshot_id": inventory.topology_snapshot_id(),
                "topology_version": inventory.version,
                "inventory_source": inventory.source,
                "inventory_discovered_at": inventory.discovered_at,
                "inventory": inventory.as_dict(),
                "topology_snapshot": asdict(inventory.to_topology_snapshot()),
            },
        )

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
            if self._topology_provider is None:
                return _topology_unavailable_response()
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
        peer_identity: PeerIdentity | None = None,
        connection_scoped: bool = False,
        connection_id: str | None = None,
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
            if peer_identity is not None:
                self._session_peer_identities[session_id] = peer_identity
            if connection_scoped:
                self._connection_scoped_sessions.add(session_id)
                if connection_id is not None:
                    self._connection_scoped_session_connections[session_id] = str(connection_id)
            for gpu in relays:
                self._relay_quotas[gpu].sessions.add(session_id)
            payload = {"session": asdict(session)}
            if peer_identity is not None:
                payload["peer_identity"] = asdict(peer_identity)
            if connection_scoped:
                payload["connection_scoped"] = True
            return DaemonResponse(ok=True, payload=payload)

    def register_job(
        self,
        job_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        container_id: str | None = None,
        process_id: int | None = None,
        weight: float = 1.0,
        peer_identity: PeerIdentity | None = None,
    ) -> DaemonResponse:
        try:
            user_id, process_id, container_id = _bind_job_identity_to_peer(
                user_id=user_id,
                process_id=process_id,
                container_id=container_id,
                peer_identity=peer_identity,
            )
        except ValueError as exc:
            return DaemonResponse(ok=False, error=str(exc))
        job = JobIdentity(
            job_id=job_id,
            user_id=user_id,
            session_id=session_id,
            container_id=container_id,
            process_id=process_id,
            weight=weight,
        )
        with self._lock:
            if job.session_id is not None and job.session_id not in self._sessions:
                return DaemonResponse(ok=False, error="unknown session")
            session_peer = (
                None
                if job.session_id is None
                else self._session_peer_identities.get(job.session_id)
            )
            try:
                _validate_peer_owner_match(
                    expected=session_peer,
                    actual=peer_identity,
                    owner_name="session",
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            self._jobs[job.job_id] = job
            if peer_identity is not None:
                self._job_peer_identities[job.job_id] = peer_identity
            payload = {"job": asdict(job)}
            if peer_identity is not None:
                payload["peer_identity"] = asdict(peer_identity)
            return DaemonResponse(ok=True, payload=payload)

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
        peer_identity: PeerIdentity | None = None,
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
            now = time.time()
            self._reap_stale_sessions_locked(now)
            self._reap_expired_leases_locked(now)
            if buffer.job_id not in self._jobs:
                return DaemonResponse(ok=False, error="unknown job")
            try:
                self._validate_peer_owns_job_locked(
                    job_id=buffer.job_id,
                    peer_identity=peer_identity,
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            if self._active_buffer_lease_ids_locked(buffer.buffer_id):
                return DaemonResponse(ok=False, error="buffer has active lease")
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
            removed = _empty_removed_summary()
            if cleanup.target_kind == "job":
                if cleanup.target_id not in self._jobs and not cleanup.force:
                    return DaemonResponse(ok=False, error="unknown job")
                _merge_removed(
                    removed,
                    self._cleanup_job_locked(
                        cleanup.target_id,
                        reason=cleanup.reason,
                    ),
                )
            elif cleanup.target_kind == "buffer":
                if cleanup.target_id not in self._buffers and not cleanup.force:
                    return DaemonResponse(ok=False, error="unknown buffer")
                transfer_ids = self._transfer_ids_for_buffer_locked(cleanup.target_id)
                for lease_id in self._active_buffer_lease_ids_locked(cleanup.target_id):
                    _merge_removed(
                        removed,
                        self._release_reservation_and_count_locked(
                            lease_id,
                            final_state=TransferStatusState.CANCELED,
                            cleanup_reason=cleanup.reason,
                        ),
                    )
                buffer = self._buffers.pop(cleanup.target_id, None)
                if buffer is not None:
                    removed["buffers"] = int(removed["buffers"]) + 1
                for transfer_id in transfer_ids:
                    status = self._transfer_statuses.get(transfer_id)
                    if status is None or status.state in _TERMINAL_TRANSFER_STATES:
                        continue
                    self._mark_transfer_terminal_locked(
                        transfer_id,
                        TransferStatusState.CANCELED,
                        error=cleanup.reason,
                    )
                    removed["transfers"] = int(removed["transfers"]) + 1
            elif cleanup.target_kind == "session":
                session = self._close_session_locked(
                    cleanup.target_id,
                    reason=cleanup.reason,
                    removed=removed,
                )
                if session is None and not cleanup.force:
                    return DaemonResponse(ok=False, error="unknown session")
            elif cleanup.target_kind == "reservation":
                released = self._release_reservation_and_count_locked(
                    cleanup.target_id,
                    final_state=TransferStatusState.CANCELED,
                    cleanup_reason=cleanup.reason,
                )
                if (
                    int(released["reservations"]) == 0
                    and int(released["staging_records"]) == 0
                    and not cleanup.force
                ):
                    return DaemonResponse(ok=False, error="unknown reservation")
                _merge_removed(removed, released)
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
            removed = _empty_removed_summary()
            session = self._close_session_locked(
                session_id,
                reason="session_closed",
                removed=removed,
            )
            if session is None:
                return DaemonResponse(ok=False, error="unknown session")
            return DaemonResponse(
                ok=True,
                payload={"session_id": session_id, "removed": removed},
            )

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
            reservation_key = str(reservation_id)
            transfer_id = self._reservation_transfers.get(reservation_key)
            if transfer_id is not None:
                status = self._transfer_statuses.get(transfer_id)
                if status is None:
                    return DaemonResponse(ok=False, error="unknown transfer")
                if status.state is not TransferStatusState.COMPLETE:
                    return DaemonResponse(ok=False, error="transfer is not complete")
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
                if requested_state is TransferStatusState.COMPLETE:
                    mismatch = str(exc)
                    failed = self._mark_transfer_terminal_locked(
                        status.transfer_id,
                        TransferStatusState.FAILED,
                        error=mismatch,
                    )
                    self._append_transfer_audit_records_locked(
                        event_type="detected_mismatch",
                        transfer_id=status.transfer_id,
                        state=TransferStatusState.FAILED,
                        reason="transfer_status_mismatch",
                        failure_reason=mismatch,
                    )
                    removed = self._release_reservations_for_transfer_locked(
                        status.transfer_id,
                        final_state=TransferStatusState.FAILED,
                        cleanup_reason="transfer_status_mismatch",
                    )
                    self._refresh_transfer_queue_record_locked(status.transfer_id)
                    return DaemonResponse(
                        ok=False,
                        error=mismatch,
                        payload={"status": asdict(failed), "removed": removed},
                    )
                return DaemonResponse(ok=False, error=str(exc))
            self._transfer_statuses[updated.transfer_id] = updated
            self._refresh_transfer_queue_record_locked(updated.transfer_id)
            removed = _empty_removed_summary()
            if updated.state is TransferStatusState.COMPLETE:
                self._append_transfer_audit_records_locked(
                    event_type="worker_completion",
                    transfer_id=updated.transfer_id,
                    state=updated.state,
                    bytes_completed=updated.bytes_completed,
                )
            elif updated.state is TransferStatusState.FAILED:
                self._append_transfer_audit_records_locked(
                    event_type="worker_failure",
                    transfer_id=updated.transfer_id,
                    state=updated.state,
                    reason=updated.error or "worker_failed",
                    failure_reason=updated.error or "worker_failed",
                    bytes_completed=updated.bytes_completed,
                )
                _merge_removed(
                    removed,
                    self._release_reservations_for_transfer_locked(
                        updated.transfer_id,
                        final_state=TransferStatusState.FAILED,
                        cleanup_reason=updated.error or "worker_failed",
                    ),
                )
            elif updated.state is TransferStatusState.CANCELED:
                self._append_transfer_audit_records_locked(
                    event_type="transfer_canceled",
                    transfer_id=updated.transfer_id,
                    state=updated.state,
                    reason=updated.error or "transfer_canceled",
                    failure_reason=updated.error or "transfer_canceled",
                    bytes_completed=updated.bytes_completed,
                )
                _merge_removed(
                    removed,
                    self._release_reservations_for_transfer_locked(
                        updated.transfer_id,
                        final_state=TransferStatusState.CANCELED,
                        cleanup_reason=updated.error or "transfer_canceled",
                    ),
                )
            return DaemonResponse(
                ok=True,
                payload={"status": asdict(updated), "removed": removed},
            )

    def submit_transfer_intent(
        self,
        intent: TransferIntent,
        peer_identity: PeerIdentity | None = None,
    ) -> DaemonResponse:
        if not isinstance(intent, TransferIntent):
            return DaemonResponse(ok=False, error="intent must be a TransferIntent")
        try:
            chunk_bytes = _intent_chunk_bytes(intent)
        except (TypeError, ValueError) as exc:
            return DaemonResponse(ok=False, error=str(exc))
        with self._lock:
            existing_transfer_id = self._intent_transfers.get(intent.intent_id)
            if existing_transfer_id is not None:
                existing = self._transfer_intents.get(intent.intent_id)
                if existing != intent:
                    return DaemonResponse(
                        ok=False,
                        error="intent_id already belongs to a different transfer intent",
                    )
                try:
                    self._validate_transfer_buffers_locked(
                        (intent.source_buffer_id, intent.destination_buffer_id),
                        job_id=intent.job_id,
                        session_id=intent.session_id,
                        peer_identity=peer_identity,
                    )
                except ValueError as exc:
                    return DaemonResponse(ok=False, error=str(exc))
                try:
                    receipt = self._receipt_for_intent_locked(intent.intent_id)
                except ValueError as exc:
                    return DaemonResponse(ok=False, error=str(exc))
                ticket_id = self._transfer_tickets.get(existing_transfer_id)
                return DaemonResponse(
                    ok=True,
                    payload={
                        "receipt": asdict(receipt),
                        "transfer_id": existing_transfer_id,
                        "ticket": (
                            None
                            if ticket_id is None
                            else asdict(self._execution_tickets[ticket_id])
                        ),
                    },
                )

        planned = self.plan_transfer(
            session_id=intent.session_id,
            total_bytes=intent.total_bytes,
            chunk_bytes=chunk_bytes,
            mode="auto",
            direction=intent.direction,
            job_id=intent.job_id,
            buffer_ids=[intent.source_buffer_id, intent.destination_buffer_id],
            ranges=intent.ranges,
            intent_id=intent.intent_id,
            workload_kind=intent.workload_kind.value,
            priority=intent.priority,
            peer_identity=peer_identity,
        )
        if not planned.ok:
            return planned

        transfer_id = str(planned.payload["transfer_id"])
        now = time.time()
        with self._lock:
            decision = self._scheduling_decisions.get(transfer_id)
            if decision is None:
                return DaemonResponse(ok=False, error="scheduling decision is unavailable")
            ticket = self._execution_ticket_for_intent_locked(
                intent=intent,
                transfer_id=transfer_id,
                decision=decision,
                now=now,
            )
            self._transfer_intents[intent.intent_id] = intent
            self._intent_transfers[intent.intent_id] = transfer_id
            self._execution_tickets[ticket.ticket_id] = ticket
            self._transfer_tickets[transfer_id] = ticket.ticket_id
            self._refresh_transfer_queue_record_locked(transfer_id, now=now)
            receipt = self._receipt_for_intent_locked(intent.intent_id)
            return DaemonResponse(
                ok=True,
                payload={
                    "receipt": asdict(receipt),
                    "transfer_id": transfer_id,
                    "decision": asdict(decision),
                    "ticket": asdict(ticket),
                    "planning": planned.payload.get("planning", {}),
                },
            )

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> DaemonResponse:
        normalized_intent_id = str(intent_id)
        timeout = 0.0 if timeout_seconds is None else max(0.0, float(timeout_seconds))
        deadline = time.time() + timeout
        while True:
            with self._lock:
                try:
                    receipt = self._receipt_for_intent_locked(normalized_intent_id)
                except ValueError as exc:
                    return DaemonResponse(ok=False, error=str(exc))
                if receipt.state in _TERMINAL_TRANSFER_STATES or timeout_seconds is None:
                    return DaemonResponse(ok=True, payload={"receipt": asdict(receipt)})
                if time.time() >= deadline:
                    return DaemonResponse(ok=True, payload={"receipt": asdict(receipt)})
            time.sleep(min(0.01, max(0.0, deadline - time.time())))

    def validate_lease(
        self,
        lease_id: str,
        token: str,
        session_id: str | None = None,
        relay_gpu: int | None = None,
        job_id: str | None = None,
        buffer_ids: Iterable[str] | None = None,
        now: float | None = None,
        peer_identity: PeerIdentity | None = None,
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
            owner_job_id = lease.job_id if lease.job_id is not None else job_id
            if owner_job_id is not None:
                try:
                    self._validate_peer_owns_job_locked(
                        job_id=str(owner_job_id),
                        peer_identity=peer_identity,
                    )
                except ValueError as exc:
                    return DaemonResponse(ok=False, error=str(exc))
            if buffer_ids is not None:
                requested_buffers = tuple(str(buffer_id) for buffer_id in buffer_ids)
                if requested_buffers != lease.buffer_ids:
                    return DaemonResponse(ok=False, error="lease buffer mismatch")
                for buffer_id in requested_buffers:
                    buffer = self._buffers.get(buffer_id)
                    if buffer is None:
                        return DaemonResponse(ok=False, error="unknown buffer")
                    if lease.job_id is not None and buffer.job_id != lease.job_id:
                        return DaemonResponse(ok=False, error="lease buffer owner mismatch")
                    try:
                        self._validate_peer_owns_buffer_locked(
                            buffer_id=buffer_id,
                            peer_identity=peer_identity,
                        )
                    except ValueError as exc:
                        return DaemonResponse(ok=False, error=str(exc))
            if lease.expires_at and checked_at > lease.expires_at:
                self._release_expired_lease_locked(lease.lease_id)
                return DaemonResponse(ok=False, error="lease expired")
            if lease.lease_id not in self._reservations:
                return DaemonResponse(ok=False, error="lease is not active")
            transfer_id = self._reservation_transfers.get(lease.lease_id)
            if transfer_id is not None:
                status = self._transfer_statuses.get(transfer_id)
                if status is not None and status.state in _TERMINAL_TRANSFER_STATES:
                    return DaemonResponse(ok=False, error="transfer is terminal")
            return DaemonResponse(ok=True, payload={"lease_token": asdict(lease)})

    def authorize_worker_transfer(
        self,
        request: WorkerTransferAuthorizationRequest,
        peer_identity: PeerIdentity | None = None,
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
            try:
                self._validate_peer_owns_job_locked(
                    job_id=request.job_id,
                    peer_identity=peer_identity,
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            if status.state in _TERMINAL_TRANSFER_STATES:
                return DaemonResponse(ok=False, error="transfer is terminal")
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
            if required_buffers != lease.buffer_ids:
                return DaemonResponse(ok=False, error="lease buffer mismatch")
            src_buffer = self._buffers.get(request.src_buffer_id)
            dst_buffer = self._buffers.get(request.dst_buffer_id)
            if src_buffer is None or dst_buffer is None:
                return DaemonResponse(ok=False, error="unknown buffer")
            try:
                self._validate_peer_owns_buffer_locked(
                    buffer_id=src_buffer.buffer_id,
                    peer_identity=peer_identity,
                )
                self._validate_peer_owns_buffer_locked(
                    buffer_id=dst_buffer.buffer_id,
                    peer_identity=peer_identity,
                )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
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
            staging_record = self._register_staging_record_locked(
                lease=lease,
                transfer_id=request.transfer_id,
                direction=request.direction,
                ranges=authorized_ranges,
                requested_bytes=requested_bytes,
                now=now,
            )
            ticket = self._execution_ticket_for_worker_locked(
                authorization,
                lease=lease,
                transfer_id=request.transfer_id,
                now=now,
            )
            self._execution_tickets[ticket.ticket_id] = ticket
            self._transfer_tickets[request.transfer_id] = ticket.ticket_id
            self._append_audit_record_locked(
                event_type="relay_authorized",
                transfer_id=request.transfer_id,
                reservation=reservation,
                lease=lease,
                staging_record=staging_record,
                ticket=ticket,
                state=status.state,
                reason="worker_authorized",
                bytes_completed=status.bytes_completed,
                now=now,
            )
            decision = self._scheduling_decisions.get(request.transfer_id)
            return DaemonResponse(
                ok=True,
                payload={
                    "authorization": asdict(authorization),
                    "ticket": asdict(ticket),
                    "decision": None if decision is None else asdict(decision),
                    "src_buffer": asdict(src_buffer),
                    "dst_buffer": asdict(dst_buffer),
                    "relay_gpu": lease.relay_gpu,
                    "lease_id": request.lease_id,
                    "transfer_id": request.transfer_id,
                    "staging_record": dict(staging_record),
                },
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
        intent_id: str | None = None,
        topology_snapshot_id: str | None = None,
        workload_kind: str = "generic",
        priority: int = 0,
        peer_identity: PeerIdentity | None = None,
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
            try:
                buffer_ids_tuple, owner_job_id = self._validate_transfer_buffers_locked(
                    buffer_ids,
                    job_id=job_id,
                    session_id=session.session_id,
                    peer_identity=peer_identity,
                )
                if buffer_ids_tuple == () and job_id is not None:
                    self._validate_peer_owns_job_locked(
                        job_id=str(job_id),
                        peer_identity=peer_identity,
                    )
            except ValueError as exc:
                return DaemonResponse(ok=False, error=str(exc))
            if self._topology_provider is None:
                return _topology_unavailable_response()
            snapshot_id = topology_snapshot_id or self._topology_snapshot_id_locked()
            plan_job_id = owner_job_id if owner_job_id is not None else job_id
            intent = (
                None
                if intent_id is None
                else self._transfer_intents.get(str(intent_id))
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
                runtime_state=self._runtime_resource_state_locked(now=now),
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                ranges=normalized_ranges,
                mode=mode,
                direction=direction,
                workload_kind=(
                    workload_kind if intent is None else intent.workload_kind
                ),
                priority=priority if intent is None else intent.priority,
                now=now,
                job_id=plan_job_id,
                intent_id=intent_id,
                topology_snapshot_id=snapshot_id,
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
                job_id=str(plan_job_id or session.session_id),
                state=TransferStatusState.SUBMITTED,
                bytes_total=int(total_bytes),
                bytes_completed=0,
                session_id=session.session_id,
            )
            self._transfer_statuses[transfer_id] = status
            self._transfer_plans[transfer_id] = dict(decision.plan)
            self._scheduling_decisions[transfer_id] = decision
            self._record_planned_transfer_locked(
                transfer_id=transfer_id,
                status=status,
                intent_id=intent_id,
                buffer_ids=buffer_ids_tuple,
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                ranges=normalized_ranges,
                direction=direction,
                decision=decision,
                now=now,
            )
            self._touch_session_locked(session.session_id, now)
            payload = {
                "decision": asdict(decision),
                "decision_id": decision.decision_id,
                "topology_snapshot_id": decision.topology_snapshot_id,
                "plan": dict(decision.plan),
                "path_summary": list(decision.path_summary),
                "stats": scheduling_decision_stats(decision).as_dict(),
                "leases": [
                    lease.as_dict() for lease in scheduling_decision_leases(decision)
                ],
            }
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
        reservation_key = str(reservation_id)
        reservation = self._reservations.get(reservation_key)
        if reservation is None:
            return None
        transfer_id = self._reservation_transfers.get(reservation_key)
        lease = self._lease_tokens.get(reservation_key)
        staging_record = self._staging_records.get(reservation_key)
        if cleanup_reason is not None:
            self._append_audit_record_locked(
                event_type="cleanup",
                transfer_id=transfer_id,
                reservation=reservation,
                lease=lease,
                staging_record=staging_record,
                state=final_state,
                reason=cleanup_reason,
                failure_reason=(
                    cleanup_reason
                    if final_state in {TransferStatusState.FAILED, TransferStatusState.CANCELED}
                    else None
                ),
                cleanup_kind="reservation",
                cleanup_target_id=reservation_key,
            )
        self._reservations.pop(reservation_key, None)
        self._lease_tokens.pop(reservation_key, None)
        self._staging_records.pop(reservation_key, None)
        transfer_id = self._reservation_transfers.pop(reservation_key, None)
        session = self._sessions.get(reservation.session_id)
        if session is not None:
            session.active_chunks = max(0, session.active_chunks - reservation.chunks)
        quota = self._relay_quotas.get(reservation.relay_gpu)
        if quota is not None:
            quota.active_chunks = max(0, quota.active_chunks - reservation.chunks)
        if transfer_id is not None:
            self._mark_transfer_terminal_if_unblocked_locked(transfer_id, final_state)
        if cleanup_reason is not None:
            self._system_cleanup_events.append(
                CleanupRequest(
                    target_kind="reservation",
                    target_id=reservation_id,
                    reason=cleanup_reason,
                    force=True,
                )
            )
        return reservation

    def _release_reservation_and_count_locked(
        self,
        reservation_id: str,
        final_state: TransferStatusState,
        cleanup_reason: str | None = None,
    ) -> dict[str, int]:
        removed = _empty_removed_summary()
        transfer_id = self._reservation_transfers.get(str(reservation_id))
        status_before = (
            None if transfer_id is None else self._transfer_statuses.get(transfer_id)
        )
        staging_record = self._staging_records.get(str(reservation_id))
        reservation = self._release_reservation_locked(
            str(reservation_id),
            final_state=final_state,
            cleanup_reason=cleanup_reason,
        )
        if reservation is not None:
            removed["reservations"] += 1
        if staging_record is not None:
            removed["staging_records"] += 1
        if status_before is not None and status_before.state not in _TERMINAL_TRANSFER_STATES:
            status_after = self._transfer_statuses.get(status_before.transfer_id)
            if status_after is not None and status_after.state in _TERMINAL_TRANSFER_STATES:
                removed["transfers"] += 1
        return removed

    def _release_reservations_for_transfer_locked(
        self,
        transfer_id: str,
        final_state: TransferStatusState,
        cleanup_reason: str | None = None,
    ) -> dict[str, int]:
        removed = _empty_removed_summary()
        for reservation_id, mapped_transfer_id in list(self._reservation_transfers.items()):
            if mapped_transfer_id != str(transfer_id):
                continue
            _merge_removed(
                removed,
                self._release_reservation_and_count_locked(
                    reservation_id,
                    final_state=final_state,
                    cleanup_reason=cleanup_reason,
                ),
            )
        return removed

    def _commit_scheduler_leases_locked(
        self,
        session: Session,
        decision: SchedulingDecision,
        transfer_id: str | None = None,
        buffer_ids: tuple[str, ...] = (),
    ) -> list[TransferReservation]:
        reservations: list[TransferReservation] = []
        for lease in scheduling_decision_leases(decision):
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

    def _record_planned_transfer_locked(
        self,
        *,
        transfer_id: str,
        status: TransferStatus,
        intent_id: str | None,
        buffer_ids: tuple[str, ...],
        total_bytes: int,
        chunk_bytes: int,
        ranges: tuple[dict[str, int], ...] | None,
        direction: str,
        decision: SchedulingDecision,
        now: float,
    ) -> None:
        self._transfer_queue.append(str(transfer_id))
        self._transfer_queue_records[str(transfer_id)] = {
            "transfer_id": str(transfer_id),
            "intent_id": None if intent_id is None else str(intent_id),
            "decision_id": decision.decision_id,
            "topology_snapshot_id": decision.topology_snapshot_id,
            "job_id": status.job_id,
            "session_id": status.session_id,
            "state": status.state.value,
            "direction": str(direction).lower(),
            "bytes_total": int(total_bytes),
            "bytes_completed": status.bytes_completed,
            "chunk_bytes": int(chunk_bytes),
            "ranges": tuple(dict(item) for item in ranges) if ranges is not None else (),
            "source_buffer_id": buffer_ids[0] if len(buffer_ids) >= 1 else None,
            "destination_buffer_id": buffer_ids[1] if len(buffer_ids) >= 2 else None,
            "buffer_ids": buffer_ids,
            "workload_kind": None,
            "priority": 0,
            "queued_at": float(now),
            "planned_at": decision.issued_at,
            "started_at": None,
            "completed_at": None,
            "fallback_reason": decision.fallback_reason,
        }
        self._refresh_transfer_queue_record_locked(str(transfer_id), now=now)
        self._runtime_state_version += 1

    def _refresh_transfer_queue_record_locked(
        self,
        transfer_id: str,
        *,
        now: float | None = None,
    ) -> dict[str, object] | None:
        record = self._transfer_queue_records.get(str(transfer_id))
        status = self._transfer_statuses.get(str(transfer_id))
        if record is None or status is None:
            return record
        previous_signature = (
            str(record.get("state", "")),
            int(record.get("bytes_completed", 0) or 0),
            record.get("error"),
            record.get("intent_id"),
            record.get("source_buffer_id"),
            record.get("destination_buffer_id"),
            record.get("workload_kind"),
            int(record.get("priority", 0) or 0),
            record.get("started_at"),
            record.get("completed_at"),
        )
        state = status.state.value
        record["state"] = state
        record["bytes_completed"] = status.bytes_completed
        if status.error is not None:
            record["error"] = status.error
        if status.state is TransferStatusState.RUNNING and record.get("started_at") is None:
            record["started_at"] = float(time.time() if now is None else now)
        if status.state in _TERMINAL_TRANSFER_STATES and record.get("completed_at") is None:
            record["completed_at"] = float(time.time() if now is None else now)
        intent = None
        intent_id = record.get("intent_id")
        if intent_id is not None:
            intent = self._transfer_intents.get(str(intent_id))
        if intent is not None:
            record["intent_id"] = intent.intent_id
            record["source_buffer_id"] = intent.source_buffer_id
            record["destination_buffer_id"] = intent.destination_buffer_id
            record["buffer_ids"] = (intent.source_buffer_id, intent.destination_buffer_id)
            record["workload_kind"] = intent.workload_kind.value
            record["priority"] = intent.priority
        updated_signature = (
            str(record.get("state", "")),
            int(record.get("bytes_completed", 0) or 0),
            record.get("error"),
            record.get("intent_id"),
            record.get("source_buffer_id"),
            record.get("destination_buffer_id"),
            record.get("workload_kind"),
            int(record.get("priority", 0) or 0),
            record.get("started_at"),
            record.get("completed_at"),
        )
        if previous_signature != updated_signature:
            self._runtime_state_version += 1
        return record

    def _runtime_resource_state_locked(
        self,
        *,
        now: float | None = None,
    ) -> dict[str, object]:
        captured_at = float(time.time() if now is None else now)
        for transfer_id in tuple(self._transfer_queue):
            self._refresh_transfer_queue_record_locked(transfer_id, now=captured_at)
        transfer_records = [
            dict(self._transfer_queue_records[transfer_id])
            for transfer_id in self._transfer_queue
            if transfer_id in self._transfer_queue_records
        ]
        queued_transfers = [
            dict(record)
            for record in transfer_records
            if str(record.get("state")) == TransferStatusState.SUBMITTED.value
        ]
        running_transfers = [
            dict(record)
            for record in transfer_records
            if str(record.get("state")) == TransferStatusState.RUNNING.value
        ]
        active_transfers = [
            dict(record)
            for record in transfer_records
            if str(record.get("state"))
            in {
                TransferStatusState.SUBMITTED.value,
                TransferStatusState.RUNNING.value,
            }
        ]
        active_by_direction: dict[str, dict[str, int]] = {}
        queued_by_direction: dict[str, dict[str, int]] = {}
        for record in active_transfers:
            direction = str(record.get("direction", "unknown"))
            bucket = active_by_direction.setdefault(
                direction,
                {"transfer_count": 0, "bytes_total": 0, "bytes_remaining": 0},
            )
            bucket["transfer_count"] += 1
            bucket["bytes_total"] += int(record.get("bytes_total", 0) or 0)
            bucket["bytes_remaining"] += max(
                0,
                int(record.get("bytes_total", 0) or 0)
                - int(record.get("bytes_completed", 0) or 0),
            )
        for record in queued_transfers:
            direction = str(record.get("direction", "unknown"))
            bucket = queued_by_direction.setdefault(
                direction,
                {"transfer_count": 0, "bytes_total": 0},
            )
            bucket["transfer_count"] += 1
            bucket["bytes_total"] += int(record.get("bytes_total", 0) or 0)
        path_records, path_summary = self._active_path_records_locked(active_transfers)
        active_reservations = [
            self._runtime_reservation_record_locked(reservation_id, reservation)
            for reservation_id, reservation in sorted(self._reservations.items())
        ]
        active_leases = [
            self._runtime_lease_record_locked(lease_id, lease)
            for lease_id, lease in sorted(self._lease_tokens.items())
            if lease_id in self._reservations
        ]
        staging_records = [dict(value) for _, value in sorted(self._staging_records.items())]
        job_runtime_state = self._job_runtime_state_locked(transfer_records)
        relay_path_summary = {
            "path_count": 0,
            "chunk_count": 0,
            "bytes_total": 0,
        }
        for key, value in path_summary.items():
            if not key.endswith(":relay"):
                continue
            relay_path_summary["path_count"] += int(value.get("path_count", 0) or 0)
            relay_path_summary["chunk_count"] += int(value.get("chunk_count", 0) or 0)
            relay_path_summary["bytes_total"] += int(value.get("bytes_total", 0) or 0)
        active_resource_usage = {
            "h2d": dict(active_by_direction.get("h2d", {})),
            "d2h": dict(active_by_direction.get("d2h", {})),
            "p2p": dict(relay_path_summary),
            "relay_staging": {
                "count": len(staging_records),
                "active_reservation_count": len(active_reservations),
                "active_lease_count": len(active_leases),
            },
        }
        return {
            "version": self._runtime_state_version,
            "captured_at": captured_at,
            "transfer_order": tuple(self._transfer_queue),
            "transfers": transfer_records,
            "queued_transfers": queued_transfers,
            "running_transfers": running_transfers,
            "active_transfers": active_transfers,
            "active_paths": path_records,
            "active_resource_usage": active_resource_usage,
            "job_runtime_state": job_runtime_state,
            "active_reservations": active_reservations,
            "active_leases": active_leases,
            "relay_staging": staging_records,
            "summary": {
                "queued_transfer_count": len(queued_transfers),
                "running_transfer_count": len(running_transfers),
                "active_transfer_count": len(active_transfers),
                "terminal_transfer_count": sum(
                    1
                    for record in transfer_records
                    if str(record.get("state"))
                    in {
                        TransferStatusState.COMPLETE.value,
                        TransferStatusState.FAILED.value,
                        TransferStatusState.CANCELED.value,
                    }
                ),
                "active_reservation_count": len(active_reservations),
                "active_lease_count": len(active_leases),
                "relay_staging_count": len(staging_records),
                "relay_path_count": relay_path_summary["path_count"],
                "relay_path_bytes_total": relay_path_summary["bytes_total"],
                "queued_bytes_by_direction": queued_by_direction,
                "active_bytes_by_direction": active_by_direction,
                "active_paths": path_summary,
                "active_resource_usage": active_resource_usage,
                "job_runtime_state": job_runtime_state,
            },
        }

    def _runtime_reservation_record_locked(
        self,
        reservation_id: str,
        reservation: TransferReservation,
    ) -> dict[str, object]:
        record = asdict(reservation)
        record["transfer_id"] = self._reservation_transfers.get(str(reservation_id))
        lease = self._lease_tokens.get(str(reservation_id))
        record["job_id"] = None if lease is None else lease.job_id
        record["buffer_ids"] = () if lease is None else lease.buffer_ids
        return record

    def _runtime_lease_record_locked(
        self,
        lease_id: str,
        lease: LeaseToken,
    ) -> dict[str, object]:
        return {
            "lease_id": lease.lease_id,
            "session_id": lease.session_id,
            "relay_gpu": lease.relay_gpu,
            "job_id": lease.job_id,
            "buffer_ids": lease.buffer_ids,
            "issued_at": lease.issued_at,
            "expires_at": lease.expires_at,
            "transfer_id": self._reservation_transfers.get(str(lease_id)),
        }

    def _job_runtime_state_locked(
        self,
        transfer_records: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        jobs = {
            job_id: {
                "job_id": job_id,
                "weight": float(job.weight),
                "queued_transfer_count": 0,
                "running_transfer_count": 0,
                "active_transfer_count": 0,
                "active_bytes_total": 0,
                "active_bytes_remaining": 0,
            }
            for job_id, job in self._jobs.items()
        }
        for record in transfer_records:
            job_id = record.get("job_id")
            if job_id is None:
                continue
            normalized = str(job_id)
            job_record = jobs.setdefault(
                normalized,
                {
                    "job_id": normalized,
                    "weight": 1.0,
                    "queued_transfer_count": 0,
                    "running_transfer_count": 0,
                    "active_transfer_count": 0,
                    "active_bytes_total": 0,
                    "active_bytes_remaining": 0,
                },
            )
            state = str(record.get("state", ""))
            if state == TransferStatusState.SUBMITTED.value:
                job_record["queued_transfer_count"] = int(
                    job_record["queued_transfer_count"]
                ) + 1
            elif state == TransferStatusState.RUNNING.value:
                job_record["running_transfer_count"] = int(
                    job_record["running_transfer_count"]
                ) + 1
            if state in {
                TransferStatusState.SUBMITTED.value,
                TransferStatusState.RUNNING.value,
            }:
                bytes_total = int(record.get("bytes_total", 0) or 0)
                bytes_completed = int(record.get("bytes_completed", 0) or 0)
                job_record["active_transfer_count"] = int(
                    job_record["active_transfer_count"]
                ) + 1
                job_record["active_bytes_total"] = int(
                    job_record["active_bytes_total"]
                ) + bytes_total
                job_record["active_bytes_remaining"] = int(
                    job_record["active_bytes_remaining"]
                ) + max(0, bytes_total - bytes_completed)
        return dict(sorted(jobs.items()))

    def _active_path_records_locked(
        self,
        active_transfers: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
        transfer_ids = {str(record["transfer_id"]) for record in active_transfers}
        records: list[dict[str, object]] = []
        summary: dict[str, dict[str, int]] = {}
        for transfer_id in sorted(transfer_ids):
            decision = self._scheduling_decisions.get(transfer_id)
            if decision is None:
                continue
            for assignment in decision.plan.get("assignments", ()) or ():
                if not isinstance(assignment, Mapping):
                    continue
                path = assignment.get("path")
                if not isinstance(path, Mapping):
                    continue
                chunks = assignment.get("chunks", ()) or ()
                chunk_count = len(chunks) if isinstance(chunks, list | tuple) else 0
                bytes_total = int(assignment.get("bytes", 0) or 0)
                if not bytes_total and isinstance(chunks, list | tuple):
                    bytes_total = sum(
                        int(chunk.get("bytes", 0) or 0)
                        for chunk in chunks
                        if isinstance(chunk, Mapping)
                    )
                kind = str(path.get("kind", "unknown"))
                direction = str(path.get("direction", "unknown"))
                key = f"{direction}:{kind}"
                bucket = summary.setdefault(
                    key,
                    {"path_count": 0, "chunk_count": 0, "bytes_total": 0},
                )
                bucket["path_count"] += 1
                bucket["chunk_count"] += chunk_count
                bucket["bytes_total"] += bytes_total
                records.append(
                    {
                        "transfer_id": transfer_id,
                        "kind": kind,
                        "direction": direction,
                        "target_device": path.get("target_device"),
                        "relay_device": path.get("relay_device"),
                        "bytes_total": bytes_total,
                        "chunk_count": chunk_count,
                    }
                )
        return records, summary

    def _execution_ticket_for_worker_locked(
        self,
        authorization: WorkerTransferAuthorization,
        *,
        lease: LeaseToken,
        transfer_id: str,
        now: float,
    ) -> ExecutionTicket:
        decision = self._scheduling_decisions.get(str(transfer_id))
        if decision is None:
            raise ValueError("scheduling decision is unavailable")
        expires_at = float(lease.expires_at or (float(now) + 30.0))
        if expires_at <= float(now):
            raise ValueError("lease expired")
        ticket_ranges = _ticket_ranges_for_plan(
            decision.plan,
            direction=authorization.direction,
        )
        return ExecutionTicket(
            ticket_id=f"ticket-{transfer_id}",
            decision_id=decision.decision_id,
            intent_id=decision.intent_id,
            topology_snapshot_id=decision.topology_snapshot_id,
            job_id=authorization.job_id,
            session_id=authorization.session_id,
            source_buffer_id=authorization.src_buffer.buffer_id,
            destination_buffer_id=authorization.dst_buffer.buffer_id,
            direction=authorization.direction,
            total_bytes=sum(item["bytes"] for item in ticket_ranges),
            ranges=ticket_ranges,
            plan=dict(decision.plan),
            issued_at=float(now),
            expires_at=expires_at,
            lease_ids=(authorization.lease_id,),
            metadata={"transfer_id": transfer_id},
        )

    def _execution_ticket_for_intent_locked(
        self,
        *,
        intent: TransferIntent,
        transfer_id: str,
        decision: SchedulingDecision,
        now: float,
    ) -> ExecutionTicket:
        ticket_ranges = _ticket_ranges_for_plan(decision.plan, direction=intent.direction)
        lease_ids = tuple(
            reservation_id
            for reservation_id, mapped_transfer_id in sorted(
                self._reservation_transfers.items()
            )
            if mapped_transfer_id == transfer_id
        )
        expires_at = max(
            [float(now) + 30.0]
            + [
                float(self._lease_tokens[lease_id].expires_at)
                for lease_id in lease_ids
                if lease_id in self._lease_tokens
                and float(self._lease_tokens[lease_id].expires_at) > float(now)
            ]
        )
        return ExecutionTicket(
            ticket_id=f"ticket-{transfer_id}",
            decision_id=decision.decision_id,
            intent_id=intent.intent_id,
            topology_snapshot_id=decision.topology_snapshot_id,
            job_id=intent.job_id,
            session_id=intent.session_id,
            source_buffer_id=intent.source_buffer_id,
            destination_buffer_id=intent.destination_buffer_id,
            direction=intent.direction,
            total_bytes=sum(item["bytes"] for item in ticket_ranges),
            ranges=ticket_ranges,
            plan=dict(decision.plan),
            issued_at=float(now),
            expires_at=expires_at,
            lease_ids=lease_ids,
            metadata={"transfer_id": transfer_id},
        )

    def _receipt_for_intent_locked(self, intent_id: str) -> TransferReceipt:
        normalized_intent_id = str(intent_id)
        transfer_id = self._intent_transfers.get(normalized_intent_id)
        if transfer_id is None:
            raise ValueError("unknown transfer intent")
        intent = self._transfer_intents.get(normalized_intent_id)
        status = self._transfer_statuses.get(transfer_id)
        decision = self._scheduling_decisions.get(transfer_id)
        ticket_id = self._transfer_tickets.get(transfer_id)
        if intent is None:
            raise ValueError("transfer intent is unavailable")
        if status is None:
            raise ValueError("transfer status is unavailable")
        if decision is None:
            raise ValueError("scheduling decision is unavailable")
        if ticket_id is None or ticket_id not in self._execution_tickets:
            raise ValueError("execution ticket is unavailable")
        error = status.error
        if status.state in {TransferStatusState.FAILED, TransferStatusState.CANCELED}:
            error = error or f"transfer {status.state.value}"
        return TransferReceipt(
            receipt_id=f"receipt-{transfer_id}",
            ticket_id=ticket_id,
            intent_id=intent.intent_id,
            decision_id=decision.decision_id,
            topology_snapshot_id=decision.topology_snapshot_id,
            job_id=intent.job_id,
            session_id=intent.session_id,
            state=status.state,
            bytes_total=status.bytes_total,
            bytes_completed=status.bytes_completed,
            started_at=decision.issued_at,
            path_stats=decision.path_summary,
            error=error,
            metadata={
                "transfer_id": transfer_id,
                "fallback_reason": decision.fallback_reason,
            },
        )

    def _topology_snapshot_id_locked(self) -> str:
        if self._topology_provider is None:
            return "topology-unavailable"
        inventory = self._topology_provider.snapshot()
        return inventory.topology_snapshot_id()

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

    def _active_buffer_lease_ids_locked(self, buffer_id: str) -> tuple[str, ...]:
        normalized = str(buffer_id)
        return tuple(
            lease_id
            for lease_id, lease in sorted(self._lease_tokens.items())
            if lease_id in self._reservations and normalized in lease.buffer_ids
        )

    def _register_staging_record_locked(
        self,
        *,
        lease: LeaseToken,
        transfer_id: str,
        direction: str,
        ranges: tuple[dict[str, int], ...],
        requested_bytes: int,
        now: float,
    ) -> dict[str, object]:
        record = {
            "staging_record_id": f"staging-{lease.lease_id}",
            "lease_id": lease.lease_id,
            "transfer_id": str(transfer_id),
            "session_id": lease.session_id,
            "job_id": lease.job_id,
            "relay_gpu": lease.relay_gpu,
            "buffer_ids": lease.buffer_ids,
            "direction": str(direction).lower(),
            "ranges": tuple(dict(item) for item in ranges),
            "requested_bytes": int(requested_bytes),
            "state": "authorized",
            "created_at": float(now),
        }
        self._staging_records[lease.lease_id] = record
        return record

    def _append_transfer_audit_records_locked(
        self,
        *,
        event_type: str,
        transfer_id: str,
        state: TransferStatusState | str,
        reason: str | None = None,
        failure_reason: str | None = None,
        bytes_completed: int | None = None,
    ) -> None:
        reservations = [
            self._reservations[reservation_id]
            for reservation_id, mapped_transfer_id in sorted(
                self._reservation_transfers.items()
            )
            if mapped_transfer_id == str(transfer_id)
            and reservation_id in self._reservations
        ]
        if reservations:
            for reservation in reservations:
                lease = self._lease_tokens.get(reservation.reservation_id)
                self._append_audit_record_locked(
                    event_type=event_type,
                    transfer_id=str(transfer_id),
                    reservation=reservation,
                    lease=lease,
                    staging_record=self._staging_records.get(reservation.reservation_id),
                    state=state,
                    reason=reason,
                    failure_reason=failure_reason,
                    bytes_completed=bytes_completed,
                )
            return
        self._append_audit_record_locked(
            event_type=event_type,
            transfer_id=str(transfer_id),
            state=state,
            reason=reason,
            failure_reason=failure_reason,
            bytes_completed=bytes_completed,
        )

    def _append_audit_record_locked(
        self,
        *,
        event_type: str,
        transfer_id: str | None = None,
        reservation: TransferReservation | None = None,
        lease: LeaseToken | None = None,
        staging_record: dict[str, object] | None = None,
        ticket: ExecutionTicket | None = None,
        state: TransferStatusState | str | None = None,
        reason: str | None = None,
        failure_reason: str | None = None,
        cleanup_kind: str | None = None,
        cleanup_target_id: str | None = None,
        session_id: str | None = None,
        bytes_completed: int | None = None,
        now: float | None = None,
    ) -> dict[str, object]:
        created_at = float(time.time() if now is None else now)
        normalized_transfer_id = None if transfer_id is None else str(transfer_id)
        if normalized_transfer_id is None and staging_record is not None:
            value = staging_record.get("transfer_id")
            normalized_transfer_id = None if value is None else str(value)
        status = (
            None
            if normalized_transfer_id is None
            else self._transfer_statuses.get(normalized_transfer_id)
        )
        decision = (
            None
            if normalized_transfer_id is None
            else self._scheduling_decisions.get(normalized_transfer_id)
        )
        ticket_id = None
        if ticket is not None:
            ticket_id = ticket.ticket_id
        elif normalized_transfer_id is not None:
            ticket_id = self._transfer_tickets.get(normalized_transfer_id)
        active_ticket = None if ticket_id is None else self._execution_tickets.get(ticket_id)
        if ticket is None:
            ticket = active_ticket
        lease_id = None
        if lease is not None:
            lease_id = lease.lease_id
        elif reservation is not None:
            lease_id = reservation.reservation_id
        elif staging_record is not None:
            value = staging_record.get("lease_id")
            lease_id = None if value is None else str(value)
        if lease is None and lease_id is not None:
            lease = self._lease_tokens.get(lease_id)
        resolved_session_id = session_id
        if resolved_session_id is None and status is not None:
            resolved_session_id = status.session_id
        if resolved_session_id is None and lease is not None:
            resolved_session_id = lease.session_id
        if resolved_session_id is None and reservation is not None:
            resolved_session_id = reservation.session_id
        if resolved_session_id is None and staging_record is not None:
            value = staging_record.get("session_id")
            resolved_session_id = None if value is None else str(value)
        job_id = None
        if status is not None:
            job_id = status.job_id
        elif lease is not None:
            job_id = lease.job_id
        elif staging_record is not None:
            value = staging_record.get("job_id")
            job_id = None if value is None else str(value)
        elif decision is not None:
            job_id = decision.job_id
        job = None if job_id is None else self._jobs.get(job_id)
        buffer_ids: tuple[str, ...] = ()
        if lease is not None:
            buffer_ids = tuple(lease.buffer_ids)
        elif staging_record is not None:
            buffer_ids = tuple(str(item) for item in staging_record.get("buffer_ids", ()))
        elif ticket is not None:
            buffer_ids = (ticket.source_buffer_id, ticket.destination_buffer_id)
        relay_gpu = None
        if reservation is not None:
            relay_gpu = reservation.relay_gpu
        elif lease is not None:
            relay_gpu = lease.relay_gpu
        elif staging_record is not None and staging_record.get("relay_gpu") is not None:
            relay_gpu = int(staging_record["relay_gpu"])
        direction = None
        if reservation is not None:
            direction = reservation.direction
        elif staging_record is not None:
            value = staging_record.get("direction")
            direction = None if value is None else str(value)
        elif ticket is not None:
            direction = ticket.direction
        bytes_total = 0
        if reservation is not None:
            bytes_total = int(reservation.bytes)
        elif staging_record is not None:
            bytes_total = int(staging_record.get("requested_bytes", 0) or 0)
        elif status is not None:
            bytes_total = int(status.bytes_total)
        completed = (
            int(bytes_completed)
            if bytes_completed is not None
            else (int(status.bytes_completed) if status is not None else 0)
        )
        if reservation is not None and bytes_total:
            completed = min(completed, bytes_total)
        started_at = None
        if staging_record is not None:
            started_at = float(staging_record.get("created_at", 0.0) or 0.0)
        elif decision is not None:
            started_at = float(decision.issued_at)
        duration_seconds = None
        if started_at:
            duration_seconds = max(0.0, created_at - started_at)
        record = {
            "audit_id": f"audit-{len(self._audit_records) + 1}",
            "event_type": str(event_type),
            "created_at": created_at,
            "transfer_id": normalized_transfer_id,
            "decision_id": None if decision is None else decision.decision_id,
            "ticket_id": ticket_id,
            "topology_snapshot_id": (
                None if decision is None else decision.topology_snapshot_id
            ),
            "lease_id": lease_id,
            "session_id": None if resolved_session_id is None else str(resolved_session_id),
            "job_id": job_id,
            "user_id": None if job is None else job.user_id,
            "process_id": None if job is None else job.process_id,
            "container_id": None if job is None else job.container_id,
            "relay_gpu": relay_gpu,
            "direction": direction,
            "bytes_total": bytes_total,
            "bytes_completed": completed,
            "duration_seconds": duration_seconds,
            "state": (
                state.value
                if isinstance(state, TransferStatusState)
                else (None if state is None else str(state))
            ),
            "reason": reason,
            "failure_reason": failure_reason,
            "cleanup_kind": cleanup_kind,
            "cleanup_target_id": cleanup_target_id,
            "source_buffer_id": None if ticket is None else ticket.source_buffer_id,
            "destination_buffer_id": (
                None if ticket is None else ticket.destination_buffer_id
            ),
            "buffer_ids": buffer_ids,
            "staging_record_id": (
                None
                if staging_record is None
                else staging_record.get("staging_record_id")
            ),
        }
        self._audit_records.append(record)
        return record

    def _cleanup_job_locked(self, job_id: str, reason: str) -> dict[str, int]:
        removed = _empty_removed_summary()
        job = self._jobs.pop(str(job_id), None)
        if job is not None:
            removed["jobs"] += 1
        self._job_peer_identities.pop(str(job_id), None)
        transfer_ids = self._transfer_ids_for_job_locked(str(job_id))
        for reservation_id, lease in list(self._lease_tokens.items()):
            if lease.job_id == str(job_id):
                _merge_removed(
                    removed,
                    self._release_reservation_and_count_locked(
                        reservation_id,
                        final_state=TransferStatusState.CANCELED,
                        cleanup_reason=reason,
                    ),
                )
        for transfer_id in transfer_ids:
            status = self._transfer_statuses.get(transfer_id)
            if status is None or status.state in _TERMINAL_TRANSFER_STATES:
                continue
            self._mark_transfer_terminal_locked(
                transfer_id,
                TransferStatusState.CANCELED,
                error=reason,
            )
            removed["transfers"] += 1
        for buffer_id, buffer in list(self._buffers.items()):
            if buffer.job_id == str(job_id):
                self._buffers.pop(buffer_id, None)
                removed["buffers"] += 1
        return removed

    def _transfer_ids_for_job_locked(self, job_id: str) -> tuple[str, ...]:
        transfer_ids = {
            transfer_id
            for transfer_id, status in self._transfer_statuses.items()
            if status.job_id == str(job_id)
        }
        for intent_id, intent in self._transfer_intents.items():
            if intent.job_id == str(job_id):
                transfer_id = self._intent_transfers.get(intent_id)
                if transfer_id is not None:
                    transfer_ids.add(transfer_id)
        for reservation_id, lease in self._lease_tokens.items():
            if lease.job_id == str(job_id):
                transfer_id = self._reservation_transfers.get(reservation_id)
                if transfer_id is not None:
                    transfer_ids.add(transfer_id)
        return tuple(sorted(transfer_ids))

    def _transfer_ids_for_session_locked(self, session_id: str) -> tuple[str, ...]:
        transfer_ids = {
            transfer_id
            for transfer_id, status in self._transfer_statuses.items()
            if status.session_id == str(session_id)
        }
        for intent_id, intent in self._transfer_intents.items():
            if intent.session_id == str(session_id):
                transfer_id = self._intent_transfers.get(intent_id)
                if transfer_id is not None:
                    transfer_ids.add(transfer_id)
        return tuple(sorted(transfer_ids))

    def _transfer_ids_for_buffer_locked(self, buffer_id: str) -> tuple[str, ...]:
        normalized = str(buffer_id)
        transfer_ids = set()
        for intent_id, intent in self._transfer_intents.items():
            if normalized in {intent.source_buffer_id, intent.destination_buffer_id}:
                transfer_id = self._intent_transfers.get(intent_id)
                if transfer_id is not None:
                    transfer_ids.add(transfer_id)
        for reservation_id, lease in self._lease_tokens.items():
            if normalized in lease.buffer_ids:
                transfer_id = self._reservation_transfers.get(reservation_id)
                if transfer_id is not None:
                    transfer_ids.add(transfer_id)
        return tuple(sorted(transfer_ids))

    def _validate_peer_owns_job_locked(
        self,
        *,
        job_id: str,
        peer_identity: PeerIdentity | None,
    ) -> None:
        if peer_identity is None or not peer_identity.authenticated:
            return
        job_key = str(job_id)
        job = self._jobs.get(job_key)
        if job is None:
            raise ValueError("unknown job")
        job_peer = self._job_peer_identities.get(job_key)
        if job_peer is None:
            raise ValueError("job owner identity is unavailable")
        _validate_peer_owner_match(
            expected=job_peer,
            actual=peer_identity,
            owner_name="job",
        )

    def _validate_peer_owns_buffer_locked(
        self,
        *,
        buffer_id: str,
        peer_identity: PeerIdentity | None,
    ) -> None:
        if peer_identity is None or not peer_identity.authenticated:
            return
        buffer = self._buffers.get(str(buffer_id))
        if buffer is None:
            raise ValueError("unknown buffer")
        try:
            self._validate_peer_owns_job_locked(
                job_id=buffer.job_id,
                peer_identity=peer_identity,
            )
        except ValueError as exc:
            if str(exc) == "job owner does not match authenticated peer":
                raise ValueError("buffer owner does not match authenticated peer") from exc
            raise

    def _validate_transfer_buffers_locked(
        self,
        buffer_ids: Iterable[str] | None,
        job_id: str | None,
        session_id: str,
        peer_identity: PeerIdentity | None = None,
    ) -> tuple[tuple[str, ...], str | None]:
        if buffer_ids is None:
            return (), None
        normalized = tuple(str(buffer_id) for buffer_id in buffer_ids)
        if not normalized:
            return (), None
        if any(not buffer_id.strip() for buffer_id in normalized):
            raise ValueError("buffer_ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("buffer_ids must be unique")
        owner_job_id = None if job_id is None else str(job_id)
        for buffer_id in normalized:
            buffer = self._buffers.get(buffer_id)
            if buffer is None:
                raise ValueError(f"unknown buffer: {buffer_id}")
            if owner_job_id is None:
                owner_job_id = buffer.job_id
            if buffer.job_id != str(owner_job_id):
                raise ValueError("buffer owner does not match job")
            self._validate_peer_owns_buffer_locked(
                buffer_id=buffer_id,
                peer_identity=peer_identity,
            )
        if owner_job_id is None:
            raise ValueError("job_id is required when buffer_ids are provided")
        job = self._jobs.get(str(owner_job_id))
        if job is None:
            raise ValueError("unknown job")
        if job.session_id != session_id:
            raise ValueError("job session does not match transfer session")
        self._validate_peer_owns_job_locked(
            job_id=str(owner_job_id),
            peer_identity=peer_identity,
        )
        return normalized, str(owner_job_id)

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
        self._refresh_transfer_queue_record_locked(transfer_id)

    def _mark_transfer_terminal_locked(
        self,
        transfer_id: str,
        final_state: TransferStatusState,
        error: str | None = None,
    ) -> TransferStatus:
        status = self._transfer_statuses.get(str(transfer_id))
        if status is None:
            raise ValueError("unknown transfer")
        if status.state in _TERMINAL_TRANSFER_STATES:
            return status
        completed = (
            status.bytes_total
            if final_state is TransferStatusState.COMPLETE
            else status.bytes_completed
        )
        terminal = TransferStatus(
            transfer_id=status.transfer_id,
            job_id=status.job_id,
            state=final_state,
            bytes_total=status.bytes_total,
            bytes_completed=completed,
            session_id=status.session_id,
            error=status.error if error is None else error,
        )
        self._transfer_statuses[terminal.transfer_id] = terminal
        self._refresh_transfer_queue_record_locked(terminal.transfer_id)
        return terminal

    def _close_session_locked(
        self,
        session_id: str,
        reason: str = "session_closed",
        removed: dict[str, object] | None = None,
    ) -> Session | None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.active = False
        session.closed_at = time.time()
        self._append_audit_record_locked(
            event_type="cleanup",
            session_id=session_id,
            state=TransferStatusState.CANCELED,
            reason=reason,
            failure_reason=reason,
            cleanup_kind="session",
            cleanup_target_id=session_id,
        )
        self._system_cleanup_events.append(
            CleanupRequest(
                target_kind="session",
                target_id=session_id,
                reason=reason,
                force=True,
            )
        )
        transfer_ids = self._transfer_ids_for_session_locked(session_id)
        for reservation_id, reservation in list(self._reservations.items()):
            if reservation.session_id == session_id:
                _merge_removed(
                    removed,
                    self._release_reservation_and_count_locked(
                        reservation_id,
                        final_state=TransferStatusState.CANCELED,
                        cleanup_reason=reason,
                    ),
                )
        for transfer_id in transfer_ids:
            status = self._transfer_statuses.get(transfer_id)
            if status is None or status.state in _TERMINAL_TRANSFER_STATES:
                continue
            self._mark_transfer_terminal_locked(
                transfer_id,
                TransferStatusState.CANCELED,
                error=reason,
            )
            if removed is not None:
                removed["transfers"] = int(removed["transfers"]) + 1
        for gpu in session.relay_gpus:
            quota = self._relay_quotas.get(gpu)
            if quota is not None:
                quota.sessions.discard(session_id)
        self._connection_scoped_sessions.discard(session_id)
        self._connection_scoped_session_connections.pop(session_id, None)
        removed_jobs = self._remove_session_jobs_and_buffers_locked(session_id)
        if removed is not None:
            removed["sessions"] = int(removed["sessions"]) + 1
            removed["jobs"] = int(removed["jobs"]) + removed_jobs["jobs"]
            removed["buffers"] = int(removed["buffers"]) + removed_jobs["buffers"]
        return session

    def _remove_session_jobs_and_buffers_locked(self, session_id: str) -> dict[str, int]:
        job_ids = {
            job_id
            for job_id, job in self._jobs.items()
            if job.session_id == session_id
        }
        removed = {"jobs": 0, "buffers": 0}
        for job_id in job_ids:
            if self._jobs.pop(job_id, None) is not None:
                removed["jobs"] += 1
            self._job_peer_identities.pop(job_id, None)
        for buffer_id, buffer in list(self._buffers.items()):
            if buffer.job_id in job_ids:
                self._buffers.pop(buffer_id, None)
                removed["buffers"] += 1
        self._session_peer_identities.pop(session_id, None)
        return removed

    def _cleanup_connection_scoped_sessions_locked(
        self,
        peer_identity: PeerIdentity | None,
        connection_id: str | None = None,
        reason: str = "socket_disconnect",
    ) -> dict[str, int]:
        removed = _empty_removed_summary()
        if peer_identity is None:
            return removed
        for session_id in sorted(tuple(self._connection_scoped_sessions)):
            if connection_id is not None:
                session_connection_id = self._connection_scoped_session_connections.get(
                    session_id
                )
                if session_connection_id != str(connection_id):
                    continue
            session_peer = self._session_peer_identities.get(session_id)
            if not _peer_identity_same_connection(session_peer, peer_identity):
                continue
            self._close_session_locked(session_id, reason=reason, removed=removed)
        return removed

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
                    "job_peer_identities": {
                        key: asdict(value)
                        for key, value in self._job_peer_identities.items()
                    },
                    "buffers": {key: asdict(value) for key, value in self._buffers.items()},
                    "sessions": {key: asdict(value) for key, value in self._sessions.items()},
                    "session_peer_identities": {
                        key: asdict(value)
                        for key, value in self._session_peer_identities.items()
                    },
                    "reservations": {
                        key: asdict(value) for key, value in self._reservations.items()
                    },
                    "staging_records": {
                        key: dict(value) for key, value in self._staging_records.items()
                    },
                    "audit_records": [dict(record) for record in self._audit_records],
                    "connection_scoped_sessions": sorted(
                        self._connection_scoped_sessions
                    ),
                    "transfer_statuses": {
                        key: asdict(value) for key, value in self._transfer_statuses.items()
                    },
                    "transfer_queue": [
                        dict(self._transfer_queue_records[transfer_id])
                        for transfer_id in self._transfer_queue
                        if transfer_id in self._transfer_queue_records
                    ],
                    "runtime_resource_state": self._runtime_resource_state_locked(
                        now=now,
                    ),
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

    def handle_request(
        self,
        request: DaemonRequest,
        connection_id: str | None = None,
    ) -> DaemonResponse:
        try:
            return self._handle_request(request, connection_id=connection_id)
        except (KeyError, TypeError, ValueError) as exc:
            return DaemonResponse(ok=False, error=f"invalid request: {exc}")

    def _handle_request(
        self,
        request: DaemonRequest,
        connection_id: str | None = None,
    ) -> DaemonResponse:
        if request.request_type == RequestType.REGISTER_JOB:
            payload = request.payload
            return self.register_job(
                job_id=str(payload["job_id"]),
                user_id=payload.get("user_id"),
                session_id=payload.get("session_id"),
                container_id=payload.get("container_id"),
                process_id=payload.get("process_id"),
                weight=float(payload.get("weight", 1.0)),
                peer_identity=request.peer_identity,
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
                peer_identity=request.peer_identity,
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
                peer_identity=request.peer_identity,
                connection_scoped=bool(payload.get("connection_scoped", False)),
                connection_id=connection_id,
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
                intent_id=payload.get("intent_id"),
                topology_snapshot_id=payload.get("topology_snapshot_id"),
                workload_kind=str(payload.get("workload_kind", "generic")),
                priority=int(payload.get("priority", 0)),
                peer_identity=request.peer_identity,
            )
        if request.request_type == RequestType.SUBMIT_TRANSFER_INTENT:
            payload = request.payload
            intent_payload = payload.get("intent")
            if not isinstance(intent_payload, dict):
                return DaemonResponse(ok=False, error="intent is required")
            return self.submit_transfer_intent(
                TransferIntent(**intent_payload),
                peer_identity=request.peer_identity,
            )
        if request.request_type == RequestType.WAIT_TRANSFER_RECEIPT:
            payload = request.payload
            return self.wait_transfer_receipt(
                intent_id=str(payload["intent_id"]),
                timeout_seconds=payload.get("timeout_seconds"),
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
                peer_identity=request.peer_identity,
            )
        if request.request_type == RequestType.AUTHORIZE_WORKER_TRANSFER:
            return self.authorize_worker_transfer(
                WorkerTransferAuthorizationRequest(**request.payload),
                peer_identity=request.peer_identity,
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
        if request.request_type == RequestType.INVALIDATE_TOPOLOGY:
            return self.invalidate_topology()
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
            "topology_snapshot_id": inventory.topology_snapshot_id(),
            "topology_version": inventory.version,
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
            "path_capabilities": _relay_path_capabilities(
                inventory,
                relay_gpu=relay,
                target_gpu=target,
                fabric_links=fabric_links,
            ),
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

    def handle_wire_message(
        self,
        data: bytes | str,
        peer_identity: PeerIdentity | None = None,
        connection_id: str | None = None,
    ) -> DaemonResponse:
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
                peer_identity=peer_identity,
            )
        except Exception as exc:
            return DaemonResponse(ok=False, error=f"invalid request: {exc}")
        return self.handle_request(request, connection_id=connection_id)

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
                    peer_identity = _peer_identity_from_socket(conn)
                    connection_id = str(uuid.uuid4())
                    data = b""
                    try:
                        while True:
                            chunk = conn.recv(65536)
                            if not chunk:
                                break
                            data += chunk
                            while b"\n" in data:
                                line, _, data = data.partition(b"\n")
                                if not line:
                                    continue
                                response = self.handle_wire_message(
                                    line,
                                    peer_identity=peer_identity,
                                    connection_id=connection_id,
                                )
                                conn.sendall(
                                    (json.dumps(asdict(response)) + "\n").encode("utf-8")
                                )
                    finally:
                        with self._lock:
                            self._cleanup_connection_scoped_sessions_locked(
                                peer_identity,
                                connection_id=connection_id,
                                reason="socket_disconnect",
                            )
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


def _topology_unavailable_response() -> DaemonResponse:
    return DaemonResponse(
        ok=False,
        error="topology provider is required; synthetic topology is test fixture only",
    )


def _bind_job_identity_to_peer(
    *,
    user_id: str | None,
    process_id: int | None,
    container_id: str | None,
    peer_identity: PeerIdentity | None,
) -> tuple[str | None, int | None, str | None]:
    if peer_identity is None or not peer_identity.authenticated:
        return user_id, process_id, container_id
    if user_id is not None and str(user_id) != str(peer_identity.user_id):
        raise ValueError("job user_id does not match authenticated peer")
    if (
        process_id is not None
        and peer_identity.process_id is not None
        and int(process_id) != int(peer_identity.process_id)
    ):
        raise ValueError("job process_id does not match authenticated peer")
    if (
        container_id is not None
        and peer_identity.container_id is not None
        and str(container_id) != str(peer_identity.container_id)
    ):
        raise ValueError("job container_id does not match authenticated peer")
    return (
        str(peer_identity.user_id),
        peer_identity.process_id if process_id is None else int(process_id),
        peer_identity.container_id if container_id is None else str(container_id),
    )


def _validate_peer_owner_match(
    *,
    expected: PeerIdentity | None,
    actual: PeerIdentity | None,
    owner_name: str,
) -> None:
    if expected is None or actual is None:
        return
    if not expected.authenticated or not actual.authenticated:
        return
    if str(expected.user_id) != str(actual.user_id):
        raise ValueError(f"{owner_name} owner does not match authenticated peer")


def _peer_identity_same_connection(
    expected: PeerIdentity | None,
    actual: PeerIdentity | None,
) -> bool:
    if expected is None or actual is None:
        return False
    if expected.authenticated and actual.authenticated:
        return (
            str(expected.user_id) == str(actual.user_id)
            and expected.process_id == actual.process_id
            and expected.group_id == actual.group_id
        )
    return (
        expected.authenticated == actual.authenticated
        and expected.source == actual.source
        and expected.unsupported_reason == actual.unsupported_reason
    )


def _peer_identity_from_socket(conn: socket.socket) -> PeerIdentity:
    if hasattr(socket, "SO_PEERCRED"):
        try:
            credentials = conn.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            pid, uid, gid = struct.unpack("3i", credentials)
            return PeerIdentity(
                authenticated=True,
                source="unix_socket_peercred",
                user_id=str(uid),
                process_id=pid,
                group_id=gid,
            )
        except OSError as exc:
            return PeerIdentity(
                authenticated=False,
                source="unix_socket_peercred",
                unsupported_reason=str(exc),
            )
    return PeerIdentity(
        authenticated=False,
        source="unix_socket",
        unsupported_reason="SO_PEERCRED is unavailable on this platform",
    )


def _relay_path_capabilities(
    inventory,
    *,
    relay_gpu: int,
    target_gpu: int | None,
    fabric_links: list[dict[str, object]],
) -> dict[str, object]:
    pcie_paths = [
        path for path in inventory.pcie_paths if path.device_id == int(relay_gpu)
    ]
    pcie_path = pcie_paths[0] if pcie_paths else None
    enabled_fabric_links = [
        link for link in fabric_links if bool(link.get("enabled", False))
    ]
    fabric_bandwidths = [
        float(link.get("bandwidth_gbps", 0.0) or 0.0)
        for link in enabled_fabric_links
    ]
    return {
        "relay_gpu": int(relay_gpu),
        "target_gpu": target_gpu,
        "has_pcie_path": pcie_path is not None,
        "pcie_root_complex": None if pcie_path is None else pcie_path.root_complex,
        "pcie_numa_node": None if pcie_path is None else pcie_path.numa_node,
        "pcie_link_generation": (
            None if pcie_path is None else pcie_path.link_generation
        ),
        "pcie_link_width": None if pcie_path is None else pcie_path.link_width,
        "pcie_negotiated_speed_gtps": (
            None if pcie_path is None else pcie_path.negotiated_speed_gtps
        ),
        "pcie_bandwidth_gbps": (
            0.0 if pcie_path is None else pcie_path.bandwidth_gbps
        ),
        "pcie_bandwidth_source": (
            None if pcie_path is None else pcie_path.bandwidth_source
        ),
        "pcie_switch_hierarchy": (
            [] if pcie_path is None else list(pcie_path.switch_hierarchy)
        ),
        "fabric_link_count": len(fabric_links),
        "enabled_fabric_link_count": len(enabled_fabric_links),
        "fabric_kinds": sorted(
            {str(link.get("fabric")) for link in enabled_fabric_links}
        ),
        "fabric_capabilities": sorted(
            {
                str(link.get("capability"))
                for link in enabled_fabric_links
                if link.get("capability") is not None
            }
        ),
        "fabric_bandwidth_gbps": sum(fabric_bandwidths),
        "p2p_enabled": bool(enabled_fabric_links),
    }


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


def _ticket_ranges_for_plan(
    plan: dict[str, object],
    *,
    direction: str,
) -> tuple[dict[str, int], ...]:
    if not isinstance(plan, dict):
        raise ValueError("transfer plan is unavailable")
    ranges: list[dict[str, int]] = []
    requested_direction = str(direction).lower()
    for assignment in plan.get("assignments", ()) or ():
        if not isinstance(assignment, dict):
            raise ValueError("transfer plan assignment must be an object")
        path = assignment.get("path")
        if not isinstance(path, dict):
            raise ValueError("transfer plan assignment path must be an object")
        if str(path.get("direction", "")).lower() != requested_direction:
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
        raise ValueError("daemon plan has no authorized chunks")
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


def _intent_chunk_bytes(intent: TransferIntent) -> int:
    for source in (intent.policy_hints, intent.metadata):
        if not isinstance(source, dict):
            continue
        value = source.get("chunk_bytes")
        if value is None:
            continue
        chunk_bytes = int(value)
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        return chunk_bytes
    return max(1, int(intent.total_bytes))


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


def _empty_removed_summary() -> dict[str, int]:
    return {
        "jobs": 0,
        "buffers": 0,
        "sessions": 0,
        "reservations": 0,
        "staging_records": 0,
        "transfers": 0,
    }


def _merge_removed(
    target: dict[str, int] | None,
    source: dict[str, int] | None,
) -> dict[str, int] | None:
    if target is None:
        return target
    if source is None:
        return target
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)
    return target
