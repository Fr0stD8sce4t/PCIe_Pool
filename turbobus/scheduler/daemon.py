from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, Mapping

from ..planner_engine import PlannerEngine
from ..planner_types import PlannerLease, PlannerStats, PlannerTransferPlan
from ..schema import (
    RelayQuota,
    SchedulingDecision,
    SchedulingDecisionState,
    Session,
    TransferMode,
    WorkloadKind,
)


@dataclass(frozen=True)
class _RelayProfile:
    relay_device: int
    target_device: int
    h2d_bw_gbps: float
    d2h_bw_gbps: float
    p2p_bw_gbps: float
    effective_bw_gbps: float
    effective_d2h_bw_gbps: float
    p2p_enabled: bool


@dataclass(frozen=True)
class _Profile:
    target_device: int
    direct_h2d_bw_gbps: float
    direct_d2h_bw_gbps: float
    relays: tuple[_RelayProfile, ...]


@dataclass(frozen=True)
class _RuntimeView:
    job_id: str | None
    workload_kind: str
    priority: int
    busy_relays: frozenset[int]
    job_weight: float
    total_weight: float
    current_job_active_bytes: int
    total_active_bytes: int
    request_charge_bytes: float
    average_weighted_active_bytes: float
    current_weighted_active_bytes: float
    projected_weighted_active_bytes: float
    fairness_threshold_bytes: float
    active_transfer_count: int
    queued_transfer_count: int

    def policy_metadata(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "job_weight": self.job_weight,
            "total_weight": self.total_weight,
            "workload_kind": self.workload_kind,
            "priority": self.priority,
            "request_charge_bytes": self.request_charge_bytes,
            "current_job_active_bytes": self.current_job_active_bytes,
            "total_active_bytes": self.total_active_bytes,
            "current_weighted_active_bytes": self.current_weighted_active_bytes,
            "projected_weighted_active_bytes": self.projected_weighted_active_bytes,
            "average_weighted_active_bytes": self.average_weighted_active_bytes,
            "fairness_threshold_bytes": self.fairness_threshold_bytes,
            "busy_relays": tuple(sorted(self.busy_relays)),
            "active_transfer_count": self.active_transfer_count,
            "queued_transfer_count": self.queued_transfer_count,
        }


class DaemonScheduler:
    def __init__(
        self,
        planner: PlannerEngine | None = None,
        lease_id_factory: Callable[[], str] | None = None,
        decision_id_factory: Callable[[], str] | None = None,
        lease_seconds: float = 30.0,
    ) -> None:
        self._planner = planner or PlannerEngine()
        self._lease_id_factory = lease_id_factory or (lambda: str(uuid.uuid4()))
        self._decision_id_factory = decision_id_factory or (lambda: str(uuid.uuid4()))
        self._lease_seconds = max(0.0, float(lease_seconds))

    def plan_transfer(
        self,
        *,
        session: Session,
        profile_entry: Mapping[str, object] | None,
        relay_quotas: Mapping[int, RelayQuota],
        total_bytes: int,
        chunk_bytes: int,
        ranges: tuple[Mapping[str, int], ...] | None = None,
        mode: TransferMode | str = TransferMode.POOL,
        direction: str = "h2d",
        runtime_state: Mapping[str, object] | None = None,
        workload_kind: WorkloadKind | str = WorkloadKind.GENERIC,
        priority: int = 0,
        now: float = 0.0,
        job_id: str | None = None,
        intent_id: str | None = None,
        topology_snapshot_id: str | None = None,
        defer_relay_admission: bool = False,
    ) -> SchedulingDecision:
        total_bytes = int(total_bytes)
        chunk_bytes = int(chunk_bytes)
        normalized_ranges = _normalize_ranges(ranges)
        direction = str(direction).lower()
        if total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if normalized_ranges is not None:
            range_bytes = sum(item["bytes"] for item in normalized_ranges)
            if range_bytes != total_bytes:
                raise ValueError("range bytes must match total_bytes")
        if direction not in {"h2d", "d2h"}:
            raise ValueError("direction must be h2d or d2h")
        if not session.active:
            raise ValueError("session is closed")

        requested_mode = _parse_transfer_mode(mode)
        planning_mode = TransferMode.POOL if requested_mode is TransferMode.AUTO else requested_mode
        runtime_view = _runtime_view(
            runtime_state=runtime_state,
            job_id=job_id,
            total_bytes=total_bytes,
            workload_kind=workload_kind,
            priority=priority,
        )
        profile, fallback_reason = self._profile_for_planning(
            profile_entry=profile_entry,
            session=session,
            relay_quotas=relay_quotas,
            direction=direction,
            runtime_view=runtime_view,
            defer_relay_admission=defer_relay_admission,
        )
        if (
            fallback_reason is None
            and planning_mode is not TransferMode.DIRECT
            and session.relay_gpus
            and not profile.relays
        ):
            fallback_reason = "no daemon-approved relay path"

        plan = self._plan_or_direct(
            total_bytes=total_bytes,
            chunk_bytes=chunk_bytes,
            ranges=normalized_ranges,
            profile=profile,
            mode=planning_mode,
            direction=direction,
        )
        leases, lease_error = self._leases_for_plan(
            plan=plan,
            session=session,
            relay_quotas=relay_quotas,
            direction=direction,
            now=now,
            job_id=job_id,
            defer_relay_admission=defer_relay_admission,
        )
        fairness_fallback = _fairness_fallback_for_plan(
            plan=plan,
            runtime_view=runtime_view,
        )
        if fairness_fallback is not None:
            lease_error = fairness_fallback
        if lease_error is not None:
            fallback_reason = lease_error
            plan = self._direct_plan(
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                ranges=normalized_ranges,
                profile=profile,
                direction=direction,
            )
            leases = ()

        stats = _stats_for_plan(
            plan,
            requested_mode=requested_mode,
            fallback_reason=fallback_reason,
        )
        return SchedulingDecision(
            decision_id=self._decision_id_factory(),
            intent_id=_contract_id(
                intent_id,
                prefix="intent",
                fallback=session.session_id,
            ),
            topology_snapshot_id=_contract_id(
                topology_snapshot_id,
                prefix="topology",
                fallback=session.session_id,
            ),
            job_id=str(job_id or session.session_id),
            session_id=session.session_id,
            state=(
                SchedulingDecisionState.FALLBACK
                if fallback_reason is not None
                else SchedulingDecisionState.PLANNED
            ),
            plan=plan.as_dict(),
            path_summary=_path_summary_for_plan(plan),
            fallback_reason=fallback_reason,
            issued_at=float(now),
            metadata={
                "leases": [lease.as_dict() for lease in leases],
                "stats": stats.as_dict(),
                "runtime_state": _runtime_state_metadata(runtime_state),
                "policy": runtime_view.policy_metadata(),
            },
        )

    def _profile_for_planning(
        self,
        *,
        profile_entry: Mapping[str, object] | None,
        session: Session,
        relay_quotas: Mapping[int, RelayQuota],
        direction: str,
        runtime_view: "_RuntimeView",
        defer_relay_admission: bool,
    ) -> tuple[_Profile, str | None]:
        payload = _profile_payload(profile_entry)
        if payload is None:
            return _direct_fallback_profile(session.target_gpu), "daemon profile miss"

        relays = []
        allowed_relays = set(int(gpu) for gpu in session.relay_gpus)
        for relay in payload.get("relays", []) or []:
            if not isinstance(relay, Mapping):
                continue
            relay_device = int(relay["relay_device"])
            if relay_device not in allowed_relays:
                continue
            if (
                not defer_relay_admission
                and not _relay_has_capacity(session, relay_quotas.get(relay_device))
            ):
                continue
            if not defer_relay_admission and relay_device in runtime_view.busy_relays:
                continue
            relays.append(
                _RelayProfile(
                    relay_device=relay_device,
                    target_device=int(relay.get("target_device", session.target_gpu)),
                    h2d_bw_gbps=float(relay.get("h2d_bw_gbps", 0.0) or 0.0),
                    d2h_bw_gbps=float(relay.get("d2h_bw_gbps", 0.0) or 0.0),
                    p2p_bw_gbps=float(relay.get("p2p_bw_gbps", 0.0) or 0.0),
                    effective_bw_gbps=float(relay.get("effective_bw_gbps", 0.0) or 0.0),
                    effective_d2h_bw_gbps=float(
                        relay.get("effective_d2h_bw_gbps", 0.0) or 0.0
                    ),
                    p2p_enabled=bool(relay.get("p2p_enabled", False)),
                )
            )

        direct_h2d = float(payload.get("direct_h2d_bw_gbps", 0.0) or 0.0)
        direct_d2h = float(payload.get("direct_d2h_bw_gbps", 0.0) or direct_h2d)
        if direction == "h2d" and direct_h2d <= 0.0:
            return _direct_fallback_profile(session.target_gpu), "daemon direct profile invalid"
        if direction == "d2h" and direct_d2h <= 0.0:
            direct_d2h = direct_h2d

        return (
            _Profile(
                target_device=int(payload.get("target_device", session.target_gpu)),
                direct_h2d_bw_gbps=direct_h2d,
                direct_d2h_bw_gbps=direct_d2h,
                relays=tuple(relays),
            ),
            None,
        )

    def _plan_or_direct(
        self,
        *,
        total_bytes: int,
        chunk_bytes: int,
        ranges: tuple[Mapping[str, int], ...] | None,
        profile: _Profile,
        mode: TransferMode,
        direction: str,
    ) -> PlannerTransferPlan:
        try:
            if ranges is not None:
                return self._planner.plan_ranges(
                    ranges,
                    chunk_bytes,
                    profile,
                    mode,
                    direction=direction,
                )
            return self._planner.plan(
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                profile=profile,
                mode=mode,
                direction=direction,
            )
        except RuntimeError:
            return self._direct_plan(
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                ranges=ranges,
                profile=profile,
                direction=direction,
            )

    def _direct_plan(
        self,
        *,
        total_bytes: int,
        chunk_bytes: int,
        ranges: tuple[Mapping[str, int], ...] | None = None,
        profile: _Profile,
        direction: str,
    ) -> PlannerTransferPlan:
        direct_profile = _Profile(
            target_device=profile.target_device,
            direct_h2d_bw_gbps=profile.direct_h2d_bw_gbps or 1.0,
            direct_d2h_bw_gbps=(
                profile.direct_d2h_bw_gbps or profile.direct_h2d_bw_gbps or 1.0
            ),
            relays=(),
        )
        if ranges is not None:
            return self._planner.plan_ranges(
                ranges,
                chunk_bytes,
                direct_profile,
                mode=TransferMode.DIRECT,
                direction=direction,
            )
        return self._planner.plan(
            total_bytes=total_bytes,
            chunk_bytes=chunk_bytes,
            profile=direct_profile,
            mode=TransferMode.DIRECT,
            direction=direction,
        )

    def _leases_for_plan(
        self,
        *,
        plan: PlannerTransferPlan,
        session: Session,
        relay_quotas: Mapping[int, RelayQuota],
        direction: str,
        now: float,
        job_id: str | None,
        defer_relay_admission: bool,
    ) -> tuple[tuple[PlannerLease, ...], str | None]:
        lease_specs: list[tuple[int, int, int]] = []
        for assignment in plan.assignments:
            if assignment.path.kind != "relay":
                continue
            relay_device = int(assignment.path.relay_device)
            chunks = len(assignment.chunks)
            bytes_limit = sum(chunk.bytes for chunk in assignment.chunks)
            if chunks <= 0:
                continue
            lease_specs.append((relay_device, chunks, bytes_limit))

        if not lease_specs:
            return (), None

        total_chunks = sum(chunks for _, chunks, _ in lease_specs)
        if (
            not defer_relay_admission
            and session.active_chunks + total_chunks > session.max_inflight_chunks
        ):
            return (), "session chunk quota is unavailable"

        leases: list[PlannerLease] = []
        for relay_device, chunks, bytes_limit in lease_specs:
            if relay_device not in session.relay_gpus:
                return (), "relay GPU is not assigned to this session"
            quota = relay_quotas.get(relay_device)
            if quota is None:
                return (), "relay chunk quota is unavailable"
            if not defer_relay_admission and not quota.can_reserve(chunks):
                return (), "relay chunk quota is unavailable"
            leases.append(
                PlannerLease(
                    lease_id=self._lease_id_factory(),
                    session_id=session.session_id,
                    relay_device=relay_device,
                    chunk_limit=chunks,
                    bytes_limit=bytes_limit,
                    direction=direction,
                    granted_at=float(now),
                    expires_at=float(now) + self._lease_seconds,
                    job_id=job_id,
                )
            )
        return tuple(leases), None


def _stats_for_plan(
    plan: PlannerTransferPlan,
    *,
    requested_mode: TransferMode,
    fallback_reason: str | None,
) -> PlannerStats:
    direct_bytes = 0
    relay_bytes = 0
    direct_chunks = 0
    relay_chunks = 0
    relay_path_count = 0
    for assignment in plan.assignments:
        assignment_bytes = sum(chunk.bytes for chunk in assignment.chunks)
        if assignment.path.kind == "relay":
            relay_bytes += assignment_bytes
            relay_chunks += len(assignment.chunks)
            relay_path_count += 1
        else:
            direct_bytes += assignment_bytes
            direct_chunks += len(assignment.chunks)
    return PlannerStats(
        bytes=int(plan.total_bytes),
        direct_bytes=direct_bytes,
        relay_bytes=relay_bytes,
        direct_chunks=direct_chunks,
        relay_chunks=relay_chunks,
        path_count=len(plan.assignments),
        relay_path_count=relay_path_count,
        fallback_reason=fallback_reason,
        requested_mode=requested_mode,
        resolved_mode=_resolved_mode_for_plan(plan),
    )


def scheduling_decision_leases(
    decision: SchedulingDecision,
) -> tuple[PlannerLease, ...]:
    if not isinstance(decision, SchedulingDecision):
        raise TypeError("decision must be a SchedulingDecision")
    leases = decision.metadata.get("leases", ())
    if not isinstance(leases, tuple | list):
        raise ValueError("scheduling decision metadata leases must be a sequence")
    return tuple(_planner_lease_from_payload(item) for item in leases)


def scheduling_decision_stats(decision: SchedulingDecision) -> PlannerStats:
    if not isinstance(decision, SchedulingDecision):
        raise TypeError("decision must be a SchedulingDecision")
    payload = decision.metadata.get("stats", {})
    if not isinstance(payload, Mapping):
        raise ValueError("scheduling decision metadata stats must be a mapping")
    return PlannerStats(
        bytes=int(payload.get("bytes", 0)),
        direct_bytes=int(payload.get("direct_bytes", 0)),
        relay_bytes=int(payload.get("relay_bytes", 0)),
        direct_chunks=int(payload.get("direct_chunks", 0)),
        relay_chunks=int(payload.get("relay_chunks", 0)),
        path_count=int(payload.get("path_count", 0)),
        relay_path_count=int(payload.get("relay_path_count", 0)),
        fallback_reason=payload.get("fallback_reason"),
        requested_mode=payload.get("requested_mode", TransferMode.POOL),
        resolved_mode=payload.get("resolved_mode", TransferMode.POOL),
    )


def _planner_lease_from_payload(payload: object) -> PlannerLease:
    if not isinstance(payload, Mapping):
        raise ValueError("scheduling decision lease must be a mapping")
    return PlannerLease(
        lease_id=str(payload["lease_id"]),
        session_id=str(payload["session_id"]),
        relay_device=int(payload["relay_device"]),
        chunk_limit=int(payload["chunk_limit"]),
        bytes_limit=int(payload.get("bytes_limit", 0)),
        direction=str(payload.get("direction", "unknown")),
        granted_at=float(payload.get("granted_at", 0.0)),
        expires_at=float(payload.get("expires_at", 0.0)),
        active=bool(payload.get("active", True)),
        job_id=payload.get("job_id"),
        reason=payload.get("reason"),
    )


def _path_summary_for_plan(
    plan: PlannerTransferPlan,
) -> tuple[dict[str, object], ...]:
    summary: list[dict[str, object]] = []
    for assignment in plan.assignments:
        path = assignment.path
        bytes_count = sum(chunk.bytes for chunk in assignment.chunks)
        summary.append(
            {
                "kind": path.kind,
                "direction": path.direction,
                "target_device": path.target_device,
                "relay_device": path.relay_device,
                "bytes": bytes_count,
                "chunk_count": len(assignment.chunks),
            }
        )
    return tuple(summary)


def _runtime_state_metadata(
    runtime_state: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(runtime_state, Mapping):
        return {
            "version": 0,
            "queued_transfer_count": 0,
            "running_transfer_count": 0,
            "active_transfer_count": 0,
        }
    summary = runtime_state.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "version": int(runtime_state.get("version", 0) or 0),
        "queued_transfer_count": int(summary.get("queued_transfer_count", 0) or 0),
        "running_transfer_count": int(summary.get("running_transfer_count", 0) or 0),
        "active_transfer_count": int(summary.get("active_transfer_count", 0) or 0),
        "active_reservation_count": int(summary.get("active_reservation_count", 0) or 0),
        "active_lease_count": int(summary.get("active_lease_count", 0) or 0),
        "relay_staging_count": int(summary.get("relay_staging_count", 0) or 0),
        "relay_path_count": int(summary.get("relay_path_count", 0) or 0),
        "relay_path_bytes_total": int(summary.get("relay_path_bytes_total", 0) or 0),
        "active_bytes_by_direction": dict(
            summary.get("active_bytes_by_direction", {}) or {}
        ),
        "queued_bytes_by_direction": dict(
            summary.get("queued_bytes_by_direction", {}) or {}
        ),
        "active_resource_usage": dict(summary.get("active_resource_usage", {}) or {}),
    }


def _runtime_view(
    *,
    runtime_state: Mapping[str, object] | None,
    job_id: str | None,
    total_bytes: int,
    workload_kind: WorkloadKind | str,
    priority: int,
) -> _RuntimeView:
    normalized_job_id = None if job_id is None else str(job_id)
    workload = WorkloadKind(workload_kind).value
    active_paths = ()
    active_transfer_count = 0
    queued_transfer_count = 0
    job_runtime_state: Mapping[str, object] = {}
    if isinstance(runtime_state, Mapping):
        active_paths = runtime_state.get("active_paths", ()) or ()
        summary = runtime_state.get("summary", {})
        if isinstance(summary, Mapping):
            active_transfer_count = int(summary.get("active_transfer_count", 0) or 0)
            queued_transfer_count = int(summary.get("queued_transfer_count", 0) or 0)
            nested_jobs = summary.get("job_runtime_state", {})
            if isinstance(nested_jobs, Mapping):
                job_runtime_state = nested_jobs
        jobs = runtime_state.get("job_runtime_state", {})
        if isinstance(jobs, Mapping):
            job_runtime_state = jobs
    busy_relays: set[int] = set()
    for record in active_paths:
        if not isinstance(record, Mapping):
            continue
        if str(record.get("kind", "")).lower() != "relay":
            continue
        relay = record.get("relay_device")
        if relay is None:
            continue
        busy_relays.add(int(relay))

    total_weight = 0.0
    total_active_bytes = 0
    current_job_active_bytes = 0
    job_weight = 1.0
    for key, value in job_runtime_state.items():
        if not isinstance(value, Mapping):
            continue
        weight = max(0.0, float(value.get("weight", 1.0) or 1.0))
        total_weight += weight
        active_bytes = int(value.get("active_bytes_remaining", 0) or 0)
        total_active_bytes += active_bytes
        if normalized_job_id is not None and str(key) == normalized_job_id:
            job_weight = weight or 1.0
            current_job_active_bytes = active_bytes
    if total_weight <= 0.0:
        total_weight = max(1.0, job_weight)
    if normalized_job_id is not None and normalized_job_id not in job_runtime_state:
        total_weight += job_weight
    request_charge = float(total_bytes) * _workload_charge_multiplier(workload)
    if int(priority) > 0:
        request_charge = request_charge / (1.0 + min(int(priority), 9) * 0.1)
    current_weighted = current_job_active_bytes / max(job_weight, 1e-12)
    projected_weighted = (current_job_active_bytes + request_charge) / max(
        job_weight,
        1e-12,
    )
    average_weighted = (
        (total_active_bytes + request_charge) / max(total_weight, 1e-12)
    )
    return _RuntimeView(
        job_id=normalized_job_id,
        workload_kind=workload,
        priority=int(priority),
        busy_relays=frozenset(busy_relays),
        job_weight=job_weight,
        total_weight=total_weight,
        current_job_active_bytes=current_job_active_bytes,
        total_active_bytes=total_active_bytes,
        request_charge_bytes=request_charge,
        average_weighted_active_bytes=average_weighted,
        current_weighted_active_bytes=current_weighted,
        projected_weighted_active_bytes=projected_weighted,
        fairness_threshold_bytes=average_weighted * 1.25,
        active_transfer_count=active_transfer_count,
        queued_transfer_count=queued_transfer_count,
    )


def _workload_charge_multiplier(workload_kind: str) -> float:
    if workload_kind == WorkloadKind.KV_CACHE.value:
        return 0.75
    if workload_kind == WorkloadKind.TRAINING_STATE.value:
        return 1.25
    if workload_kind == WorkloadKind.OPTIMIZER_STATE.value:
        return 1.25
    return 1.0


def _fairness_fallback_for_plan(
    *,
    plan: PlannerTransferPlan,
    runtime_view: _RuntimeView,
) -> str | None:
    has_relay = any(assignment.path.kind == "relay" for assignment in plan.assignments)
    if not has_relay:
        return None
    if runtime_view.total_active_bytes <= 0:
        return None
    if runtime_view.projected_weighted_active_bytes <= runtime_view.fairness_threshold_bytes:
        return None
    return "weighted fairness limit prefers direct fallback"


def _contract_id(value: str | None, *, prefix: str, fallback: str) -> str:
    if value is not None and str(value).strip():
        return str(value)
    return f"{prefix}-{fallback}"


def _resolved_mode_for_plan(plan: PlannerTransferPlan) -> TransferMode:
    has_direct = any(assignment.path.kind == "direct" for assignment in plan.assignments)
    has_relay = any(assignment.path.kind == "relay" for assignment in plan.assignments)
    if has_direct and has_relay:
        return TransferMode.POOL
    if has_relay:
        return TransferMode.RELAY
    return TransferMode.DIRECT


def _profile_payload(profile_entry: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if not profile_entry:
        return None
    profile = profile_entry.get("profile")
    if isinstance(profile, Mapping):
        return profile
    return profile_entry


def _direct_fallback_profile(target_gpu: int) -> _Profile:
    return _Profile(
        target_device=int(target_gpu),
        direct_h2d_bw_gbps=1.0,
        direct_d2h_bw_gbps=1.0,
        relays=(),
    )


def _relay_has_capacity(session: Session, quota: RelayQuota | None) -> bool:
    return (
        quota is not None
        and session.active_chunks < session.max_inflight_chunks
        and quota.active_chunks < quota.max_inflight_chunks
    )


def _parse_transfer_mode(mode: TransferMode | str) -> TransferMode:
    if isinstance(mode, TransferMode):
        return mode
    value = str(mode)
    try:
        return TransferMode(value)
    except ValueError:
        return TransferMode[value.upper()]


def _normalize_ranges(
    ranges: tuple[Mapping[str, int], ...] | None,
) -> tuple[Mapping[str, int], ...] | None:
    if ranges is None:
        return None
    normalized = []
    for item in ranges:
        if not isinstance(item, Mapping):
            raise ValueError("ranges must contain mappings")
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


__all__ = [
    "DaemonScheduler",
    "SchedulingDecision",
    "scheduling_decision_leases",
    "scheduling_decision_stats",
]
