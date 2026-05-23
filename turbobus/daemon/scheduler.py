from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, Mapping

from ..planner_engine import PlannerEngine
from ..planner_types import PlannerLease, PlannerStats, PlannerTransferPlan
from ..schema import RelayQuota, Session, TransferMode


@dataclass(frozen=True)
class SchedulerDecision:
    plan: PlannerTransferPlan
    leases: tuple[PlannerLease, ...]
    stats: PlannerStats

    def as_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.as_dict(),
            "leases": [lease.as_dict() for lease in self.leases],
            "stats": self.stats.as_dict(),
        }


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


class DaemonScheduler:
    def __init__(
        self,
        planner: PlannerEngine | None = None,
        lease_id_factory: Callable[[], str] | None = None,
        lease_seconds: float = 30.0,
    ) -> None:
        self._planner = planner or PlannerEngine()
        self._lease_id_factory = lease_id_factory or (lambda: str(uuid.uuid4()))
        self._lease_seconds = max(0.0, float(lease_seconds))

    def plan_transfer(
        self,
        *,
        session: Session,
        profile_entry: Mapping[str, object] | None,
        relay_quotas: Mapping[int, RelayQuota],
        total_bytes: int,
        chunk_bytes: int,
        mode: TransferMode | str = TransferMode.POOL,
        direction: str = "h2d",
        now: float = 0.0,
        job_id: str | None = None,
    ) -> SchedulerDecision:
        total_bytes = int(total_bytes)
        chunk_bytes = int(chunk_bytes)
        direction = str(direction).lower()
        if total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if direction not in {"h2d", "d2h"}:
            raise ValueError("direction must be h2d or d2h")
        if not session.active:
            raise ValueError("session is closed")

        requested_mode = _parse_transfer_mode(mode)
        planning_mode = TransferMode.POOL if requested_mode is TransferMode.AUTO else requested_mode
        profile, fallback_reason = self._profile_for_planning(
            profile_entry=profile_entry,
            session=session,
            relay_quotas=relay_quotas,
            direction=direction,
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
        )
        if lease_error is not None:
            fallback_reason = lease_error
            plan = self._direct_plan(
                total_bytes=total_bytes,
                chunk_bytes=chunk_bytes,
                profile=profile,
                direction=direction,
            )
            leases = ()

        stats = _stats_for_plan(
            plan,
            requested_mode=requested_mode,
            fallback_reason=fallback_reason,
        )
        return SchedulerDecision(plan=plan, leases=leases, stats=stats)

    def _profile_for_planning(
        self,
        *,
        profile_entry: Mapping[str, object] | None,
        session: Session,
        relay_quotas: Mapping[int, RelayQuota],
        direction: str,
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
            if not _relay_has_capacity(session, relay_quotas.get(relay_device)):
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
        profile: _Profile,
        mode: TransferMode,
        direction: str,
    ) -> PlannerTransferPlan:
        try:
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
                profile=profile,
                direction=direction,
            )

    def _direct_plan(
        self,
        *,
        total_bytes: int,
        chunk_bytes: int,
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
        if session.active_chunks + total_chunks > session.max_inflight_chunks:
            return (), "session chunk quota is unavailable"

        leases: list[PlannerLease] = []
        for relay_device, chunks, bytes_limit in lease_specs:
            if relay_device not in session.relay_gpus:
                return (), "relay GPU is not assigned to this session"
            quota = relay_quotas.get(relay_device)
            if quota is None or not quota.can_reserve(chunks):
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


__all__ = ["DaemonScheduler", "SchedulerDecision"]
