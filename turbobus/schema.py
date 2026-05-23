from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransferMode(str, Enum):
    AUTO = "auto"
    POOL = "pool"
    DIRECT = "direct"
    RELAY = "relay"


@dataclass(frozen=True)
class AutoTransferDecision:
    requested_mode: TransferMode
    resolved_mode: TransferMode
    request_bytes: int
    request_chunks: int
    direct_h2d_bw_gbps: float
    relay_effective_bw_gbps: float
    eligible_relay_devices: tuple[int, ...]
    reason: str


class RequestType(str, Enum):
    REGISTER_JOB = "REGISTER_JOB"
    REGISTER_SESSION = "REGISTER_SESSION"
    REGISTER_BUFFER = "REGISTER_BUFFER"
    PROFILE = "PROFILE"
    GET_PROFILE = "GET_PROFILE"
    PUT_PROFILE = "PUT_PROFILE"
    INVALIDATE_PROFILE = "INVALIDATE_PROFILE"
    PLAN_TRANSFER = "PLAN_TRANSFER"
    TRANSFER_STATUS = "TRANSFER_STATUS"
    RESERVE_TRANSFER = "RESERVE_TRANSFER"
    ISSUE_LEASE = "ISSUE_LEASE"
    RELEASE_TRANSFER = "RELEASE_TRANSFER"
    CLEANUP = "CLEANUP"
    CLOSE_SESSION = "CLOSE_SESSION"


class TransferStatusState(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class JobIdentity:
    job_id: str
    user_id: str | None = None
    session_id: str | None = None
    container_id: str | None = None
    process_id: int | None = None

    def __post_init__(self) -> None:
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        if self.process_id is not None and int(self.process_id) < 0:
            raise ValueError("process_id must be non-negative")
        object.__setattr__(self, "job_id", str(self.job_id))
        if self.user_id is not None:
            object.__setattr__(self, "user_id", str(self.user_id))
        if self.session_id is not None:
            object.__setattr__(self, "session_id", str(self.session_id))
        if self.container_id is not None:
            object.__setattr__(self, "container_id", str(self.container_id))
        if self.process_id is not None:
            object.__setattr__(self, "process_id", int(self.process_id))


@dataclass(frozen=True)
class BufferRegistration:
    buffer_id: str
    job_id: str
    kind: str
    size_bytes: int
    device_index: int | None = None
    address: int | None = None
    pinned: bool = False

    def __post_init__(self) -> None:
        if not str(self.buffer_id).strip():
            raise ValueError("buffer_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        size_bytes = int(self.size_bytes)
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.device_index is not None and int(self.device_index) < 0:
            raise ValueError("device_index must be non-negative")
        if self.address is not None and int(self.address) < 0:
            raise ValueError("address must be non-negative")
        if not str(self.kind).strip():
            raise ValueError("kind must be non-empty")
        object.__setattr__(self, "buffer_id", str(self.buffer_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "size_bytes", size_bytes)
        if self.device_index is not None:
            object.__setattr__(self, "device_index", int(self.device_index))
        if self.address is not None:
            object.__setattr__(self, "address", int(self.address))


@dataclass(frozen=True)
class LeaseToken:
    lease_id: str
    session_id: str
    relay_gpu: int
    job_id: str | None = None
    issued_at: float = 0.0
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.lease_id).strip():
            raise ValueError("lease_id must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")
        if int(self.relay_gpu) < 0:
            raise ValueError("relay_gpu must be non-negative")
        issued_at = float(self.issued_at)
        expires_at = float(self.expires_at)
        if expires_at and expires_at < issued_at:
            raise ValueError("expires_at must not be earlier than issued_at")
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "relay_gpu", int(self.relay_gpu))
        if self.job_id is not None:
            object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True)
class TransferStatus:
    transfer_id: str
    job_id: str
    state: TransferStatusState
    bytes_total: int
    bytes_completed: int = 0
    session_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        bytes_total = int(self.bytes_total)
        bytes_completed = int(self.bytes_completed)
        if bytes_total < 0:
            raise ValueError("bytes_total must be non-negative")
        if bytes_completed < 0:
            raise ValueError("bytes_completed must be non-negative")
        if bytes_completed > bytes_total:
            raise ValueError("bytes_completed cannot exceed bytes_total")
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "state", TransferStatusState(self.state))
        object.__setattr__(self, "bytes_total", bytes_total)
        object.__setattr__(self, "bytes_completed", bytes_completed)
        if self.session_id is not None:
            object.__setattr__(self, "session_id", str(self.session_id))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))


@dataclass(frozen=True)
class CleanupRequest:
    target_kind: str
    target_id: str
    reason: str
    force: bool = False

    def __post_init__(self) -> None:
        if not str(self.target_kind).strip():
            raise ValueError("target_kind must be non-empty")
        if not str(self.target_id).strip():
            raise ValueError("target_id must be non-empty")
        if not str(self.reason).strip():
            raise ValueError("reason must be non-empty")
        object.__setattr__(self, "target_kind", str(self.target_kind))
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "reason", str(self.reason))


@dataclass
class Session:
    session_id: str
    target_gpu: int
    relay_gpus: list[int]
    max_inflight_chunks: int
    active_chunks: int = 0
    active: bool = True
    created_at: float = 0.0
    last_seen: float = 0.0
    closed_at: float | None = None


@dataclass
class RelayQuota:
    relay_gpu: int
    max_sessions: int = 1
    max_inflight_chunks: int = 8
    sessions: set[str] = field(default_factory=set)
    active_chunks: int = 0

    def can_attach(self) -> bool:
        return len(self.sessions) < self.max_sessions

    def can_reserve(self, chunks: int) -> bool:
        return self.active_chunks + chunks <= self.max_inflight_chunks


@dataclass
class TransferReservation:
    reservation_id: str
    session_id: str
    relay_gpu: int
    chunks: int
    bytes: int = 0
    direction: str = "unknown"


@dataclass
class DaemonRequest:
    request_type: RequestType
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DaemonResponse:
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


__all__ = [
    "AutoTransferDecision",
    "BufferRegistration",
    "CleanupRequest",
    "DaemonRequest",
    "DaemonResponse",
    "JobIdentity",
    "LeaseToken",
    "RelayQuota",
    "RequestType",
    "Session",
    "TransferMode",
    "TransferReservation",
    "TransferStatus",
    "TransferStatusState",
]
