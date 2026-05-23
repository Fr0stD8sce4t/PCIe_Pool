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
    REGISTER_SESSION = "REGISTER_SESSION"
    PROFILE = "PROFILE"
    GET_PROFILE = "GET_PROFILE"
    PUT_PROFILE = "PUT_PROFILE"
    INVALIDATE_PROFILE = "INVALIDATE_PROFILE"
    PLAN_TRANSFER = "PLAN_TRANSFER"
    RESERVE_TRANSFER = "RESERVE_TRANSFER"
    RELEASE_TRANSFER = "RELEASE_TRANSFER"
    CLOSE_SESSION = "CLOSE_SESSION"


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
