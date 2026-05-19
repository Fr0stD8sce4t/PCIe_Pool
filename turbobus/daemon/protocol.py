from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestType(str, Enum):
    REGISTER_SESSION = "REGISTER_SESSION"
    PROFILE = "PROFILE"
    FETCH_TO_GPU = "FETCH_TO_GPU"
    WAIT = "WAIT"
    CLOSE_SESSION = "CLOSE_SESSION"


@dataclass
class Session:
    session_id: str
    target_gpu: int
    relay_gpus: list[int]
    max_inflight_chunks: int
    active: bool = True


@dataclass
class RelayQuota:
    relay_gpu: int
    max_sessions: int = 1
    max_inflight_chunks: int = 8
    sessions: set[str] = field(default_factory=set)

    def can_attach(self) -> bool:
        return len(self.sessions) < self.max_sessions


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

