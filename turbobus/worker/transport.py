from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkerServiceTransport(Protocol):
    def handle_message(self, message: str | bytes) -> str:
        raise NotImplementedError

    def handle_observability_message(self, message: str | bytes) -> str:
        raise NotImplementedError


@dataclass
class WorkerServiceLoopbackTransport:
    endpoint: WorkerServiceTransport

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, WorkerServiceTransport):
            raise TypeError("endpoint must be a WorkerServiceTransport")

    def handle_message(self, message: str | bytes) -> str:
        return self.endpoint.handle_message(message)

    def handle_observability_message(self, message: str | bytes) -> str:
        return self.endpoint.handle_observability_message(message)


__all__ = [
    "WorkerServiceLoopbackTransport",
    "WorkerServiceTransport",
]
