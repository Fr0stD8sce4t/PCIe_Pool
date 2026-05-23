from __future__ import annotations

from dataclasses import asdict, dataclass

from .codec import (
    decode_worker_response_envelope,
    decode_worker_observability_request_envelope,
    encode_worker_observability_snapshot,
    handle_worker_service_message,
)
from .helper import WorkerTransferService


_DEGRADED_FINAL_STATES = frozenset(
    {
        "authorization_failed",
        "cleanup_failed",
        "parse_failed",
        "status_failed",
    }
)


@dataclass(frozen=True)
class WorkerEndpointEvent:
    request_bytes: int
    response_bytes: int
    ok: bool
    final_state: str | None = None
    error: str | None = None
    has_completion: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerEndpointObservabilityEvent:
    request_bytes: int
    response_bytes: int
    request_type: str = "observability"
    response_type: str = "snapshot"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class WorkerServiceEndpoint:
    def __init__(
        self,
        daemon_client=None,
        service: WorkerTransferService | None = None,
        max_events: int | None = None,
    ) -> None:
        if service is None:
            if daemon_client is None:
                raise ValueError("daemon_client is required when service is not provided")
            service = WorkerTransferService(daemon_client)
        if not isinstance(service, WorkerTransferService):
            raise TypeError("service must be a WorkerTransferService")
        if max_events is not None:
            max_events = int(max_events)
            if max_events <= 0:
                raise ValueError("max_events must be positive")
        self.service = service
        self.max_events = max_events
        self.events: list[WorkerEndpointEvent] = []
        self.observability_events: list[WorkerEndpointObservabilityEvent] = []
        self.last_event: WorkerEndpointEvent | None = None
        self.last_observability_event: WorkerEndpointObservabilityEvent | None = None

    def handle_message(self, message: str | bytes) -> str:
        request_bytes = _message_size(message)
        response_message = handle_worker_service_message(self.service, message)
        response = decode_worker_response_envelope(response_message)
        event = WorkerEndpointEvent(
            request_bytes=request_bytes,
            response_bytes=_message_size(response_message),
            ok=response.ok,
            final_state=response.final_state,
            error=response.error,
            has_completion=response.completion is not None,
        )
        self.last_event = event
        self.events.append(event)
        self._trim_events()
        return response_message

    def handle_observability_message(self, message: str | bytes) -> str:
        request_bytes = _message_size(message)
        request = decode_worker_observability_request_envelope(message)
        response_message = encode_worker_observability_snapshot(
            self.observability_snapshot()
        )
        event = WorkerEndpointObservabilityEvent(
            request_bytes=request_bytes,
            response_bytes=_message_size(response_message),
            request_type=request.request_type,
        )
        self.last_observability_event = event
        self.observability_events.append(event)
        self._trim_observability_events()
        return response_message

    def describe(self) -> dict[str, object]:
        final_state_counts: dict[str, int] = {}
        error_count = 0
        completion_count = 0
        for event in self.events:
            final_state = event.final_state or "unknown"
            final_state_counts[final_state] = final_state_counts.get(final_state, 0) + 1
            if event.error is not None:
                error_count += 1
            if event.has_completion:
                completion_count += 1
        retained_event_count = len(self.events)
        observability_retained_event_count = len(self.observability_events)
        return {
            "total_requests": retained_event_count,
            "retained_event_count": retained_event_count,
            "max_events": self.max_events,
            "history_bounded": self.max_events is not None,
            "last_event": (
                self.last_event.as_dict() if self.last_event is not None else None
            ),
            "events": self.event_snapshot(),
            "health": self.health_snapshot(),
            "metrics": self.metrics_snapshot(),
            "observability_total_requests": observability_retained_event_count,
            "observability_retained_event_count": observability_retained_event_count,
            "observability_last_event": (
                self.last_observability_event.as_dict()
                if self.last_observability_event is not None
                else None
            ),
            "observability_events": self.observability_event_snapshot(),
            "final_state_counts": final_state_counts,
            "error_count": error_count,
            "completion_count": completion_count,
        }

    def clear_events(self) -> dict[str, object]:
        snapshot = self.describe()
        self.events.clear()
        self.observability_events.clear()
        self.last_event = None
        self.last_observability_event = None
        return snapshot

    def event_snapshot(self) -> tuple[dict[str, object], ...]:
        return tuple(event.as_dict() for event in self.events)

    def observability_event_snapshot(self) -> tuple[dict[str, object], ...]:
        return tuple(event.as_dict() for event in self.observability_events)

    def health_snapshot(self) -> dict[str, object]:
        degraded_final_states: set[str] = set()
        degraded_event_count = 0
        for event in self.events:
            final_state = event.final_state or "unknown"
            if not event.ok or final_state in _DEGRADED_FINAL_STATES:
                degraded_event_count += 1
                degraded_final_states.add(final_state)
        ready = degraded_event_count == 0
        return {
            "status": "ready" if ready else "degraded",
            "ready": ready,
            "retained_event_count": len(self.events),
            "degraded_event_count": degraded_event_count,
            "degraded_final_states": tuple(sorted(degraded_final_states)),
            "last_final_state": (
                self.last_event.final_state if self.last_event is not None else None
            ),
            "last_ok": self.last_event.ok if self.last_event is not None else None,
        }

    def metrics_snapshot(self) -> dict[str, object]:
        retained_event_count = len(self.events)
        request_bytes_total = sum(event.request_bytes for event in self.events)
        response_bytes_total = sum(event.response_bytes for event in self.events)
        if retained_event_count == 0:
            average_request_bytes = None
            average_response_bytes = None
        else:
            average_request_bytes = request_bytes_total / retained_event_count
            average_response_bytes = response_bytes_total / retained_event_count
        return {
            "retained_event_count": retained_event_count,
            "request_bytes_total": request_bytes_total,
            "response_bytes_total": response_bytes_total,
            "average_request_bytes": average_request_bytes,
            "average_response_bytes": average_response_bytes,
            "last_request_bytes": (
                self.last_event.request_bytes if self.last_event is not None else None
            ),
            "last_response_bytes": (
                self.last_event.response_bytes if self.last_event is not None else None
            ),
        }

    def _trim_observability_events(self) -> None:
        if self.max_events is None:
            return
        extra_count = len(self.observability_events) - self.max_events
        if extra_count > 0:
            del self.observability_events[:extra_count]

    def observability_snapshot(self) -> dict[str, object]:
        description = self.describe()
        return {
            "describe": description,
            "events": description["events"],
            "health": description["health"],
            "metrics": description["metrics"],
            "observability_events": description["observability_events"],
        }

    def _trim_events(self) -> None:
        if self.max_events is None:
            return
        extra_count = len(self.events) - self.max_events
        if extra_count > 0:
            del self.events[:extra_count]


def _message_size(message: str | bytes) -> int:
    if isinstance(message, bytes):
        return len(message)
    if isinstance(message, str):
        return len(message.encode("utf-8"))
    raise TypeError("message must be str or bytes")


__all__ = [
    "WorkerEndpointEvent",
    "WorkerEndpointObservabilityEvent",
    "WorkerServiceEndpoint",
]
