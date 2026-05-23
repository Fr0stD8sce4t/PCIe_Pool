from __future__ import annotations

from dataclasses import asdict, dataclass

from .codec import decode_worker_response_envelope, handle_worker_service_message
from .helper import WorkerTransferService


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


class WorkerServiceEndpoint:
    def __init__(
        self,
        daemon_client=None,
        service: WorkerTransferService | None = None,
    ) -> None:
        if service is None:
            if daemon_client is None:
                raise ValueError("daemon_client is required when service is not provided")
            service = WorkerTransferService(daemon_client)
        if not isinstance(service, WorkerTransferService):
            raise TypeError("service must be a WorkerTransferService")
        self.service = service
        self.events: list[WorkerEndpointEvent] = []
        self.last_event: WorkerEndpointEvent | None = None

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
        return {
            "total_requests": len(self.events),
            "last_event": (
                self.last_event.as_dict() if self.last_event is not None else None
            ),
            "final_state_counts": final_state_counts,
            "error_count": error_count,
            "completion_count": completion_count,
        }

    def clear_events(self) -> dict[str, object]:
        snapshot = self.describe()
        self.events.clear()
        self.last_event = None
        return snapshot


def _message_size(message: str | bytes) -> int:
    if isinstance(message, bytes):
        return len(message)
    if isinstance(message, str):
        return len(message.encode("utf-8"))
    raise TypeError("message must be str or bytes")


__all__ = ["WorkerEndpointEvent", "WorkerServiceEndpoint"]
