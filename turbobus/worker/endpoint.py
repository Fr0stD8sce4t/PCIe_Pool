from __future__ import annotations

from .codec import handle_worker_service_message
from .helper import WorkerTransferService


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

    def handle_message(self, message: str | bytes) -> str:
        return handle_worker_service_message(self.service, message)


__all__ = [
    "WorkerServiceEndpoint",
]
