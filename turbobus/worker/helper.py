from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping

from ..schema import (
    BufferRegistration,
    DaemonResponse,
    TransferStatusState,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)


class WorkerTransferState(str, Enum):
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    COMPLETE = "complete"


class UnsupportedWorkerExecution(RuntimeError):
    pass


class WorkerAuthorizationError(RuntimeError):
    pass


class WorkerStatusReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerTransferRequest:
    authorization: WorkerTransferAuthorization

    @classmethod
    def from_authorization_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "WorkerTransferRequest":
        authorization_payload = payload.get("authorization", payload)
        if not isinstance(authorization_payload, Mapping):
            raise ValueError("authorization payload must be a mapping")
        return cls(
            authorization=WorkerTransferAuthorization(
                transfer_id=str(authorization_payload["transfer_id"]),
                lease_id=str(authorization_payload["lease_id"]),
                session_id=str(authorization_payload["session_id"]),
                job_id=str(authorization_payload["job_id"]),
                src_buffer=_buffer_from_payload(authorization_payload["src_buffer"]),
                dst_buffer=_buffer_from_payload(authorization_payload["dst_buffer"]),
                direction=str(authorization_payload["direction"]),
                ranges=tuple(authorization_payload.get("ranges", ())),
                relay_gpu=authorization_payload.get("relay_gpu"),
            )
        )

    @property
    def transfer_id(self) -> str:
        return self.authorization.transfer_id

    def as_dict(self) -> dict[str, object]:
        return {"authorization": asdict(self.authorization)}


@dataclass(frozen=True)
class WorkerTransferResult:
    transfer_id: str
    state: WorkerTransferState
    error: str | None = None
    bytes_completed: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        bytes_completed = int(self.bytes_completed)
        if bytes_completed < 0:
            raise ValueError("bytes_completed must be non-negative")
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "state", WorkerTransferState(self.state))
        object.__setattr__(self, "bytes_completed", bytes_completed)
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return {
            "transfer_id": self.transfer_id,
            "state": self.state.value,
            "error": self.error,
            "bytes_completed": self.bytes_completed,
            "metadata": dict(self.metadata),
        }


class WorkerTransferUnsupportedExecutor:
    def execute(self, request: WorkerTransferRequest) -> WorkerTransferResult:
        if not isinstance(request, WorkerTransferRequest):
            raise TypeError("request must be a WorkerTransferRequest")
        return WorkerTransferResult(
            transfer_id=request.transfer_id,
            state=WorkerTransferState.UNSUPPORTED,
            error="worker execution is not implemented yet",
            bytes_completed=0,
            metadata={
                "relay_gpu": request.authorization.relay_gpu,
                "src_buffer_id": request.authorization.src_buffer.buffer_id,
                "dst_buffer_id": request.authorization.dst_buffer.buffer_id,
            },
        )

    def execute_or_raise(self, request: WorkerTransferRequest) -> WorkerTransferResult:
        result = self.execute(request)
        raise UnsupportedWorkerExecution(result.error or "worker execution is unsupported")


class WorkerTransferAuthorizer:
    def __init__(self, daemon_client) -> None:
        self.daemon_client = daemon_client

    def authorize(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> WorkerTransferRequest:
        response: DaemonResponse = self.daemon_client.authorize_worker_transfer(request)
        if not response.ok:
            raise WorkerAuthorizationError(
                response.error or "worker transfer authorization failed"
            )
        try:
            return WorkerTransferRequest.from_authorization_payload(response.payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkerAuthorizationError(
                f"invalid worker authorization response: {exc}"
            ) from exc


class WorkerTransferStatusReporter:
    def __init__(self, daemon_client) -> None:
        self.daemon_client = daemon_client

    def report(self, result: WorkerTransferResult) -> DaemonResponse:
        if not isinstance(result, WorkerTransferResult):
            raise TypeError("result must be a WorkerTransferResult")
        daemon_state = _daemon_state_for_worker_state(result.state)
        error = result.error
        if result.state == WorkerTransferState.UNSUPPORTED and error is None:
            error = "worker execution is unsupported"
        if result.state == WorkerTransferState.FAILED and error is None:
            error = "worker transfer failed"
        response: DaemonResponse = self.daemon_client.transfer_status(
            result.transfer_id,
            state=daemon_state.value,
            bytes_completed=result.bytes_completed,
            error=error,
        )
        if not response.ok:
            raise WorkerStatusReportError(
                response.error or "worker transfer status report failed"
            )
        return response


class WorkerTransferClient:
    def __init__(
        self,
        daemon_client,
        executor: WorkerTransferUnsupportedExecutor | None = None,
        status_reporter: WorkerTransferStatusReporter | None = None,
    ) -> None:
        self.authorizer = WorkerTransferAuthorizer(daemon_client)
        self.executor = executor or WorkerTransferUnsupportedExecutor()
        self.status_reporter = status_reporter or WorkerTransferStatusReporter(
            daemon_client
        )

    def authorize(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> WorkerTransferRequest:
        return self.authorizer.authorize(request)

    def submit(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> WorkerTransferResult:
        worker_request = self.authorize(request)
        return self.executor.execute(worker_request)

    def submit_and_report(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> WorkerTransferResult:
        result = self.submit(request)
        self.status_reporter.report(result)
        return result


def _buffer_from_payload(payload: object) -> BufferRegistration:
    if not isinstance(payload, Mapping):
        raise ValueError("buffer payload must be a mapping")
    return BufferRegistration(
        buffer_id=str(payload["buffer_id"]),
        job_id=str(payload["job_id"]),
        kind=str(payload["kind"]),
        size_bytes=int(payload["size_bytes"]),
        device_index=payload.get("device_index"),
        address=payload.get("address"),
        pinned=bool(payload.get("pinned", False)),
    )


def _daemon_state_for_worker_state(
    state: WorkerTransferState,
) -> TransferStatusState:
    worker_state = WorkerTransferState(state)
    if worker_state == WorkerTransferState.COMPLETE:
        return TransferStatusState.COMPLETE
    return TransferStatusState.FAILED
