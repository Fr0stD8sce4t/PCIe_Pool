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


class WorkerCleanupError(RuntimeError):
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


@dataclass(frozen=True)
class WorkerTransferLifecycleRecord:
    authorization_request: WorkerTransferAuthorizationRequest
    worker_request: WorkerTransferRequest | None = None
    result: WorkerTransferResult | None = None
    status_update: Mapping[str, object] | None = None
    status_response: DaemonResponse | None = None
    cleanup_target_kind: str | None = None
    cleanup_target_id: str | None = None
    cleanup_response: DaemonResponse | None = None
    final_state: str = "created"
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.authorization_request, WorkerTransferAuthorizationRequest):
            raise TypeError("authorization_request must be a WorkerTransferAuthorizationRequest")
        if self.worker_request is not None and not isinstance(
            self.worker_request,
            WorkerTransferRequest,
        ):
            raise TypeError("worker_request must be a WorkerTransferRequest")
        if self.result is not None and not isinstance(self.result, WorkerTransferResult):
            raise TypeError("result must be a WorkerTransferResult")
        if self.status_update is not None and not isinstance(self.status_update, Mapping):
            raise TypeError("status_update must be a mapping")
        if self.status_response is not None and not isinstance(
            self.status_response,
            DaemonResponse,
        ):
            raise TypeError("status_response must be a DaemonResponse")
        if self.cleanup_response is not None and not isinstance(
            self.cleanup_response,
            DaemonResponse,
        ):
            raise TypeError("cleanup_response must be a DaemonResponse")
        final_state = str(self.final_state)
        if not final_state.strip():
            raise ValueError("final_state must be non-empty")
        object.__setattr__(self, "final_state", final_state)
        if self.cleanup_target_kind is not None:
            object.__setattr__(self, "cleanup_target_kind", str(self.cleanup_target_kind))
        if self.cleanup_target_id is not None:
            object.__setattr__(self, "cleanup_target_id", str(self.cleanup_target_id))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        if self.status_update is not None:
            object.__setattr__(self, "status_update", dict(self.status_update))

    def as_dict(self) -> dict[str, object]:
        cleanup_target = None
        if self.cleanup_target_kind is not None or self.cleanup_target_id is not None:
            cleanup_target = {
                "target_kind": self.cleanup_target_kind,
                "target_id": self.cleanup_target_id,
            }
        return {
            "authorization_request": asdict(self.authorization_request),
            "worker_request": (
                self.worker_request.as_dict()
                if self.worker_request is not None
                else None
            ),
            "result": self.result.as_dict() if self.result is not None else None,
            "status_update": (
                dict(self.status_update)
                if self.status_update is not None
                else None
            ),
            "status_response": (
                asdict(self.status_response)
                if self.status_response is not None
                else None
            ),
            "cleanup_target": cleanup_target,
            "cleanup_response": (
                asdict(self.cleanup_response)
                if self.cleanup_response is not None
                else None
            ),
            "final_state": self.final_state,
            "error": self.error,
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
        status_update = _daemon_status_update_for_result(result)
        response: DaemonResponse = self.daemon_client.transfer_status(
            status_update["transfer_id"],
            state=status_update["state"],
            bytes_completed=status_update["bytes_completed"],
            error=status_update["error"],
        )
        if not response.ok:
            raise WorkerStatusReportError(
                response.error or "worker transfer status report failed"
            )
        return response


class WorkerTransferCleanupCoordinator:
    def __init__(self, daemon_client) -> None:
        self.daemon_client = daemon_client

    def cleanup_authorization_failure(
        self,
        request: WorkerTransferAuthorizationRequest,
        target_kind: str = "reservation",
        reason: str = "worker_authorization_failed",
        force: bool = True,
    ) -> DaemonResponse:
        if not isinstance(request, WorkerTransferAuthorizationRequest):
            raise TypeError("request must be a WorkerTransferAuthorizationRequest")
        return self._cleanup(
            target_kind=target_kind,
            target_id=_cleanup_target_id(
                target_kind,
                lease_id=request.lease_id,
                session_id=request.session_id,
            ),
            reason=reason,
            force=force,
        )

    def cleanup_execution_failure(
        self,
        request: WorkerTransferRequest,
        result: WorkerTransferResult,
        target_kind: str = "reservation",
        reason: str | None = None,
        force: bool = True,
    ) -> DaemonResponse:
        if not isinstance(request, WorkerTransferRequest):
            raise TypeError("request must be a WorkerTransferRequest")
        if not isinstance(result, WorkerTransferResult):
            raise TypeError("result must be a WorkerTransferResult")
        if result.state == WorkerTransferState.COMPLETE:
            return DaemonResponse(
                ok=True,
                payload={
                    "cleanup_skipped": True,
                    "reason": "transfer_complete",
                },
            )
        return self._cleanup(
            target_kind=target_kind,
            target_id=_cleanup_target_id(
                target_kind,
                lease_id=request.authorization.lease_id,
                session_id=request.authorization.session_id,
            ),
            reason=reason or f"worker_{result.state.value}",
            force=force,
        )

    def _cleanup(
        self,
        target_kind: str,
        target_id: str,
        reason: str,
        force: bool,
    ) -> DaemonResponse:
        response: DaemonResponse = self.daemon_client.cleanup(
            target_kind=target_kind,
            target_id=target_id,
            reason=reason,
            force=force,
        )
        if not response.ok:
            raise WorkerCleanupError(response.error or "worker cleanup failed")
        return response


class WorkerTransferClient:
    def __init__(
        self,
        daemon_client,
        executor: WorkerTransferUnsupportedExecutor | None = None,
        status_reporter: WorkerTransferStatusReporter | None = None,
        cleanup_coordinator: WorkerTransferCleanupCoordinator | None = None,
    ) -> None:
        self.authorizer = WorkerTransferAuthorizer(daemon_client)
        self.executor = executor or WorkerTransferUnsupportedExecutor()
        self.status_reporter = status_reporter or WorkerTransferStatusReporter(
            daemon_client
        )
        self.cleanup_coordinator = cleanup_coordinator or WorkerTransferCleanupCoordinator(
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

    def submit_report_and_cleanup(
        self,
        request: WorkerTransferAuthorizationRequest,
        cleanup_target_kind: str = "reservation",
    ) -> WorkerTransferResult:
        lifecycle = self.submit_report_cleanup_lifecycle(
            request,
            cleanup_target_kind=cleanup_target_kind,
        )
        if lifecycle.final_state == "authorization_failed":
            raise WorkerAuthorizationError(
                lifecycle.error or "worker transfer authorization failed"
            )
        if lifecycle.final_state == "status_failed":
            raise WorkerStatusReportError(
                lifecycle.error or "worker transfer status report failed"
            )
        if lifecycle.final_state == "cleanup_failed":
            raise WorkerCleanupError(lifecycle.error or "worker cleanup failed")
        if lifecycle.result is None:
            raise RuntimeError("worker lifecycle completed without a result")
        return lifecycle.result

    def submit_report_cleanup_lifecycle(
        self,
        request: WorkerTransferAuthorizationRequest,
        cleanup_target_kind: str = "reservation",
    ) -> WorkerTransferLifecycleRecord:
        try:
            worker_request = self.authorize(request)
        except WorkerAuthorizationError as exc:
            cleanup_target_id = _cleanup_target_id(
                cleanup_target_kind,
                lease_id=request.lease_id,
                session_id=request.session_id,
            )
            try:
                cleanup_response = self.cleanup_coordinator.cleanup_authorization_failure(
                    request,
                    target_kind=cleanup_target_kind,
                )
            except WorkerCleanupError as cleanup_exc:
                return WorkerTransferLifecycleRecord(
                    authorization_request=request,
                    cleanup_target_kind=cleanup_target_kind,
                    cleanup_target_id=cleanup_target_id,
                    final_state="cleanup_failed",
                    error=str(cleanup_exc),
                )
            return WorkerTransferLifecycleRecord(
                authorization_request=request,
                cleanup_target_kind=cleanup_target_kind,
                cleanup_target_id=cleanup_target_id,
                cleanup_response=cleanup_response,
                final_state="authorization_failed",
                error=str(exc),
            )
        result = self.executor.execute(worker_request)
        status_update = _daemon_status_update_for_result(result)
        try:
            status_response = self.status_reporter.report(result)
        except WorkerStatusReportError as exc:
            return WorkerTransferLifecycleRecord(
                authorization_request=request,
                worker_request=worker_request,
                result=result,
                status_update=status_update,
                final_state="status_failed",
                error=str(exc),
            )
        cleanup_target_id = None
        if result.state != WorkerTransferState.COMPLETE:
            cleanup_target_id = _cleanup_target_id(
                cleanup_target_kind,
                lease_id=worker_request.authorization.lease_id,
                session_id=worker_request.authorization.session_id,
            )
        try:
            cleanup_response = self.cleanup_coordinator.cleanup_execution_failure(
                worker_request,
                result,
                target_kind=cleanup_target_kind,
            )
        except WorkerCleanupError as exc:
            return WorkerTransferLifecycleRecord(
                authorization_request=request,
                worker_request=worker_request,
                result=result,
                status_update=status_update,
                status_response=status_response,
                cleanup_target_kind=cleanup_target_kind,
                cleanup_target_id=cleanup_target_id,
                final_state="cleanup_failed",
                error=str(exc),
            )
        return WorkerTransferLifecycleRecord(
            authorization_request=request,
            worker_request=worker_request,
            result=result,
            status_update=status_update,
            status_response=status_response,
            cleanup_target_kind=cleanup_target_kind,
            cleanup_target_id=cleanup_target_id,
            cleanup_response=cleanup_response,
            final_state=result.state.value,
            error=result.error,
        )


class WorkerTransferService:
    def __init__(
        self,
        daemon_client,
        transfer_client: WorkerTransferClient | None = None,
    ) -> None:
        self.transfer_client = transfer_client or WorkerTransferClient(daemon_client)

    def handle_lifecycle(
        self,
        request: WorkerTransferAuthorizationRequest,
        cleanup_target_kind: str = "reservation",
    ) -> WorkerTransferLifecycleRecord:
        if not isinstance(request, WorkerTransferAuthorizationRequest):
            raise TypeError("request must be a WorkerTransferAuthorizationRequest")
        return self.transfer_client.submit_report_cleanup_lifecycle(
            request,
            cleanup_target_kind=cleanup_target_kind,
        )

    def handle(
        self,
        request: WorkerTransferAuthorizationRequest,
        cleanup_target_kind: str = "reservation",
    ) -> dict[str, object]:
        return self.handle_lifecycle(
            request,
            cleanup_target_kind=cleanup_target_kind,
        ).as_dict()


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


def _daemon_status_update_for_result(result: WorkerTransferResult) -> dict[str, object]:
    daemon_state = _daemon_state_for_worker_state(result.state)
    error = result.error
    if result.state == WorkerTransferState.UNSUPPORTED and error is None:
        error = "worker execution is unsupported"
    if result.state == WorkerTransferState.FAILED and error is None:
        error = "worker transfer failed"
    return {
        "transfer_id": result.transfer_id,
        "state": daemon_state.value,
        "bytes_completed": result.bytes_completed,
        "error": error,
    }


def _cleanup_target_id(target_kind: str, lease_id: str, session_id: str) -> str:
    normalized = str(target_kind)
    if normalized == "reservation":
        return str(lease_id)
    if normalized == "session":
        return str(session_id)
    raise ValueError("worker cleanup target must be reservation or session")
