from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping

from ..schema import (
    BufferRegistration,
    DaemonResponse,
    ExecutionTicket,
    SchedulingDecision,
    TransferStatusState,
    WorkerDataPlaneCompletion,
    WorkerDataPlaneRequest,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)
from .resources import (
    WorkerDataPlaneResourceBinder,
    WorkerDataPlaneResourceError,
    WorkerDataPlaneResources,
)
from .staging_pool import WorkerStagingPool, WorkerStagingSlot


class WorkerTransferState(str, Enum):
    FAILED = "failed"
    COMPLETE = "complete"


class WorkerAuthorizationError(RuntimeError):
    pass


class WorkerStatusReportError(RuntimeError):
    pass


class WorkerCleanupError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerTransferRequest:
    authorization: WorkerTransferAuthorization
    ticket: ExecutionTicket
    data_plane: WorkerDataPlaneRequest | None = None

    @classmethod
    def from_authorization_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "WorkerTransferRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("authorization payload must be a mapping")
        if payload.get("ticket") is None:
            raise ValueError(
                "daemon worker authorization response must include execution ticket"
            )
        return cls.from_execution_ticket_payload(payload)

    @classmethod
    def from_execution_ticket_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "WorkerTransferRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("execution ticket payload must be a mapping")
        ticket_payload = payload.get("ticket")
        if not isinstance(ticket_payload, Mapping):
            raise ValueError("execution ticket payload must include ticket")
        decision_payload = payload.get("decision")
        return cls.from_execution_ticket(
            ExecutionTicket(**dict(ticket_payload)),
            src_buffer=_buffer_from_payload(payload["src_buffer"]),
            dst_buffer=_buffer_from_payload(payload["dst_buffer"]),
            relay_gpu=payload.get("relay_gpu"),
            lease_id=payload.get("lease_id"),
            transfer_id=payload.get("transfer_id"),
            decision=(
                None
                if decision_payload is None
                else SchedulingDecision(**dict(decision_payload))
            ),
            plan_generation=payload.get("plan_generation"),
        )

    @classmethod
    def from_authorization(
        cls,
        authorization: WorkerTransferAuthorization,
    ) -> "WorkerTransferRequest":
        raise ValueError(
            "worker transfer requests must be built from daemon-issued "
            "ExecutionTicket objects"
        )

    @classmethod
    def from_execution_ticket(
        cls,
        ticket: ExecutionTicket,
        *,
        src_buffer: BufferRegistration,
        dst_buffer: BufferRegistration,
        relay_gpu: int | None = None,
        lease_id: str | None = None,
        transfer_id: str | None = None,
        decision: SchedulingDecision | None = None,
        plan_generation: object | None = None,
    ) -> "WorkerTransferRequest":
        _validate_ticket_matches_buffers(ticket, src_buffer, dst_buffer)
        _validate_daemon_issued_ticket(ticket, plan_generation=plan_generation)
        if decision is not None:
            _validate_ticket_matches_decision(ticket, decision)
        relay = _relay_gpu_for_ticket(ticket, relay_gpu)
        ranges = _relay_ranges_from_ticket_plan(ticket, relay_gpu=relay)
        resolved_lease_id = _lease_id_for_ticket(ticket, lease_id)
        resolved_transfer_id = _transfer_id_for_ticket(ticket, transfer_id)
        authorization = WorkerTransferAuthorization(
            transfer_id=resolved_transfer_id,
            lease_id=resolved_lease_id,
            session_id=ticket.session_id,
            job_id=ticket.job_id,
            src_buffer=src_buffer,
            dst_buffer=dst_buffer,
            direction=ticket.direction,
            ranges=ranges,
            relay_gpu=relay,
            plan=dict(ticket.plan),
        )
        return cls(
            authorization=authorization,
            data_plane=WorkerDataPlaneRequest.from_authorization(authorization),
            ticket=ticket,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.authorization, WorkerTransferAuthorization):
            raise TypeError("authorization must be a WorkerTransferAuthorization")
        data_plane = self.data_plane
        if data_plane is None:
            data_plane = WorkerDataPlaneRequest.from_authorization(self.authorization)
        if not isinstance(data_plane, WorkerDataPlaneRequest):
            raise TypeError("data_plane must be a WorkerDataPlaneRequest")
        if data_plane.transfer_id != self.authorization.transfer_id:
            raise ValueError("data-plane transfer id does not match authorization")
        if data_plane.lease_id != self.authorization.lease_id:
            raise ValueError("data-plane lease id does not match authorization")
        if data_plane.session_id != self.authorization.session_id:
            raise ValueError("data-plane session id does not match authorization")
        if data_plane.job_id != self.authorization.job_id:
            raise ValueError("data-plane job id does not match authorization")
        if data_plane.relay_gpu != self.authorization.relay_gpu:
            raise ValueError("data-plane relay does not match authorization")
        if data_plane.direction != self.authorization.direction:
            raise ValueError("data-plane direction does not match authorization")
        if data_plane.src_handle.buffer_id != self.authorization.src_buffer.buffer_id:
            raise ValueError("data-plane src handle does not match authorization")
        if data_plane.dst_handle.buffer_id != self.authorization.dst_buffer.buffer_id:
            raise ValueError("data-plane dst handle does not match authorization")
        if data_plane.ranges != self.authorization.ranges:
            raise ValueError("data-plane ranges do not match authorization")
        if data_plane.plan != self.authorization.plan:
            raise ValueError("data-plane plan does not match authorization")
        if not isinstance(self.ticket, ExecutionTicket):
            raise TypeError("ticket must be an ExecutionTicket")
        _validate_daemon_issued_ticket(self.ticket)
        _validate_ticket_matches_worker_request(self.ticket, self.authorization)
        object.__setattr__(self, "data_plane", data_plane)

    @property
    def transfer_id(self) -> str:
        return self.authorization.transfer_id

    def as_dict(self) -> dict[str, object]:
        return {
            "authorization": asdict(self.authorization),
            "data_plane": asdict(self.data_plane),
            "ticket": asdict(self.ticket),
        }


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

    def data_plane_completion(self, lease_id: str) -> WorkerDataPlaneCompletion:
        status_update = _daemon_status_update_for_result(self)
        return WorkerDataPlaneCompletion(
            transfer_id=self.transfer_id,
            lease_id=lease_id,
            state=status_update["state"],
            bytes_completed=self.bytes_completed,
            error=(
                status_update["error"]
                if status_update["state"] == TransferStatusState.FAILED.value
                else None
            ),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class WorkerTransferLifecycleRecord:
    authorization_request: WorkerTransferAuthorizationRequest
    worker_request: WorkerTransferRequest | None = None
    staging_slot: WorkerStagingSlot | None = None
    staging_release: WorkerStagingSlot | None = None
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
        if self.staging_slot is not None and not isinstance(
            self.staging_slot,
            WorkerStagingSlot,
        ):
            raise TypeError("staging_slot must be a WorkerStagingSlot")
        if self.staging_release is not None and not isinstance(
            self.staging_release,
            WorkerStagingSlot,
        ):
            raise TypeError("staging_release must be a WorkerStagingSlot")
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
            "staging_slot": (
                self.staging_slot.as_dict()
                if self.staging_slot is not None
                else None
            ),
            "staging_release": (
                self.staging_release.as_dict()
                if self.staging_release is not None
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

    def completion_envelope(self) -> "WorkerDataPlaneCompletionEnvelope":
        return WorkerDataPlaneCompletionEnvelope.from_lifecycle(self)


@dataclass(frozen=True)
class WorkerDataPlaneCompletionEnvelope:
    ok: bool
    transfer_id: str | None = None
    lease_id: str | None = None
    final_state: str | None = None
    staging_slot: Mapping[str, object] | None = None
    worker_result: Mapping[str, object] | None = None
    daemon_status_update: Mapping[str, object] | None = None
    daemon_status_response: Mapping[str, object] | None = None
    daemon_cleanup_response: Mapping[str, object] | None = None
    staging_release: Mapping[str, object] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ok", bool(self.ok))
        for field_name in (
            "staging_slot",
            "worker_result",
            "daemon_status_update",
            "daemon_status_response",
            "daemon_cleanup_response",
            "staging_release",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, dict(value))
        if self.transfer_id is not None:
            object.__setattr__(self, "transfer_id", str(self.transfer_id))
        if self.lease_id is not None:
            object.__setattr__(self, "lease_id", str(self.lease_id))
        if self.final_state is not None:
            object.__setattr__(self, "final_state", str(self.final_state))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))

    @classmethod
    def from_lifecycle(
        cls,
        lifecycle: WorkerTransferLifecycleRecord,
    ) -> "WorkerDataPlaneCompletionEnvelope":
        if not isinstance(lifecycle, WorkerTransferLifecycleRecord):
            raise TypeError("lifecycle must be a WorkerTransferLifecycleRecord")
        payload = lifecycle.as_dict()
        return cls(
            ok=True,
            transfer_id=_lifecycle_transfer_id(lifecycle),
            lease_id=_lifecycle_lease_id(lifecycle),
            final_state=lifecycle.final_state,
            staging_slot=payload["staging_slot"],
            worker_result=payload["result"],
            daemon_status_update=payload["status_update"],
            daemon_status_response=payload["status_response"],
            daemon_cleanup_response=payload["cleanup_response"],
            staging_release=payload["staging_release"],
            error=lifecycle.error,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "transfer_id": self.transfer_id,
            "lease_id": self.lease_id,
            "final_state": self.final_state,
            "staging_slot": (
                dict(self.staging_slot) if self.staging_slot is not None else None
            ),
            "worker_result": (
                dict(self.worker_result) if self.worker_result is not None else None
            ),
            "daemon_status_update": (
                dict(self.daemon_status_update)
                if self.daemon_status_update is not None
                else None
            ),
            "daemon_status_response": (
                dict(self.daemon_status_response)
                if self.daemon_status_response is not None
                else None
            ),
            "daemon_cleanup_response": (
                dict(self.daemon_cleanup_response)
                if self.daemon_cleanup_response is not None
                else None
            ),
            "staging_release": (
                dict(self.staging_release)
                if self.staging_release is not None
                else None
            ),
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkerServiceRequestEnvelope:
    payload: Mapping[str, object]
    cleanup_target_kind: str = "reservation"

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("worker service payload must be a mapping")
        cleanup_target_kind = str(self.cleanup_target_kind)
        if cleanup_target_kind not in {"reservation", "session"}:
            raise ValueError("cleanup_target_kind must be reservation or session")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "cleanup_target_kind", cleanup_target_kind)

    def as_dict(self) -> dict[str, object]:
        return {
            "payload": dict(self.payload),
            "cleanup_target_kind": self.cleanup_target_kind,
        }


@dataclass(frozen=True)
class WorkerServiceResponseEnvelope:
    ok: bool
    completion: Mapping[str, object] | None = None
    error: str | None = None
    final_state: str | None = None

    def __post_init__(self) -> None:
        if self.completion is not None and not isinstance(self.completion, Mapping):
            raise TypeError("completion must be a mapping")
        object.__setattr__(self, "ok", bool(self.ok))
        if self.completion is not None:
            object.__setattr__(self, "completion", dict(self.completion))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        if self.final_state is not None:
            object.__setattr__(self, "final_state", str(self.final_state))

    @classmethod
    def from_lifecycle(
        cls,
        lifecycle: WorkerTransferLifecycleRecord,
    ) -> "WorkerServiceResponseEnvelope":
        return cls(
            ok=True,
            completion=lifecycle.completion_envelope().as_dict(),
            final_state=lifecycle.final_state,
            error=lifecycle.error,
        )

    @classmethod
    def from_error(cls, error: str) -> "WorkerServiceResponseEnvelope":
        return cls(ok=False, error=str(error), final_state="parse_failed")

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "completion": (
                dict(self.completion) if self.completion is not None else None
            ),
            "error": self.error,
            "final_state": self.final_state,
        }


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
            worker_request = WorkerTransferRequest.from_authorization_payload(
                response.payload
            )
            _require_daemon_worker_plan(worker_request)
            return worker_request
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
            release = getattr(self.daemon_client, "release_transfer", None)
            if not callable(release):
                raise WorkerCleanupError(
                    "daemon client cannot release completed worker transfer"
                )
            response: DaemonResponse = release(request.authorization.lease_id)
            if not response.ok:
                raise WorkerCleanupError(
                    response.error or "worker completion release failed"
                )
            return response
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

    def cleanup_status_report_failure(
        self,
        request: WorkerTransferRequest,
        target_kind: str = "reservation",
        reason: str = "worker_status_report_failed",
        force: bool = True,
    ) -> DaemonResponse:
        if not isinstance(request, WorkerTransferRequest):
            raise TypeError("request must be a WorkerTransferRequest")
        return self._cleanup(
            target_kind=target_kind,
            target_id=_cleanup_target_id(
                target_kind,
                lease_id=request.authorization.lease_id,
                session_id=request.authorization.session_id,
            ),
            reason=reason,
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
        executor: object | None = None,
        status_reporter: WorkerTransferStatusReporter | None = None,
        cleanup_coordinator: WorkerTransferCleanupCoordinator | None = None,
        staging_pool: WorkerStagingPool | None = None,
        resource_binder: WorkerDataPlaneResourceBinder | None = None,
    ) -> None:
        if executor is None:
            executor = _default_worker_executor()
            if resource_binder is None:
                resource_binder = WorkerDataPlaneResourceBinder()
        self.authorizer = WorkerTransferAuthorizer(daemon_client)
        self.executor = executor
        self.status_reporter = status_reporter or WorkerTransferStatusReporter(
            daemon_client
        )
        self.cleanup_coordinator = cleanup_coordinator or WorkerTransferCleanupCoordinator(
            daemon_client
        )
        self.staging_pool = staging_pool or WorkerStagingPool()
        self.resource_binder = resource_binder

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
        staging_slot = self.staging_pool.allocate(worker_request.data_plane)
        try:
            try:
                result = self._execute(worker_request, staging_slot)
            except Exception as exc:
                result = _failed_worker_result_from_exception(
                    worker_request,
                    staging_slot,
                    exc,
                )
            return _validate_worker_completion_bytes(worker_request, result)
        finally:
            self.staging_pool.release(staging_slot.slot_id, worker_request.data_plane)

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
        staging_slot = self.staging_pool.allocate(worker_request.data_plane)
        try:
            result = _validate_worker_completion_bytes(
                worker_request,
                self._execute(worker_request, staging_slot),
            )
        except Exception as exc:
            result = _failed_worker_result_from_exception(
                worker_request,
                staging_slot,
                exc,
            )
        status_update = _daemon_status_update_for_result(result)
        try:
            status_response = self.status_reporter.report(result)
        except WorkerStatusReportError as exc:
            staging_release = self.staging_pool.release(
                staging_slot.slot_id,
                worker_request.data_plane,
            )
            cleanup_target_id = _cleanup_target_id(
                cleanup_target_kind,
                lease_id=worker_request.authorization.lease_id,
                session_id=worker_request.authorization.session_id,
            )
            try:
                cleanup_response = (
                    self.cleanup_coordinator.cleanup_status_report_failure(
                        worker_request,
                        target_kind=cleanup_target_kind,
                    )
                )
            except WorkerCleanupError as cleanup_exc:
                return WorkerTransferLifecycleRecord(
                    authorization_request=request,
                    worker_request=worker_request,
                    staging_slot=staging_slot,
                    staging_release=staging_release,
                    result=result,
                    status_update=status_update,
                    cleanup_target_kind=cleanup_target_kind,
                    cleanup_target_id=cleanup_target_id,
                    final_state="cleanup_failed",
                    error=str(cleanup_exc),
                )
            return WorkerTransferLifecycleRecord(
                authorization_request=request,
                worker_request=worker_request,
                staging_slot=staging_slot,
                staging_release=staging_release,
                result=result,
                status_update=status_update,
                cleanup_target_kind=cleanup_target_kind,
                cleanup_target_id=cleanup_target_id,
                cleanup_response=cleanup_response,
                final_state="status_failed",
                error=str(exc),
            )
        cleanup_target_id = (
            worker_request.authorization.lease_id
            if result.state == WorkerTransferState.COMPLETE
            else _cleanup_target_id(
                cleanup_target_kind,
                lease_id=worker_request.authorization.lease_id,
                session_id=worker_request.authorization.session_id,
            )
        )
        try:
            cleanup_response = self.cleanup_coordinator.cleanup_execution_failure(
                worker_request,
                result,
                target_kind=cleanup_target_kind,
            )
        except WorkerCleanupError as exc:
            staging_release = self.staging_pool.release(
                staging_slot.slot_id,
                worker_request.data_plane,
            )
            return WorkerTransferLifecycleRecord(
                authorization_request=request,
                worker_request=worker_request,
                staging_slot=staging_slot,
                staging_release=staging_release,
                result=result,
                status_update=status_update,
                status_response=status_response,
                cleanup_target_kind=cleanup_target_kind,
                cleanup_target_id=cleanup_target_id,
                final_state="cleanup_failed",
                error=str(exc),
            )
        staging_release = self.staging_pool.release(
            staging_slot.slot_id,
            worker_request.data_plane,
        )
        return WorkerTransferLifecycleRecord(
            authorization_request=request,
            worker_request=worker_request,
            staging_slot=staging_slot,
            staging_release=staging_release,
            result=result,
            status_update=status_update,
            status_response=status_response,
            cleanup_target_kind=cleanup_target_kind,
            cleanup_target_id=cleanup_target_id,
            cleanup_response=cleanup_response,
            final_state=result.state.value,
            error=result.error,
        )

    def _execute(
        self,
        worker_request: WorkerTransferRequest,
        staging_slot: WorkerStagingSlot,
    ) -> WorkerTransferResult:
        if self.resource_binder is None:
            return self.executor.execute(worker_request, staging_slot)
        with self.resource_binder.bind(worker_request.data_plane) as resources:
            return _execute_worker_transfer(
                self.executor,
                worker_request,
                staging_slot,
                resources,
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

    def parse_authorization_request(
        self,
        payload: Mapping[str, object],
    ) -> WorkerTransferAuthorizationRequest:
        return parse_worker_authorization_request_payload(payload)

    def handle_envelope(
        self,
        envelope: WorkerServiceRequestEnvelope | Mapping[str, object],
    ) -> WorkerServiceResponseEnvelope:
        try:
            request_envelope = (
                envelope
                if isinstance(envelope, WorkerServiceRequestEnvelope)
                else WorkerServiceRequestEnvelope(
                    payload=envelope.get("payload", envelope),
                    cleanup_target_kind=str(
                        envelope.get("cleanup_target_kind", "reservation")
                    ),
                )
            )
            lifecycle = self.handle_lifecycle(
                self.parse_authorization_request(request_envelope.payload),
                cleanup_target_kind=request_envelope.cleanup_target_kind,
            )
            return WorkerServiceResponseEnvelope.from_lifecycle(lifecycle)
        except (KeyError, TypeError, ValueError) as exc:
            return WorkerServiceResponseEnvelope.from_error(str(exc))

    def handle_envelope_payload(
        self,
        envelope: WorkerServiceRequestEnvelope | Mapping[str, object],
    ) -> dict[str, object]:
        return self.handle_envelope(envelope).as_dict()


def parse_worker_authorization_request_payload(
    payload: Mapping[str, object],
) -> WorkerTransferAuthorizationRequest:
    if not isinstance(payload, Mapping):
        raise ValueError("worker authorization payload must be a mapping")
    authorization_payload = payload.get("authorization_request", payload)
    if not isinstance(authorization_payload, Mapping):
        raise ValueError("worker authorization payload must be a mapping")
    try:
        return WorkerTransferAuthorizationRequest(
            transfer_id=str(authorization_payload["transfer_id"]),
            lease_id=str(authorization_payload["lease_id"]),
            token=str(authorization_payload["token"]),
            session_id=str(authorization_payload["session_id"]),
            job_id=str(authorization_payload["job_id"]),
            src_buffer_id=str(authorization_payload["src_buffer_id"]),
            dst_buffer_id=str(authorization_payload["dst_buffer_id"]),
            direction=str(authorization_payload["direction"]),
            ranges=tuple(authorization_payload.get("ranges", ())),
            relay_gpu=authorization_payload.get("relay_gpu"),
        )
    except KeyError as exc:
        raise ValueError(f"missing worker authorization field: {exc.args[0]}") from exc


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
        handle_type=str(payload.get("handle_type", "registered_buffer")),
        metadata=dict(payload.get("metadata") or {}),
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
    if result.state == WorkerTransferState.FAILED and error is None:
        error = "worker transfer failed"
    return {
        "transfer_id": result.transfer_id,
        "state": daemon_state.value,
        "bytes_completed": result.bytes_completed,
        "error": error,
    }


def _validate_worker_completion_bytes(
    request: WorkerTransferRequest,
    result: WorkerTransferResult,
) -> WorkerTransferResult:
    if not isinstance(request, WorkerTransferRequest):
        raise TypeError("request must be a WorkerTransferRequest")
    if not isinstance(result, WorkerTransferResult):
        raise TypeError("result must be a WorkerTransferResult")
    if result.state != WorkerTransferState.COMPLETE:
        return result
    expected_bytes = _expected_worker_completion_bytes(request)
    if result.bytes_completed == expected_bytes:
        return result
    reported_bytes = int(result.bytes_completed)
    safe_completed = min(reported_bytes, expected_bytes)
    return WorkerTransferResult(
        transfer_id=result.transfer_id,
        state=WorkerTransferState.FAILED,
        error=(
            "worker completed "
            f"{reported_bytes} of {expected_bytes} daemon-planned bytes"
        ),
        bytes_completed=safe_completed,
        metadata={
            **dict(result.metadata),
            "completion_validation": "planned_bytes_mismatch",
            "expected_bytes": expected_bytes,
            "reported_bytes": reported_bytes,
        },
    )


def _expected_worker_completion_bytes(request: WorkerTransferRequest) -> int:
    plan = request.data_plane.plan
    total_bytes = 0
    for assignment in plan.get("assignments", ()) or ():
        if not isinstance(assignment, Mapping):
            raise ValueError("daemon plan assignment must be an object")
        for chunk in assignment.get("chunks", ()) or ():
            if not isinstance(chunk, Mapping):
                raise ValueError("daemon plan chunk must be an object")
            total_bytes += int(chunk["bytes"])
    if total_bytes <= 0:
        total_bytes = sum(int(item["bytes"]) for item in request.data_plane.ranges)
    if total_bytes <= 0:
        raise ValueError("daemon worker plan has no bytes to complete")
    return total_bytes


def _require_daemon_worker_plan(request: WorkerTransferRequest) -> None:
    plan = request.data_plane.plan
    if not plan:
        raise ValueError("daemon worker authorization did not include a transfer plan")
    assignments = plan.get("assignments")
    if not assignments:
        raise ValueError("daemon worker authorization plan has no assignments")

    relay_gpu = int(request.data_plane.relay_gpu)
    direction = request.data_plane.direction
    target_handle = (
        request.data_plane.dst_handle
        if direction == "h2d"
        else request.data_plane.src_handle
    )
    if target_handle.device_index is None:
        raise ValueError("worker target handle requires a CUDA device index")
    target_device = int(target_handle.device_index)
    relay_ranges: list[dict[str, int]] = []
    plan_total_bytes = 0
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise ValueError("daemon plan assignment must be an object")
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            raise ValueError("daemon plan assignment path must be an object")
        path_kind = str(path.get("kind", "")).lower()
        if path_kind not in {"direct", "relay"}:
            raise ValueError("daemon plan path must be direct or relay")
        if str(path.get("direction", "")).lower() != direction:
            raise ValueError("daemon plan direction does not match worker request")
        if int(path.get("target_device", target_device)) != target_device:
            raise ValueError("daemon plan target does not match worker device")
        if not bool(path.get("enabled", True)):
            raise ValueError("daemon plan path is disabled")
        if path_kind == "direct":
            chunks = assignment.get("chunks", ()) or ()
        else:
            if int(path.get("relay_device", -1)) != relay_gpu:
                raise ValueError("daemon plan relay does not match worker lease")
            chunks = assignment.get("chunks", ()) or ()
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("daemon plan chunk must be an object")
            chunk_payload = {
                "src_offset": int(chunk["src_offset"]),
                "dst_offset": int(chunk["dst_offset"]),
                "bytes": int(chunk["bytes"]),
            }
            plan_total_bytes += int(chunk_payload["bytes"])
            if path_kind == "relay":
                relay_ranges.append(chunk_payload)
    if plan_total_bytes <= 0:
        raise ValueError("daemon plan has no assigned bytes")
    declared_total_bytes = int(plan.get("total_bytes", -1))
    if declared_total_bytes != plan_total_bytes:
        raise ValueError("daemon plan total bytes do not match assigned chunks")
    if not relay_ranges:
        raise ValueError("daemon plan has no authorized relay chunks")
    if tuple(relay_ranges) != request.data_plane.ranges:
        raise ValueError("authorized ranges do not match daemon plan")


def _validate_ticket_matches_buffers(
    ticket: ExecutionTicket,
    src_buffer: BufferRegistration,
    dst_buffer: BufferRegistration,
) -> None:
    if not isinstance(ticket, ExecutionTicket):
        raise TypeError("ticket must be an ExecutionTicket")
    if not isinstance(src_buffer, BufferRegistration):
        raise TypeError("src_buffer must be a BufferRegistration")
    if not isinstance(dst_buffer, BufferRegistration):
        raise TypeError("dst_buffer must be a BufferRegistration")
    if src_buffer.buffer_id != ticket.source_buffer_id:
        raise ValueError("ticket source buffer does not match worker buffer")
    if dst_buffer.buffer_id != ticket.destination_buffer_id:
        raise ValueError("ticket destination buffer does not match worker buffer")
    if src_buffer.job_id != ticket.job_id or dst_buffer.job_id != ticket.job_id:
        raise ValueError("ticket job does not match worker buffers")


def _validate_ticket_matches_decision(
    ticket: ExecutionTicket,
    decision: SchedulingDecision,
) -> None:
    if ticket.decision_id != decision.decision_id:
        raise ValueError("ticket decision_id does not match scheduling decision")
    if ticket.intent_id != decision.intent_id:
        raise ValueError("ticket intent_id does not match scheduling decision")
    if ticket.topology_snapshot_id != decision.topology_snapshot_id:
        raise ValueError("ticket topology_snapshot_id does not match scheduling decision")
    if ticket.job_id != decision.job_id:
        raise ValueError("ticket job_id does not match scheduling decision")
    if ticket.session_id != decision.session_id:
        raise ValueError("ticket session_id does not match scheduling decision")
    if dict(ticket.plan) != dict(decision.plan):
        raise ValueError("ticket plan does not match scheduling decision")
    ticket_generation = ticket.metadata.get("plan_generation")
    decision_generation = decision.metadata.get("plan_generation")
    if (
        ticket_generation is not None
        and decision_generation is not None
        and int(ticket_generation) != int(decision_generation)
    ):
        raise ValueError("ticket plan_generation does not match scheduling decision")


def _validate_daemon_issued_ticket(
    ticket: ExecutionTicket,
    *,
    plan_generation: object | None = None,
) -> None:
    if not isinstance(ticket, ExecutionTicket):
        raise TypeError("ticket must be an ExecutionTicket")
    issuer = str(ticket.metadata.get("issuer", ""))
    if issuer != "turbobus-daemon":
        raise ValueError("execution ticket must be issued by turbobus-daemon")
    generation = ticket.metadata.get("plan_generation")
    if generation is None:
        raise ValueError("execution ticket missing plan_generation")
    if int(generation) <= 0:
        raise ValueError("execution ticket plan_generation must be positive")
    if plan_generation is not None and int(plan_generation) != int(generation):
        raise ValueError("execution ticket plan_generation is stale")
    transfer_id = ticket.metadata.get("transfer_id")
    if transfer_id is None or not str(transfer_id).strip():
        raise ValueError("execution ticket missing transfer_id")


def _validate_ticket_matches_worker_request(
    ticket: ExecutionTicket,
    authorization: WorkerTransferAuthorization,
) -> None:
    if ticket.job_id != authorization.job_id:
        raise ValueError("ticket job does not match worker authorization")
    if ticket.session_id != authorization.session_id:
        raise ValueError("ticket session does not match worker authorization")
    if ticket.source_buffer_id != authorization.src_buffer.buffer_id:
        raise ValueError("ticket source buffer does not match worker authorization")
    if ticket.destination_buffer_id != authorization.dst_buffer.buffer_id:
        raise ValueError("ticket destination buffer does not match worker authorization")
    if ticket.direction != authorization.direction:
        raise ValueError("ticket direction does not match worker authorization")
    if dict(ticket.plan) != authorization.plan:
        raise ValueError("ticket plan does not match worker authorization")
    relay = _relay_gpu_for_ticket(ticket, authorization.relay_gpu)
    if relay != authorization.relay_gpu:
        raise ValueError("ticket relay does not match worker authorization")
    if _relay_ranges_from_ticket_plan(ticket, relay_gpu=relay) != authorization.ranges:
        raise ValueError("ticket ranges do not match worker authorization")


def _relay_gpu_for_ticket(
    ticket: ExecutionTicket,
    relay_gpu: int | None,
) -> int:
    relay_devices = []
    for assignment in ticket.plan.get("assignments", ()) or ():
        if not isinstance(assignment, Mapping):
            raise ValueError("ticket plan assignment must be an object")
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            raise ValueError("ticket plan assignment path must be an object")
        if str(path.get("kind", "")).lower() == "relay":
            relay_devices.append(int(path.get("relay_device", -1)))
    relay_devices = sorted(set(relay_devices))
    if relay_gpu is not None:
        relay = int(relay_gpu)
        if relay not in relay_devices:
            raise ValueError("ticket relay does not match daemon plan")
        return relay
    if len(relay_devices) != 1:
        raise ValueError("worker ticket requires exactly one relay path")
    return relay_devices[0]


def _lease_id_for_ticket(ticket: ExecutionTicket, lease_id: str | None) -> str:
    if lease_id is not None:
        resolved = str(lease_id)
        if ticket.lease_ids and resolved not in ticket.lease_ids:
            raise ValueError("worker lease_id does not match ticket")
        return resolved
    if len(ticket.lease_ids) != 1:
        raise ValueError("worker ticket requires exactly one lease id")
    return ticket.lease_ids[0]


def _transfer_id_for_ticket(ticket: ExecutionTicket, transfer_id: str | None) -> str:
    if transfer_id is not None:
        return str(transfer_id)
    transfer_id = ticket.metadata.get("transfer_id")
    if transfer_id is None:
        return ticket.ticket_id
    return str(transfer_id)


def _relay_ranges_from_ticket_plan(
    ticket: ExecutionTicket,
    *,
    relay_gpu: int,
) -> tuple[dict[str, int], ...]:
    ranges: list[dict[str, int]] = []
    for assignment in ticket.plan.get("assignments", ()) or ():
        if not isinstance(assignment, Mapping):
            raise ValueError("ticket plan assignment must be an object")
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            raise ValueError("ticket plan assignment path must be an object")
        if str(path.get("kind", "")).lower() != "relay":
            continue
        if int(path.get("relay_device", -1)) != int(relay_gpu):
            continue
        if str(path.get("direction", "")).lower() != ticket.direction:
            raise ValueError("ticket plan direction does not match ticket")
        for chunk in assignment.get("chunks", ()) or ():
            if not isinstance(chunk, Mapping):
                raise ValueError("ticket plan chunk must be an object")
            ranges.append(
                {
                    "src_offset": int(chunk["src_offset"]),
                    "dst_offset": int(chunk["dst_offset"]),
                    "bytes": int(chunk["bytes"]),
                }
            )
    if not ranges:
        raise ValueError("ticket plan has no authorized relay chunks")
    relay_bytes = sum(item["bytes"] for item in ranges)
    if relay_bytes > ticket.total_bytes:
        raise ValueError("ticket relay ranges exceed ticket total bytes")
    return tuple(ranges)


def _failed_worker_result_from_exception(
    worker_request: WorkerTransferRequest,
    staging_slot: WorkerStagingSlot,
    exc: Exception,
) -> WorkerTransferResult:
    return WorkerTransferResult(
        transfer_id=worker_request.transfer_id,
        state=WorkerTransferState.FAILED,
        error=str(exc) or exc.__class__.__name__,
        bytes_completed=0,
        metadata={
            "relay_gpu": worker_request.authorization.relay_gpu,
            "src_buffer_id": worker_request.authorization.src_buffer.buffer_id,
            "dst_buffer_id": worker_request.authorization.dst_buffer.buffer_id,
            "staging_slot_id": staging_slot.slot_id,
        },
    )


def _cleanup_target_id(target_kind: str, lease_id: str, session_id: str) -> str:
    normalized = str(target_kind)
    if normalized == "reservation":
        return str(lease_id)
    if normalized == "session":
        return str(session_id)
    raise ValueError("worker cleanup target must be reservation or session")


def _execute_worker_transfer(
    executor,
    request: WorkerTransferRequest,
    staging_slot: WorkerStagingSlot,
    resources: WorkerDataPlaneResources,
) -> WorkerTransferResult:
    execute_bound = getattr(executor, "execute_bound", None)
    if callable(execute_bound):
        return execute_bound(request, staging_slot, resources)
    return executor.execute(request, staging_slot)


def _default_worker_executor():
    from .cuda_executor import CudaWorkerExecutor

    return CudaWorkerExecutor()


def _lifecycle_transfer_id(lifecycle: WorkerTransferLifecycleRecord) -> str:
    if lifecycle.result is not None:
        return lifecycle.result.transfer_id
    if lifecycle.worker_request is not None:
        return lifecycle.worker_request.transfer_id
    return lifecycle.authorization_request.transfer_id


def _lifecycle_lease_id(lifecycle: WorkerTransferLifecycleRecord) -> str:
    if lifecycle.worker_request is not None:
        return lifecycle.worker_request.authorization.lease_id
    return lifecycle.authorization_request.lease_id


__all__ = [
    "WorkerAuthorizationError",
    "WorkerCleanupError",
    "WorkerDataPlaneCompletion",
    "WorkerDataPlaneCompletionEnvelope",
    "WorkerDataPlaneResourceBinder",
    "WorkerDataPlaneResourceError",
    "WorkerDataPlaneResources",
    "WorkerDataPlaneRequest",
    "WorkerServiceRequestEnvelope",
    "WorkerServiceResponseEnvelope",
    "WorkerStatusReportError",
    "WorkerTransferAuthorizer",
    "WorkerTransferClient",
    "WorkerTransferCleanupCoordinator",
    "WorkerTransferLifecycleRecord",
    "WorkerTransferRequest",
    "WorkerTransferResult",
    "WorkerTransferService",
    "WorkerTransferState",
    "WorkerTransferStatusReporter",
    "parse_worker_authorization_request_payload",
]
