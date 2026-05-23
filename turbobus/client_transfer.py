from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .client import CudaIpcDeviceBuffer, SharedPinnedCpuBuffer
from .schema import BufferRegistration, DaemonResponse, WorkerTransferAuthorizationRequest
from .transfer import TransferRange, TransferRequest
from .worker import (
    CudaWorkerExecutor,
    WorkerDataPlaneCompletionEnvelope,
    WorkerDataPlaneResourceBinder,
    WorkerServiceRequestEnvelope,
    WorkerTransferClient,
    WorkerTransferLifecycleRecord,
)


@dataclass(frozen=True)
class WorkerManagedTransferResult:
    transfer_id: str
    session_id: str
    job_id: str
    source_buffer_id: str
    target_buffer_id: str
    plan: Mapping[str, object]
    lease_token: Mapping[str, object]
    authorization_request: WorkerTransferAuthorizationRequest
    worker_lifecycle: WorkerTransferLifecycleRecord | None
    final_status: Mapping[str, object]
    worker_completion: WorkerDataPlaneCompletionEnvelope | None = None

    @property
    def bytes_completed(self) -> int:
        return int(self.final_status.get("bytes_completed", 0))

    @property
    def state(self) -> str:
        state = self.final_status.get("state", "unknown")
        return str(getattr(state, "value", state))


@dataclass
class WorkerManagedTransferClient:
    daemon_client: object
    worker_client: object
    target_gpu: int
    relay_gpus: Iterable[int]
    max_inflight_chunks: int = 8
    _session_id: str | None = field(default=None, init=False, repr=False)

    def open_session(self) -> str:
        if self._session_id is not None:
            return self._session_id
        response = self.daemon_client.register_session(
            int(self.target_gpu),
            [int(gpu) for gpu in self.relay_gpus],
            int(self.max_inflight_chunks),
        )
        _require_ok(response, "daemon session registration failed")
        session_id = str(response.payload["session"]["session_id"])
        self._session_id = session_id
        return session_id

    def close_session(self) -> DaemonResponse:
        if self._session_id is None:
            return DaemonResponse(ok=True, payload={"closed": False})
        response = self.daemon_client.close_session(self._session_id)
        if response.ok:
            self._session_id = None
        return response

    def fetch_shared_cpu_to_cuda_ipc(
        self,
        source: SharedPinnedCpuBuffer,
        target: CudaIpcDeviceBuffer,
        *,
        ranges: Iterable[TransferRange | tuple[int, int, int] | dict] | None = None,
        chunk_bytes: int = 16 * 1024 * 1024,
        mode: str = "relay",
        job_id: str | None = None,
        user_id: str | None = None,
    ) -> WorkerManagedTransferResult:
        if not isinstance(source, SharedPinnedCpuBuffer):
            raise TypeError("source must be a SharedPinnedCpuBuffer")
        if not isinstance(target, CudaIpcDeviceBuffer):
            raise TypeError("target must be a CudaIpcDeviceBuffer")
        return self._submit_worker_managed_transfer(
            source,
            target,
            direction="h2d",
            ranges=ranges,
            chunk_bytes=chunk_bytes,
            mode=mode,
            job_id=job_id,
            user_id=user_id,
        )

    def offload_cuda_ipc_to_shared_cpu(
        self,
        source: CudaIpcDeviceBuffer,
        target: SharedPinnedCpuBuffer,
        *,
        ranges: Iterable[TransferRange | tuple[int, int, int] | dict] | None = None,
        chunk_bytes: int = 16 * 1024 * 1024,
        mode: str = "relay",
        job_id: str | None = None,
        user_id: str | None = None,
    ) -> WorkerManagedTransferResult:
        if not isinstance(source, CudaIpcDeviceBuffer):
            raise TypeError("source must be a CudaIpcDeviceBuffer")
        if not isinstance(target, SharedPinnedCpuBuffer):
            raise TypeError("target must be a SharedPinnedCpuBuffer")
        return self._submit_worker_managed_transfer(
            source,
            target,
            direction="d2h",
            ranges=ranges,
            chunk_bytes=chunk_bytes,
            mode=mode,
            job_id=job_id,
            user_id=user_id,
        )

    def _submit_worker_managed_transfer(
        self,
        source: SharedPinnedCpuBuffer | CudaIpcDeviceBuffer,
        target: SharedPinnedCpuBuffer | CudaIpcDeviceBuffer,
        *,
        direction: str,
        ranges: Iterable[TransferRange | tuple[int, int, int] | dict] | None,
        chunk_bytes: int,
        mode: str,
        job_id: str | None,
        user_id: str | None,
    ) -> WorkerManagedTransferResult:
        job = str(job_id or source.job_id)
        if target.job_id != job or source.job_id != job:
            raise ValueError("source and target buffers must belong to the transfer job")
        session_id = self.open_session()
        _require_ok(
            self.daemon_client.register_job(
                job_id=job,
                user_id=user_id,
                session_id=session_id,
            ),
            "daemon job registration failed",
        )
        source_registration = source.buffer_registration()
        target_registration = target.buffer_registration()
        _register_buffer(self.daemon_client, source_registration)
        _register_buffer(self.daemon_client, target_registration)

        transfer_request = TransferRequest.from_ranges(
            _ranges_or_full_buffer(ranges, source.size_bytes, target.size_bytes),
            chunk_bytes=int(chunk_bytes),
            direction=direction,
            mode=mode,
            job_id=job,
            metadata={
                "buffer_ids": (
                    source.buffer_id,
                    target.buffer_id,
                )
            },
        )
        planned = _plan_transfer_request(
            self.daemon_client,
            session_id,
            transfer_request,
            mode=mode,
        )
        _require_ok(planned, "daemon transfer planning failed")
        lease_token = _single_lease_token(self.daemon_client, planned)
        try:
            _require_single_relay_worker_plan(
                planned.payload,
                lease_token,
                direction=direction,
            )
        except Exception:
            _cleanup_planned_relay_lease(self.daemon_client, lease_token)
            raise
        authorization_request = WorkerTransferAuthorizationRequest(
            transfer_id=str(planned.payload["transfer_id"]),
            lease_id=str(lease_token["lease_id"]),
            token=str(lease_token["token"]),
            session_id=session_id,
            job_id=job,
            src_buffer_id=source.buffer_id,
            dst_buffer_id=target.buffer_id,
            direction=direction,
            ranges=(),
            relay_gpu=int(lease_token["relay_gpu"]),
        )
        try:
            worker_execution = _submit_worker_execution(
                self.worker_client,
                authorization_request,
            )
        except Exception:
            _cleanup_planned_relay_lease(
                self.daemon_client,
                lease_token,
                reason="worker_execution_exception",
                strict=False,
            )
            raise
        try:
            status = self.daemon_client.transfer_status(
                str(planned.payload["transfer_id"])
            )
            _require_ok(status, "daemon transfer status query failed")
            final_status = dict(status.payload["status"])
        except Exception:
            _cleanup_planned_relay_lease(
                self.daemon_client,
                lease_token,
                reason="daemon_status_query_failed",
                strict=False,
            )
            raise
        if worker_execution.final_state != "complete":
            _cleanup_planned_relay_lease(
                self.daemon_client,
                lease_token,
                reason="worker_completion_not_complete",
                strict=False,
            )
            raise RuntimeError(
                worker_execution.error
                or final_status.get("error")
                or "worker-managed transfer did not complete"
            )
        try:
            _require_daemon_transfer_complete(
                final_status,
                expected_bytes=transfer_request.total_bytes,
            )
        except Exception:
            _cleanup_planned_relay_lease(
                self.daemon_client,
                lease_token,
                reason="daemon_completion_mismatch",
                strict=False,
            )
            raise
        return WorkerManagedTransferResult(
            transfer_id=str(planned.payload["transfer_id"]),
            session_id=session_id,
            job_id=job,
            source_buffer_id=source.buffer_id,
            target_buffer_id=target.buffer_id,
            plan=planned.payload,
            lease_token=lease_token,
            authorization_request=authorization_request,
            worker_lifecycle=worker_execution.lifecycle,
            worker_completion=worker_execution.completion,
            final_status=final_status,
        )


def _ranges_or_full_buffer(
    ranges: Iterable[TransferRange | tuple[int, int, int] | dict] | None,
    source_bytes: int,
    target_bytes: int,
) -> tuple[TransferRange | tuple[int, int, int] | dict, ...]:
    if ranges is not None:
        return tuple(ranges)
    bytes_to_copy = min(int(source_bytes), int(target_bytes))
    if bytes_to_copy <= 0:
        raise ValueError("transfer buffers must be non-empty")
    return (TransferRange(src_offset=0, dst_offset=0, bytes=bytes_to_copy),)


def _register_buffer(daemon_client, registration: BufferRegistration) -> None:
    response = daemon_client.register_buffer(
        buffer_id=registration.buffer_id,
        job_id=registration.job_id,
        kind=registration.kind,
        size_bytes=registration.size_bytes,
        device_index=registration.device_index,
        address=registration.address,
        pinned=registration.pinned,
        handle_type=registration.handle_type,
        metadata=registration.metadata,
    )
    _require_ok(response, "daemon buffer registration failed")


def _plan_transfer_request(
    daemon_client,
    session_id: str,
    request: TransferRequest,
    *,
    mode: str,
) -> DaemonResponse:
    planner = getattr(daemon_client, "plan_transfer_request", None)
    if callable(planner):
        return planner(session_id, request, mode=mode)
    return daemon_client.plan_transfer(
        session_id=session_id,
        total_bytes=request.total_bytes,
        chunk_bytes=request.chunk_bytes,
        mode=mode,
        direction=request.direction.value,
        job_id=request.job_id,
        buffer_ids=list(request.metadata["buffer_ids"]),
    )


def _single_lease_token(daemon_client, response: DaemonResponse) -> Mapping[str, object]:
    lease_tokens = response.payload.get("lease_tokens") or ()
    if len(lease_tokens) != 1:
        for lease_token in lease_tokens:
            _cleanup_planned_relay_lease(daemon_client, lease_token)
        raise RuntimeError("worker-managed transfer requires exactly one relay lease")
    return dict(lease_tokens[0])


def _require_single_relay_worker_plan(
    plan_payload: Mapping[str, object],
    lease_token: Mapping[str, object],
    *,
    direction: str,
) -> None:
    plan = plan_payload.get("plan")
    if not isinstance(plan, Mapping):
        raise RuntimeError("daemon response did not include a transfer plan")
    relay_gpu = int(lease_token["relay_gpu"])
    expected_direction = str(direction).lower()
    found_relay_chunks = False
    for assignment in plan.get("assignments", ()) or ():
        if not isinstance(assignment, Mapping):
            raise RuntimeError("daemon transfer plan assignment must be a mapping")
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            raise RuntimeError("daemon transfer plan assignment has no path")
        path_kind = str(path.get("kind", "")).lower()
        path_direction = str(path.get("direction", "")).lower()
        assignment_relay = int(path.get("relay_device", -1))
        if path_direction != expected_direction:
            raise RuntimeError(
                f"worker-managed transfer requires daemon {expected_direction} plans"
            )
        if path_kind == "direct":
            continue
        if path_kind != "relay" or assignment_relay != relay_gpu:
            raise RuntimeError(
                "worker-managed transfer currently supports direct chunks "
                "plus the leased relay only"
            )
        if assignment.get("chunks"):
            found_relay_chunks = True
    if not found_relay_chunks:
        raise RuntimeError("daemon relay plan did not include worker chunks")


def _cleanup_planned_relay_lease(
    daemon_client,
    lease_token: Mapping[str, object],
    *,
    reason: str = "unsupported_worker_plan",
    strict: bool = True,
) -> None:
    cleanup = getattr(daemon_client, "cleanup", None)
    if not callable(cleanup):
        return
    response = cleanup(
        target_kind="reservation",
        target_id=str(lease_token["lease_id"]),
        reason=reason,
        force=True,
    )
    if strict:
        _require_ok(response, "daemon reservation cleanup failed")


def _require_daemon_transfer_complete(
    final_status: Mapping[str, object],
    *,
    expected_bytes: int,
) -> None:
    if not isinstance(final_status, Mapping):
        raise TypeError("final_status must be a mapping")
    expected = int(expected_bytes)
    state = final_status.get("state", "unknown")
    state_text = str(getattr(state, "value", state))
    if state_text != "complete":
        error = final_status.get("error")
        suffix = f": {error}" if error else ""
        raise RuntimeError(
            f"daemon transfer status did not complete: {state_text}{suffix}"
        )
    bytes_total = int(final_status.get("bytes_total", expected))
    if bytes_total != expected:
        raise RuntimeError(
            f"daemon transfer byte total mismatch: {bytes_total} != {expected}"
        )
    bytes_completed = int(final_status.get("bytes_completed", -1))
    if bytes_completed != expected:
        raise RuntimeError(
            "daemon transfer completed an unexpected byte count: "
            f"{bytes_completed} != {expected}"
        )


@dataclass(frozen=True)
class _WorkerExecutionResult:
    final_state: str | None
    error: str | None
    lifecycle: WorkerTransferLifecycleRecord | None
    completion: WorkerDataPlaneCompletionEnvelope | None


def _submit_worker_execution(
    worker_client,
    request: WorkerTransferAuthorizationRequest,
) -> _WorkerExecutionResult:
    lifecycle_submitter = getattr(worker_client, "submit_report_cleanup_lifecycle", None)
    if callable(lifecycle_submitter):
        lifecycle = lifecycle_submitter(request, cleanup_target_kind="reservation")
        return _WorkerExecutionResult(
            final_state=lifecycle.final_state,
            error=lifecycle.error,
            lifecycle=lifecycle,
            completion=lifecycle.completion_envelope(),
        )
    envelope_submitter = getattr(worker_client, "submit_envelope", None)
    if callable(envelope_submitter):
        completion = envelope_submitter(
            WorkerServiceRequestEnvelope(
                payload={
                    "transfer_id": request.transfer_id,
                    "lease_id": request.lease_id,
                    "token": request.token,
                    "session_id": request.session_id,
                    "job_id": request.job_id,
                    "src_buffer_id": request.src_buffer_id,
                    "dst_buffer_id": request.dst_buffer_id,
                    "direction": request.direction,
                    "ranges": list(request.ranges),
                    "relay_gpu": request.relay_gpu,
                },
                cleanup_target_kind="reservation",
            )
        )
        return _WorkerExecutionResult(
            final_state=completion.final_state,
            error=completion.error,
            lifecycle=None,
            completion=completion,
        )
    raise TypeError("worker_client must submit worker-managed transfers")


def _require_ok(response: DaemonResponse, message: str) -> None:
    if not isinstance(response, DaemonResponse):
        raise TypeError("daemon response must be a DaemonResponse")
    if not response.ok:
        raise RuntimeError(response.error or message)


def make_worker_managed_transfer_client(
    daemon_client,
    *,
    target_gpu: int,
    relay_gpus: Iterable[int],
    worker_client: object | None = None,
    max_inflight_chunks: int = 8,
) -> WorkerManagedTransferClient:
    return WorkerManagedTransferClient(
        daemon_client=daemon_client,
        worker_client=worker_client or WorkerTransferClient(
            daemon_client,
            executor=CudaWorkerExecutor(),
            resource_binder=WorkerDataPlaneResourceBinder(),
        ),
        target_gpu=int(target_gpu),
        relay_gpus=tuple(int(gpu) for gpu in relay_gpus),
        max_inflight_chunks=int(max_inflight_chunks),
    )


__all__ = [
    "WorkerManagedTransferClient",
    "WorkerManagedTransferResult",
    "make_worker_managed_transfer_client",
]
