from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .backends.cuda import default_cuda_backend
from .client import CudaIpcDeviceBuffer, SharedPinnedCpuBuffer
from .runtime_engine import RuntimeOptions
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
    lease_token: Mapping[str, object] | None
    authorization_request: WorkerTransferAuthorizationRequest | None
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


class _WorkerCompletionEnvelopeError(RuntimeError):
    pass


@dataclass
class WorkerManagedTransferClient:
    daemon_client: object
    worker_client: object
    target_gpu: int
    relay_gpus: Iterable[int]
    max_inflight_chunks: int = 8
    backend: object = default_cuda_backend
    runtime_options: RuntimeOptions = field(default_factory=RuntimeOptions)
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
        if _is_direct_only_worker_plan(planned.payload):
            return _execute_direct_fallback_transfer(
                daemon_client=self.daemon_client,
                backend=self.backend,
                runtime_options=self.runtime_options,
                transfer_request=transfer_request,
                planned_payload=planned.payload,
                session_id=session_id,
                job_id=job,
                source=source,
                target=target,
            )
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
                expected_bytes=transfer_request.total_bytes,
            )
        except _WorkerCompletionEnvelopeError:
            _cleanup_planned_relay_lease(
                self.daemon_client,
                lease_token,
                reason="worker_completion_invalid",
                strict=False,
            )
            raise
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
        ranges=[item.as_dict() for item in request.ranges] if request.ranges else None,
    )


def _is_direct_only_worker_plan(plan_payload: Mapping[str, object]) -> bool:
    if plan_payload.get("lease_tokens") or plan_payload.get("reservations"):
        return False
    plan = plan_payload.get("plan")
    if not isinstance(plan, Mapping):
        return False
    assignments = plan.get("assignments", ()) or ()
    if not assignments:
        return False
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            return False
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            return False
        if str(path.get("kind", "")).lower() != "direct":
            return False
    return True


def _execute_direct_fallback_transfer(
    *,
    daemon_client,
    backend,
    runtime_options: RuntimeOptions,
    transfer_request: TransferRequest,
    planned_payload: Mapping[str, object],
    session_id: str,
    job_id: str,
    source: SharedPinnedCpuBuffer | CudaIpcDeviceBuffer,
    target: SharedPinnedCpuBuffer | CudaIpcDeviceBuffer,
) -> WorkerManagedTransferResult:
    transfer_id = str(planned_payload["transfer_id"])
    try:
        _execute_direct_plan(
            backend=backend,
            runtime_options=runtime_options,
            direction=transfer_request.direction.value,
            plan_payload=dict(planned_payload["plan"]),
            source=source,
            target=target,
        )
    except Exception as exc:
        daemon_client.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed=0,
            error=str(exc) or exc.__class__.__name__,
        )
        raise
    completed = daemon_client.transfer_status(
        transfer_id,
        state="complete",
        bytes_completed=transfer_request.total_bytes,
    )
    _require_ok(completed, "daemon direct transfer completion update failed")
    status = daemon_client.transfer_status(transfer_id)
    _require_ok(status, "daemon transfer status query failed")
    final_status = dict(status.payload["status"])
    _require_daemon_transfer_complete(
        final_status,
        expected_bytes=transfer_request.total_bytes,
    )
    return WorkerManagedTransferResult(
        transfer_id=transfer_id,
        session_id=session_id,
        job_id=job_id,
        source_buffer_id=source.buffer_id,
        target_buffer_id=target.buffer_id,
        plan=planned_payload,
        lease_token=None,
        authorization_request=None,
        worker_lifecycle=None,
        worker_completion=None,
        final_status=final_status,
    )


def _execute_direct_plan(
    *,
    backend,
    runtime_options: RuntimeOptions,
    direction: str,
    plan_payload: Mapping[str, object],
    source: SharedPinnedCpuBuffer | CudaIpcDeviceBuffer,
    target: SharedPinnedCpuBuffer | CudaIpcDeviceBuffer,
) -> None:
    if direction == "h2d":
        if not isinstance(source, SharedPinnedCpuBuffer):
            raise TypeError("direct h2d source must be a SharedPinnedCpuBuffer")
        if not isinstance(target, CudaIpcDeviceBuffer):
            raise TypeError("direct h2d target must be a CudaIpcDeviceBuffer")
        _require_device_pointer(target)
        _run_direct_plan(
            backend=backend,
            runtime_options=runtime_options,
            target_device=target.device_index,
            plan_payload=plan_payload,
            host_buffer=source,
            device_ptr=int(target.device_ptr),
            device_bytes=target.size_bytes,
            direction=direction,
        )
        return
    if not isinstance(source, CudaIpcDeviceBuffer):
        raise TypeError("direct d2h source must be a CudaIpcDeviceBuffer")
    if not isinstance(target, SharedPinnedCpuBuffer):
        raise TypeError("direct d2h target must be a SharedPinnedCpuBuffer")
    _require_device_pointer(source)
    _run_direct_plan(
        backend=backend,
        runtime_options=runtime_options,
        target_device=source.device_index,
        plan_payload=plan_payload,
        host_buffer=target,
        device_ptr=int(source.device_ptr),
        device_bytes=source.size_bytes,
        direction=direction,
    )


def _run_direct_plan(
    *,
    backend,
    runtime_options: RuntimeOptions,
    target_device: int,
    plan_payload: Mapping[str, object],
    host_buffer: SharedPinnedCpuBuffer,
    device_ptr: int,
    device_bytes: int,
    direction: str,
) -> None:
    native_plan = backend.make_transfer_plan(plan_payload)
    runtime = backend.create_runtime(runtime_options)
    backend.initialize_runtime(runtime, int(target_device), [])
    host_buffer.register_for_cuda(backend)
    try:
        if direction == "h2d":
            handle = backend.fetch_plan_to_gpu(
                runtime,
                host_buffer.address,
                host_buffer.size_bytes,
                device_ptr,
                int(device_bytes),
                native_plan,
            )
        else:
            handle = backend.offload_plan_to_cpu(
                runtime,
                device_ptr,
                int(device_bytes),
                host_buffer.address,
                host_buffer.size_bytes,
                native_plan,
            )
        backend.wait(runtime, handle)
    finally:
        host_buffer.unregister_from_cuda()


def _require_device_pointer(buffer: CudaIpcDeviceBuffer) -> None:
    if buffer.device_ptr is None or int(buffer.device_ptr) <= 0:
        raise ValueError("direct fallback requires a local CUDA device pointer")


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
    *,
    expected_bytes: int,
) -> _WorkerExecutionResult:
    lifecycle_submitter = getattr(worker_client, "submit_report_cleanup_lifecycle", None)
    if callable(lifecycle_submitter):
        lifecycle = lifecycle_submitter(request, cleanup_target_kind="reservation")
        completion = lifecycle.completion_envelope()
        _require_worker_completion_matches_request(
            completion,
            request,
            expected_bytes=expected_bytes,
        )
        return _WorkerExecutionResult(
            final_state=lifecycle.final_state,
            error=lifecycle.error,
            lifecycle=lifecycle,
            completion=completion,
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
        _require_worker_completion_matches_request(
            completion,
            request,
            expected_bytes=expected_bytes,
        )
        return _WorkerExecutionResult(
            final_state=completion.final_state,
            error=completion.error,
            lifecycle=None,
            completion=completion,
        )
    raise TypeError("worker_client must submit worker-managed transfers")


def _require_worker_completion_matches_request(
    completion: WorkerDataPlaneCompletionEnvelope,
    request: WorkerTransferAuthorizationRequest,
    *,
    expected_bytes: int,
) -> None:
    if not isinstance(completion, WorkerDataPlaneCompletionEnvelope):
        raise _WorkerCompletionEnvelopeError(
            "worker completion must be a WorkerDataPlaneCompletionEnvelope"
        )
    if completion.transfer_id is not None and completion.transfer_id != request.transfer_id:
        raise _WorkerCompletionEnvelopeError("worker completion transfer mismatch")
    if completion.lease_id is not None and completion.lease_id != request.lease_id:
        raise _WorkerCompletionEnvelopeError("worker completion lease mismatch")
    _require_worker_mapping_matches_request(
        completion.worker_result,
        request,
        label="worker result",
    )
    _require_worker_mapping_matches_request(
        completion.daemon_status_update,
        request,
        label="worker daemon status update",
    )
    _require_worker_daemon_response_matches_request(
        completion.daemon_status_response,
        request,
    )
    final_state = "" if completion.final_state is None else str(completion.final_state)
    if final_state == "complete":
        if not completion.ok:
            raise _WorkerCompletionEnvelopeError("worker completion was not ok")
        if completion.transfer_id is None:
            raise _WorkerCompletionEnvelopeError("worker completion missing transfer id")
        if completion.lease_id is None:
            raise _WorkerCompletionEnvelopeError("worker completion missing lease id")
        if completion.worker_result is None:
            raise _WorkerCompletionEnvelopeError("worker completion missing worker result")
        result_state = _state_text(completion.worker_result.get("state", ""))
        if result_state != "complete":
            raise _WorkerCompletionEnvelopeError("worker result did not complete")
        _require_worker_completed_bytes(
            completion.worker_result,
            int(expected_bytes),
            label="worker result",
        )
        if completion.daemon_status_update is None:
            raise _WorkerCompletionEnvelopeError(
                "worker completion missing daemon status update"
            )
        if completion.daemon_status_response is None:
            raise _WorkerCompletionEnvelopeError(
                "worker completion missing daemon status response"
            )
        update_state = _state_text(completion.daemon_status_update.get("state", ""))
        if update_state != "complete":
            raise _WorkerCompletionEnvelopeError(
                "worker daemon status update did not complete"
            )
        _require_worker_completed_bytes(
            completion.daemon_status_update,
            int(expected_bytes),
            label="worker daemon status update",
        )
        if not bool(completion.daemon_status_response.get("ok", False)):
            raise _WorkerCompletionEnvelopeError(
                "worker daemon status response was not ok"
            )
        _require_worker_daemon_response_completed_bytes(
            completion.daemon_status_response,
            int(expected_bytes),
        )
        if completion.daemon_cleanup_response is None:
            raise _WorkerCompletionEnvelopeError(
                "worker completion missing daemon release response"
            )
        _require_worker_release_response_matches_request(
            completion.daemon_cleanup_response,
            request,
        )
        _require_worker_staging_release_matches_request(
            completion.staging_release,
            request,
        )


def _require_worker_mapping_matches_request(
    payload: Mapping[str, object] | None,
    request: WorkerTransferAuthorizationRequest,
    *,
    label: str,
) -> None:
    if payload is None:
        return
    transfer_id = payload.get("transfer_id")
    if transfer_id is not None and str(transfer_id) != request.transfer_id:
        raise _WorkerCompletionEnvelopeError(f"{label} transfer mismatch")
    lease_id = payload.get("lease_id")
    if lease_id is not None and str(lease_id) != request.lease_id:
        raise _WorkerCompletionEnvelopeError(f"{label} lease mismatch")


def _require_worker_daemon_response_matches_request(
    response: Mapping[str, object] | None,
    request: WorkerTransferAuthorizationRequest,
) -> None:
    if response is None:
        return
    payload = response.get("payload")
    if not isinstance(payload, Mapping):
        return
    status = payload.get("status")
    if not isinstance(status, Mapping):
        return
    _require_worker_mapping_matches_request(
        status,
        request,
        label="worker daemon status response",
    )


def _require_worker_daemon_response_completed_bytes(
    response: Mapping[str, object] | None,
    expected_bytes: int,
) -> None:
    if response is None:
        return
    payload = response.get("payload")
    if not isinstance(payload, Mapping):
        return
    status = payload.get("status")
    if not isinstance(status, Mapping):
        return
    status_state = _state_text(status.get("state", ""))
    if status_state and status_state != "complete":
        raise _WorkerCompletionEnvelopeError(
            "worker daemon status response did not complete"
        )
    _require_worker_completed_bytes(
        status,
        expected_bytes,
        label="worker daemon status response",
    )


def _require_worker_release_response_matches_request(
    response: Mapping[str, object],
    request: WorkerTransferAuthorizationRequest,
) -> None:
    if not bool(response.get("ok", False)):
        raise _WorkerCompletionEnvelopeError(
            "worker daemon release response was not ok"
        )
    payload = response.get("payload")
    if not isinstance(payload, Mapping):
        return
    reservation_id = payload.get("reservation_id")
    if reservation_id is not None and str(reservation_id) != request.lease_id:
        raise _WorkerCompletionEnvelopeError(
            "worker daemon release response reservation mismatch"
        )


def _require_worker_staging_release_matches_request(
    release: Mapping[str, object] | None,
    request: WorkerTransferAuthorizationRequest,
) -> None:
    if release is None:
        raise _WorkerCompletionEnvelopeError(
            "worker completion missing staging release"
        )
    if bool(release.get("active", True)):
        raise _WorkerCompletionEnvelopeError("worker staging release is still active")
    transfer_id = release.get("transfer_id")
    if transfer_id is not None and str(transfer_id) != request.transfer_id:
        raise _WorkerCompletionEnvelopeError("worker staging release transfer mismatch")
    lease_id = release.get("lease_id")
    if lease_id is not None and str(lease_id) != request.lease_id:
        raise _WorkerCompletionEnvelopeError("worker staging release lease mismatch")


def _require_worker_completed_bytes(
    payload: Mapping[str, object],
    expected_bytes: int,
    *,
    label: str,
) -> None:
    if "bytes_completed" not in payload:
        raise _WorkerCompletionEnvelopeError(f"{label} missing completed bytes")
    try:
        bytes_completed = int(payload["bytes_completed"])
    except (TypeError, ValueError) as exc:
        raise _WorkerCompletionEnvelopeError(
            f"{label} completed bytes are invalid"
        ) from exc
    if bytes_completed != int(expected_bytes):
        raise _WorkerCompletionEnvelopeError(
            f"{label} completed byte mismatch: "
            f"{bytes_completed} != {int(expected_bytes)}"
        )
    if "bytes_total" not in payload:
        return
    try:
        bytes_total = int(payload["bytes_total"])
    except (TypeError, ValueError) as exc:
        raise _WorkerCompletionEnvelopeError(
            f"{label} total bytes are invalid"
        ) from exc
    if bytes_total != int(expected_bytes):
        raise _WorkerCompletionEnvelopeError(
            f"{label} total byte mismatch: {bytes_total} != {int(expected_bytes)}"
        )


def _state_text(state: object) -> str:
    return str(getattr(state, "value", state)).lower()


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
    backend=default_cuda_backend,
    runtime_options: RuntimeOptions | None = None,
) -> WorkerManagedTransferClient:
    options = runtime_options or RuntimeOptions()
    return WorkerManagedTransferClient(
        daemon_client=daemon_client,
        worker_client=worker_client or WorkerTransferClient(
            daemon_client,
            executor=CudaWorkerExecutor(backend=backend, options=options),
            resource_binder=WorkerDataPlaneResourceBinder(backend=backend),
        ),
        target_gpu=int(target_gpu),
        relay_gpus=tuple(int(gpu) for gpu in relay_gpus),
        max_inflight_chunks=int(max_inflight_chunks),
        backend=backend,
        runtime_options=options,
    )


__all__ = [
    "WorkerManagedTransferClient",
    "WorkerManagedTransferResult",
    "make_worker_managed_transfer_client",
]
