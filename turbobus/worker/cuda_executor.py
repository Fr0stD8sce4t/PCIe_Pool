from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..backends.cuda import default_cuda_backend
from ..runtime_engine import RuntimeOptions
from .helper import (
    WorkerTransferRequest,
    WorkerTransferResult,
    WorkerTransferState,
)
from .resources import WorkerDataPlaneResources
from .staging_pool import WorkerStagingSlot


class CudaWorkerExecutor:
    """CUDA worker executor for daemon-authorized worker-managed transfers."""

    def __init__(
        self,
        *,
        backend=default_cuda_backend,
        options: RuntimeOptions | None = None,
    ) -> None:
        self.backend = backend
        self.options = options or RuntimeOptions()

    def execute(
        self,
        request: WorkerTransferRequest,
        staging_slot: WorkerStagingSlot,
    ) -> WorkerTransferResult:
        _validate_request_and_slot(request, staging_slot)
        return _failed_result(
            request,
            staging_slot,
            "CUDA worker execution requires bound data-plane resources",
        )

    def execute_bound(
        self,
        request: WorkerTransferRequest,
        staging_slot: WorkerStagingSlot,
        resources: WorkerDataPlaneResources,
    ) -> WorkerTransferResult:
        _validate_request_and_slot(request, staging_slot)
        if not isinstance(resources, WorkerDataPlaneResources):
            raise TypeError("resources must be WorkerDataPlaneResources")
        if resources.request != request.data_plane:
            return _failed_result(
                request,
                staging_slot,
                "bound resources do not match the worker request",
            )
        target_device = _target_device_for_request(request)
        if target_device is None:
            return _failed_result(
                request,
                staging_slot,
                "CUDA worker executor requires a GPU device index",
            )

        try:
            plan_payload = _worker_plan_payload(request, int(target_device))
            native_plan = self.backend.make_transfer_plan(plan_payload)
            runtime = self.backend.create_runtime(_runtime_options_for_request(
                self.options,
                request,
            ))
            self.backend.initialize_runtime(
                runtime,
                int(target_device),
                [request.data_plane.relay_gpu],
            )
            if request.data_plane.direction == "h2d":
                handle = self.backend.fetch_plan_to_gpu(
                    runtime,
                    resources.host_ptr,
                    resources.host_bytes,
                    resources.device_ptr,
                    resources.device_bytes,
                    native_plan,
                )
            else:
                handle = self.backend.offload_plan_to_cpu(
                    runtime,
                    resources.device_ptr,
                    resources.device_bytes,
                    resources.host_ptr,
                    resources.host_bytes,
                    native_plan,
                )
            self.backend.wait(runtime, handle)
            stats = self.backend.stats(runtime, handle)
        except Exception as exc:
            return _failed_result(request, staging_slot, str(exc))

        bytes_completed = _stats_int(stats, "bytes", int(plan_payload["total_bytes"]))
        planned_direct_bytes = _assignment_byte_count(plan_payload, "direct")
        planned_relay_bytes = _assignment_byte_count(plan_payload, "relay")
        direct_chunks = _stats_int(
            stats,
            "direct_chunks",
            _assignment_chunk_count(plan_payload, "direct"),
        )
        relay_chunks = _stats_int(
            stats,
            "relay_chunks",
            _assignment_chunk_count(plan_payload, "relay"),
        )
        return WorkerTransferResult(
            transfer_id=request.transfer_id,
            state=WorkerTransferState.COMPLETE,
            bytes_completed=bytes_completed,
            metadata={
                "executor": "cuda_worker",
                "path": _metadata_path(
                    direction=request.data_plane.direction,
                    direct_chunks=direct_chunks,
                ),
                "plan_source": "daemon",
                "relay_gpu": request.data_plane.relay_gpu,
                "target_device": int(target_device),
                "src_buffer_id": request.data_plane.src_handle.buffer_id,
                "dst_buffer_id": request.data_plane.dst_handle.buffer_id,
                "staging_slot_id": staging_slot.slot_id,
                "direct_bytes": _stats_int(
                    stats,
                    "direct_bytes",
                    planned_direct_bytes,
                ),
                "direct_chunks": direct_chunks,
                "relay_bytes": _stats_int(
                    stats,
                    "relay_bytes",
                    planned_relay_bytes,
                ),
                "relay_chunks": relay_chunks,
            },
        )


def _runtime_options_for_request(
    options: RuntimeOptions,
    request: WorkerTransferRequest,
) -> RuntimeOptions:
    max_chunk_bytes = request.data_plane.staging.max_chunk_bytes
    return replace(
        options,
        chunk_bytes=max(int(options.chunk_bytes), int(max_chunk_bytes)),
    )


def _target_device_for_request(request: WorkerTransferRequest) -> int | None:
    handle = (
        request.data_plane.dst_handle
        if request.data_plane.direction == "h2d"
        else request.data_plane.src_handle
    )
    return handle.device_index


def _worker_plan_payload(
    request: WorkerTransferRequest,
    target_device: int,
) -> dict[str, object]:
    if not request.data_plane.plan:
        raise ValueError("CUDA worker executor requires a daemon-issued transfer plan")
    return _relay_scoped_daemon_plan_payload(request, int(target_device))


def _relay_scoped_daemon_plan_payload(
    request: WorkerTransferRequest,
    target_device: int,
) -> dict[str, object]:
    source_plan = dict(request.data_plane.plan)
    assignments: list[dict[str, object]] = []
    relay_ranges: list[dict[str, int]] = []
    total_bytes = 0
    relay_gpu = int(request.data_plane.relay_gpu)
    for assignment in source_plan.get("assignments", ()) or ():
        if not isinstance(assignment, dict):
            raise ValueError("daemon plan assignment must be a mapping")
        path = assignment.get("path")
        if not isinstance(path, dict):
            raise ValueError("daemon plan assignment path must be a mapping")
        path_kind = str(path.get("kind", "")).lower()
        if path_kind not in {"direct", "relay"}:
            raise ValueError("daemon plan path must be direct or relay")
        if str(path.get("direction", "")).lower() != request.data_plane.direction:
            raise ValueError("daemon plan direction does not match worker request")
        plan_path = dict(path)
        if int(plan_path.get("target_device", target_device)) != int(target_device):
            raise ValueError("daemon plan target does not match worker target")
        if not bool(plan_path.get("enabled", True)):
            raise ValueError("daemon plan path is disabled")
        if path_kind == "relay":
            if int(plan_path.get("relay_device", -1)) != relay_gpu:
                raise ValueError("daemon plan relay does not match worker lease")
            plan_path["relay_device"] = relay_gpu
        else:
            plan_path["relay_device"] = -1
        plan_path["enabled"] = True
        chunks = []
        for chunk in assignment.get("chunks", ()) or ():
            if not isinstance(chunk, dict):
                raise ValueError("daemon plan chunk must be a mapping")
            chunks.append(
                {
                    "src_offset": int(chunk["src_offset"]),
                    "dst_offset": int(chunk["dst_offset"]),
                    "bytes": int(chunk["bytes"]),
                }
            )
        if not chunks:
            continue
        chunk_bytes = sum(int(chunk["bytes"]) for chunk in chunks)
        total_bytes += chunk_bytes
        if path_kind == "relay":
            relay_ranges.extend(chunks)
        assignments.append(
            {
                "path": plan_path,
                "chunks": chunks,
                "bytes": chunk_bytes,
                "chunk_count": len(chunks),
            }
        )
    if not assignments:
        raise ValueError("daemon plan has no authorized relay chunks")
    declared_total_bytes = int(source_plan.get("total_bytes", total_bytes))
    if declared_total_bytes != total_bytes:
        raise ValueError("daemon plan total bytes do not match assigned chunks")
    if tuple(relay_ranges) != request.data_plane.ranges:
        raise ValueError("authorized ranges do not match daemon plan")
    return {
        "total_bytes": declared_total_bytes,
        "chunk_bytes": int(
            source_plan.get("chunk_bytes", request.data_plane.staging.max_chunk_bytes)
        ),
        "assignments": assignments,
    }


def _assignment_chunk_count(plan_payload: dict[str, object], path_kind: str) -> int:
    total = 0
    for assignment in plan_payload.get("assignments", ()) or ():
        path = assignment.get("path") if isinstance(assignment, dict) else None
        if not isinstance(path, dict):
            continue
        if str(path.get("kind", "")).lower() != path_kind:
            continue
        total += len(assignment.get("chunks", ()) or ())
    return total


def _assignment_byte_count(plan_payload: dict[str, object], path_kind: str) -> int:
    total = 0
    for assignment in plan_payload.get("assignments", ()) or ():
        path = assignment.get("path") if isinstance(assignment, dict) else None
        if not isinstance(path, dict):
            continue
        if str(path.get("kind", "")).lower() != path_kind:
            continue
        for chunk in assignment.get("chunks", ()) or ():
            if isinstance(chunk, dict):
                total += int(chunk.get("bytes", 0))
    return total


def _metadata_path(*, direction: str, direct_chunks: int) -> str:
    prefix = "pool" if direct_chunks > 0 else "relay"
    return f"{prefix}_{direction}"


def _validate_request_and_slot(
    request: WorkerTransferRequest,
    staging_slot: WorkerStagingSlot,
) -> None:
    if not isinstance(request, WorkerTransferRequest):
        raise TypeError("request must be a WorkerTransferRequest")
    if request.ticket is None:
        raise ValueError("CUDA worker executor requires a daemon-issued ExecutionTicket")
    if not isinstance(staging_slot, WorkerStagingSlot):
        raise TypeError("staging_slot must be a WorkerStagingSlot")
    if staging_slot.transfer_id != request.transfer_id:
        raise ValueError("staging slot transfer does not match request")
    if staging_slot.lease_id != request.authorization.lease_id:
        raise ValueError("staging slot lease does not match request")
    if staging_slot.relay_gpu != request.authorization.relay_gpu:
        raise ValueError("staging slot relay does not match request")


def _failed_result(
    request: WorkerTransferRequest,
    staging_slot: WorkerStagingSlot,
    error: str,
) -> WorkerTransferResult:
    return WorkerTransferResult(
        transfer_id=request.transfer_id,
        state=WorkerTransferState.FAILED,
        error=error,
        bytes_completed=0,
        metadata={
            "executor": "cuda_worker",
            "relay_gpu": request.authorization.relay_gpu,
            "src_buffer_id": request.authorization.src_buffer.buffer_id,
            "dst_buffer_id": request.authorization.dst_buffer.buffer_id,
            "staging_slot_id": staging_slot.slot_id,
        },
    )


def _stats_int(stats: Any, field_name: str, default: int) -> int:
    value = getattr(stats, field_name, default)
    return int(value if value is not None else default)


__all__ = ["CudaWorkerExecutor"]
