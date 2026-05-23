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
    """CUDA worker executor for the first daemon-authorized H2D relay path."""

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
        if request.data_plane.direction != "h2d":
            return _failed_result(
                request,
                staging_slot,
                "CUDA worker executor currently supports only h2d relay transfers",
            )
        target_device = request.data_plane.dst_handle.device_index
        if target_device is None:
            return _failed_result(
                request,
                staging_slot,
                "CUDA worker executor requires a target GPU device index",
            )

        try:
            plan_payload = _relay_h2d_plan_payload(request, int(target_device))
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
            handle = self.backend.fetch_plan_to_gpu(
                runtime,
                resources.source_host_ptr,
                resources.source_bytes,
                resources.target_device_ptr,
                resources.target_device_bytes,
                native_plan,
            )
            self.backend.wait(runtime, handle)
            stats = self.backend.stats(runtime, handle)
        except Exception as exc:
            return _failed_result(request, staging_slot, str(exc))

        bytes_completed = _stats_int(stats, "bytes", request.data_plane.staging.total_bytes)
        return WorkerTransferResult(
            transfer_id=request.transfer_id,
            state=WorkerTransferState.COMPLETE,
            bytes_completed=bytes_completed,
            metadata={
                "executor": "cuda_worker",
                "path": "relay_h2d",
                "relay_gpu": request.data_plane.relay_gpu,
                "target_device": int(target_device),
                "src_buffer_id": request.data_plane.src_handle.buffer_id,
                "dst_buffer_id": request.data_plane.dst_handle.buffer_id,
                "staging_slot_id": staging_slot.slot_id,
                "relay_bytes": _stats_int(stats, "relay_bytes", bytes_completed),
                "relay_chunks": _stats_int(
                    stats,
                    "relay_chunks",
                    len(request.data_plane.ranges),
                ),
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


def _relay_h2d_plan_payload(
    request: WorkerTransferRequest,
    target_device: int,
) -> dict[str, object]:
    ranges = [dict(item) for item in request.data_plane.ranges]
    total_bytes = sum(int(item["bytes"]) for item in ranges)
    return {
        "total_bytes": total_bytes,
        "chunk_bytes": request.data_plane.staging.max_chunk_bytes,
        "assignments": [
            {
                "path": {
                    "kind": "relay",
                    "direction": "h2d",
                    "target_device": int(target_device),
                    "relay_device": request.data_plane.relay_gpu,
                    "enabled": True,
                },
                "chunks": ranges,
                "bytes": total_bytes,
                "chunk_count": len(ranges),
            }
        ],
    }


def _validate_request_and_slot(
    request: WorkerTransferRequest,
    staging_slot: WorkerStagingSlot,
) -> None:
    if not isinstance(request, WorkerTransferRequest):
        raise TypeError("request must be a WorkerTransferRequest")
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
