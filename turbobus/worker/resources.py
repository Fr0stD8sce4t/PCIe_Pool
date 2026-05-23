from __future__ import annotations

from dataclasses import dataclass

from ..backends.cuda import default_cuda_backend
from ..client import SharedPinnedCpuBuffer
from ..schema import BufferRegistration, WorkerBufferHandle, WorkerDataPlaneRequest


class WorkerDataPlaneResourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerDataPlaneResources:
    request: WorkerDataPlaneRequest
    source_cpu_buffer: SharedPinnedCpuBuffer
    cuda_host_registered: bool = False

    @property
    def source_host_ptr(self) -> int:
        return self.source_cpu_buffer.address

    @property
    def source_bytes(self) -> int:
        return self.source_cpu_buffer.size_bytes

    def close(self) -> None:
        self.source_cpu_buffer.close()

    def as_dict(self) -> dict[str, object]:
        return {
            "transfer_id": self.request.transfer_id,
            "lease_id": self.request.lease_id,
            "src_buffer_id": self.request.src_handle.buffer_id,
            "src_handle_type": self.request.src_handle.handle_type,
            "source_host_ptr": self.source_host_ptr,
            "source_bytes": self.source_bytes,
            "cuda_host_registered": self.cuda_host_registered,
        }


class WorkerDataPlaneResourceBinding:
    def __init__(
        self,
        request: WorkerDataPlaneRequest,
        *,
        backend=default_cuda_backend,
        register_cuda_host: bool = True,
    ) -> None:
        if not isinstance(request, WorkerDataPlaneRequest):
            raise TypeError("request must be a WorkerDataPlaneRequest")
        self.request = request
        self.backend = backend
        self.register_cuda_host = bool(register_cuda_host)
        self._resources: WorkerDataPlaneResources | None = None

    def __enter__(self) -> WorkerDataPlaneResources:
        source_buffer: SharedPinnedCpuBuffer | None = None
        try:
            source_buffer = SharedPinnedCpuBuffer.open_from_registration(
                _registration_from_worker_handle(self.request.src_handle)
            )
            if self.register_cuda_host:
                source_buffer.register_for_cuda(self.backend)
            self._resources = WorkerDataPlaneResources(
                request=self.request,
                source_cpu_buffer=source_buffer,
                cuda_host_registered=self.register_cuda_host,
            )
            return self._resources
        except Exception as exc:
            if source_buffer is not None:
                source_buffer.close()
            raise WorkerDataPlaneResourceError(
                f"failed to bind shared CPU source buffer: {exc}"
            ) from exc

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._resources is not None:
            self._resources.close()
            self._resources = None


class WorkerDataPlaneResourceBinder:
    def __init__(
        self,
        *,
        backend=default_cuda_backend,
        register_cuda_host: bool = True,
    ) -> None:
        self.backend = backend
        self.register_cuda_host = bool(register_cuda_host)

    def bind(
        self,
        request: WorkerDataPlaneRequest,
    ) -> WorkerDataPlaneResourceBinding:
        return WorkerDataPlaneResourceBinding(
            request,
            backend=self.backend,
            register_cuda_host=self.register_cuda_host,
        )


def _registration_from_worker_handle(handle: WorkerBufferHandle) -> BufferRegistration:
    if not isinstance(handle, WorkerBufferHandle):
        raise TypeError("handle must be a WorkerBufferHandle")
    if handle.handle_type != "shared_pinned_cpu":
        raise WorkerDataPlaneResourceError(
            "worker shared CPU binding requires a shared_pinned_cpu source handle"
        )
    return BufferRegistration(
        buffer_id=handle.buffer_id,
        job_id=handle.job_id,
        kind=handle.kind,
        size_bytes=handle.size_bytes,
        device_index=handle.device_index,
        address=handle.address,
        pinned=handle.pinned,
        handle_type=handle.handle_type,
        metadata=handle.metadata,
    )


__all__ = [
    "WorkerDataPlaneResourceBinder",
    "WorkerDataPlaneResourceBinding",
    "WorkerDataPlaneResourceError",
    "WorkerDataPlaneResources",
]
