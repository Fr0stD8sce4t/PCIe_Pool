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
    cpu_buffer: SharedPinnedCpuBuffer
    device_ptr: int
    device_bytes: int
    cuda_host_registered: bool = False

    @property
    def host_ptr(self) -> int:
        return self.cpu_buffer.address

    @property
    def host_bytes(self) -> int:
        return self.cpu_buffer.size_bytes

    @property
    def source_cpu_buffer(self) -> SharedPinnedCpuBuffer:
        return self.cpu_buffer

    @property
    def source_host_ptr(self) -> int:
        return self.host_ptr

    @property
    def source_bytes(self) -> int:
        return self.host_bytes

    @property
    def target_device_ptr(self) -> int:
        return self.device_ptr

    @property
    def target_device_bytes(self) -> int:
        return self.device_bytes

    def close(self) -> None:
        self.cpu_buffer.close()

    def as_dict(self) -> dict[str, object]:
        cpu_handle = (
            self.request.src_handle
            if self.request.direction == "h2d"
            else self.request.dst_handle
        )
        device_handle = (
            self.request.dst_handle
            if self.request.direction == "h2d"
            else self.request.src_handle
        )
        return {
            "transfer_id": self.request.transfer_id,
            "lease_id": self.request.lease_id,
            "direction": self.request.direction,
            "src_buffer_id": self.request.src_handle.buffer_id,
            "src_handle_type": self.request.src_handle.handle_type,
            "dst_buffer_id": self.request.dst_handle.buffer_id,
            "dst_handle_type": self.request.dst_handle.handle_type,
            "cpu_buffer_id": cpu_handle.buffer_id,
            "cpu_handle_type": cpu_handle.handle_type,
            "host_ptr": self.host_ptr,
            "host_bytes": self.host_bytes,
            "device_buffer_id": device_handle.buffer_id,
            "device_handle_type": device_handle.handle_type,
            "device_ptr": self.device_ptr,
            "device_bytes": self.device_bytes,
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
        self._device_ptr: int | None = None

    def __enter__(self) -> WorkerDataPlaneResources:
        cpu_buffer: SharedPinnedCpuBuffer | None = None
        try:
            cpu_handle = _cpu_handle_for_request(self.request)
            device_handle = _device_handle_for_request(self.request)
            cpu_buffer = SharedPinnedCpuBuffer.open_from_registration(
                _registration_from_worker_handle(cpu_handle)
            )
            if self.register_cuda_host:
                cpu_buffer.register_for_cuda(self.backend)
            self._device_ptr = _open_cuda_ipc_device_handle(
                self.backend,
                device_handle,
            )
            self._resources = WorkerDataPlaneResources(
                request=self.request,
                cpu_buffer=cpu_buffer,
                device_ptr=self._device_ptr,
                device_bytes=device_handle.size_bytes,
                cuda_host_registered=self.register_cuda_host,
            )
            return self._resources
        except Exception as exc:
            if self._device_ptr is not None:
                self.backend.close_device_ipc_handle(self._device_ptr)
                self._device_ptr = None
            if cpu_buffer is not None:
                cpu_buffer.close()
            raise WorkerDataPlaneResourceError(
                f"failed to bind worker data-plane resources: {exc}"
            ) from exc

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._resources is not None:
                self._resources.close()
                self._resources = None
        finally:
            if self._device_ptr is not None:
                self.backend.close_device_ipc_handle(self._device_ptr)
                self._device_ptr = None


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


def _cpu_handle_for_request(request: WorkerDataPlaneRequest) -> WorkerBufferHandle:
    return request.src_handle if request.direction == "h2d" else request.dst_handle


def _device_handle_for_request(request: WorkerDataPlaneRequest) -> WorkerBufferHandle:
    return request.dst_handle if request.direction == "h2d" else request.src_handle


def _open_cuda_ipc_device_handle(backend, handle: WorkerBufferHandle) -> int:
    if not isinstance(handle, WorkerBufferHandle):
        raise TypeError("handle must be a WorkerBufferHandle")
    if handle.handle_type != "cuda_ipc_device":
        raise WorkerDataPlaneResourceError(
            "worker device binding requires a cuda_ipc_device handle"
        )
    return backend.open_device_ipc_handle(handle.metadata["cuda_ipc_handle"])


__all__ = [
    "WorkerDataPlaneResourceBinder",
    "WorkerDataPlaneResourceBinding",
    "WorkerDataPlaneResourceError",
    "WorkerDataPlaneResources",
]
