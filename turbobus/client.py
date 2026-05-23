from __future__ import annotations

import ctypes
import uuid
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any

from .backends.cuda import default_cuda_backend
from .schema import BufferRegistration, DaemonResponse


def _shared_memory_name(prefix: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in str(prefix)
    ).strip("_")
    if not normalized:
        normalized = "turbobus"
    return f"{normalized}_{uuid.uuid4().hex}"


@dataclass
class SharedPinnedCpuBuffer:
    buffer_id: str
    job_id: str
    size_bytes: int
    shared_memory_name: str
    shared_memory_size_bytes: int
    _shared_memory: shared_memory.SharedMemory = field(repr=False, compare=False)
    offset_bytes: int = 0
    owner: bool = False
    _cuda_registered: bool = field(default=False, init=False, repr=False)
    _cuda_backend: Any | None = field(default=None, init=False, repr=False)
    _cuda_registered_address: int | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _unlinked: bool = field(default=False, init=False, repr=False)

    @classmethod
    def allocate(
        cls,
        buffer_id: str,
        job_id: str,
        size_bytes: int,
        *,
        name_prefix: str = "turbobus",
    ) -> "SharedPinnedCpuBuffer":
        size_bytes = int(size_bytes)
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        name = _shared_memory_name(name_prefix)
        shared = shared_memory.SharedMemory(name=name, create=True, size=size_bytes)
        return cls(
            buffer_id=str(buffer_id),
            job_id=str(job_id),
            size_bytes=size_bytes,
            shared_memory_name=shared.name,
            shared_memory_size_bytes=size_bytes,
            owner=True,
            _shared_memory=shared,
        )

    @classmethod
    def open_from_registration(
        cls,
        registration: BufferRegistration,
    ) -> "SharedPinnedCpuBuffer":
        if not isinstance(registration, BufferRegistration):
            raise TypeError("registration must be a BufferRegistration")
        if registration.handle_type != "shared_pinned_cpu":
            raise ValueError("registration must be a shared_pinned_cpu buffer")
        metadata = registration.metadata
        name = str(metadata["shared_memory_name"])
        offset = int(metadata["offset_bytes"])
        shared_size = int(
            metadata.get(
                "shared_memory_size_bytes",
                offset + registration.size_bytes,
            )
        )
        shared = shared_memory.SharedMemory(name=name, create=False)
        return cls(
            buffer_id=registration.buffer_id,
            job_id=registration.job_id,
            size_bytes=registration.size_bytes,
            shared_memory_name=name,
            shared_memory_size_bytes=shared_size,
            offset_bytes=offset,
            owner=False,
            _shared_memory=shared,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "shared_memory_name": self.shared_memory_name,
            "offset_bytes": self.offset_bytes,
            "shared_memory_size_bytes": self.shared_memory_size_bytes,
        }

    @property
    def view(self) -> memoryview:
        if self._closed:
            raise RuntimeError("shared CPU buffer is closed")
        start = self.offset_bytes
        end = start + self.size_bytes
        return self._shared_memory.buf[start:end]

    @property
    def address(self) -> int:
        view = self.view
        char = ctypes.c_char.from_buffer(view)
        try:
            return ctypes.addressof(char)
        finally:
            del char
            view.release()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def cuda_registered(self) -> bool:
        return self._cuda_registered

    def write(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        payload = bytes(data)
        offset = int(offset)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        end = offset + len(payload)
        if end > self.size_bytes:
            raise ValueError("write exceeds shared CPU buffer")
        view = self.view
        try:
            view[offset:end] = payload
        finally:
            view.release()

    def read(self, size: int | None = None, offset: int = 0) -> bytes:
        offset = int(offset)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        size_bytes = self.size_bytes - offset if size is None else int(size)
        if size_bytes < 0:
            raise ValueError("size must be non-negative")
        end = offset + size_bytes
        if end > self.size_bytes:
            raise ValueError("read exceeds shared CPU buffer")
        view = self.view
        try:
            return bytes(view[offset:end])
        finally:
            view.release()

    def buffer_registration(self) -> BufferRegistration:
        return BufferRegistration(
            buffer_id=self.buffer_id,
            job_id=self.job_id,
            kind="cpu_pinned",
            size_bytes=self.size_bytes,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata=self.metadata,
        )

    def register_with_daemon(self, daemon_client) -> DaemonResponse:
        return daemon_client.register_buffer(
            buffer_id=self.buffer_id,
            job_id=self.job_id,
            kind="cpu_pinned",
            size_bytes=self.size_bytes,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata=self.metadata,
        )

    def register_for_cuda(self, backend=default_cuda_backend) -> None:
        if self._cuda_registered:
            return
        address = self.address
        backend.register_host_memory(address, self.size_bytes)
        self._cuda_backend = backend
        self._cuda_registered_address = address
        self._cuda_registered = True

    def unregister_from_cuda(self) -> None:
        if not self._cuda_registered:
            return
        backend = self._cuda_backend or default_cuda_backend
        backend.unregister_host_memory(self._cuda_registered_address)
        self._cuda_registered = False
        self._cuda_backend = None
        self._cuda_registered_address = None

    def close(self) -> None:
        if self._closed:
            return
        self.unregister_from_cuda()
        self._shared_memory.close()
        self._closed = True

    def unlink(self) -> None:
        if self._unlinked:
            return
        self._shared_memory.unlink()
        self._unlinked = True

    def release(self) -> None:
        try:
            self.close()
        finally:
            if self.owner:
                self.unlink()

    def __enter__(self) -> "SharedPinnedCpuBuffer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class SharedPinnedCpuBufferAllocator:
    def __init__(self, name_prefix: str = "turbobus") -> None:
        self.name_prefix = str(name_prefix)

    def allocate(
        self,
        buffer_id: str,
        job_id: str,
        size_bytes: int,
    ) -> SharedPinnedCpuBuffer:
        return SharedPinnedCpuBuffer.allocate(
            buffer_id=buffer_id,
            job_id=job_id,
            size_bytes=size_bytes,
            name_prefix=self.name_prefix,
        )


@dataclass(frozen=True)
class CudaIpcDeviceBuffer:
    buffer_id: str
    job_id: str
    device_index: int
    size_bytes: int
    cuda_ipc_handle: bytes
    device_ptr: int | None = None

    @classmethod
    def from_device_pointer(
        cls,
        buffer_id: str,
        job_id: str,
        device_index: int,
        size_bytes: int,
        device_ptr: int,
        *,
        backend=default_cuda_backend,
    ) -> "CudaIpcDeviceBuffer":
        size_bytes = int(size_bytes)
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        ptr = int(device_ptr)
        if ptr <= 0:
            raise ValueError("device_ptr must be positive")
        _set_cuda_device_if_available(backend, int(device_index))
        return cls(
            buffer_id=str(buffer_id),
            job_id=str(job_id),
            device_index=int(device_index),
            size_bytes=size_bytes,
            cuda_ipc_handle=backend.export_device_ipc_handle(ptr),
            device_ptr=ptr,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {"cuda_ipc_handle": self.cuda_ipc_handle.hex()}

    def buffer_registration(self) -> BufferRegistration:
        return BufferRegistration(
            buffer_id=self.buffer_id,
            job_id=self.job_id,
            kind="gpu",
            size_bytes=self.size_bytes,
            device_index=self.device_index,
            address=self.device_ptr,
            handle_type="cuda_ipc_device",
            metadata=self.metadata,
        )

    def register_with_daemon(self, daemon_client) -> DaemonResponse:
        return daemon_client.register_buffer(
            buffer_id=self.buffer_id,
            job_id=self.job_id,
            kind="gpu",
            size_bytes=self.size_bytes,
            device_index=self.device_index,
            address=self.device_ptr,
            handle_type="cuda_ipc_device",
            metadata=self.metadata,
        )


def _set_cuda_device_if_available(backend, device_index: int) -> None:
    setter = getattr(backend, "set_device", None)
    if callable(setter):
        setter(int(device_index))


__all__ = [
    "CudaIpcDeviceBuffer",
    "SharedPinnedCpuBuffer",
    "SharedPinnedCpuBufferAllocator",
]
