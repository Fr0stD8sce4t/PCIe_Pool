from __future__ import annotations

from typing import Any, Iterable

from .. import runtime_engine
from ..schema import TransferMode


class CudaNativeBackend:
    """Backend facade for the current CUDA native extension."""

    def __init__(self, runtime_engine_module=runtime_engine) -> None:
        self._runtime_engine = runtime_engine_module

    def bind_runtime(self, native_module: Any, torch_module: Any) -> None:
        self._runtime_engine._turbobus = native_module
        self._runtime_engine.torch = torch_module

    def require_available(self) -> None:
        self._runtime_engine._require_extension()

    def require_torch(self) -> None:
        self._runtime_engine._require_torch()

    def set_device(self, device_index: int) -> None:
        device = int(device_index)
        if device < 0:
            raise ValueError("device_index must be non-negative")
        self.require_available()
        setter = getattr(self._runtime_engine._turbobus, "set_device", None)
        if not callable(setter):
            raise RuntimeError("native runtime does not support CUDA device selection")
        setter(device)

    def transfer_mode_value(self, mode: TransferMode | str) -> Any:
        return self._runtime_engine._runtime_transfer_mode_value(mode)

    def create_runtime(self, options: Any) -> Any:
        self.require_available()
        return self._runtime_engine._turbobus.Runtime(options.to_native())

    def initialize_runtime(
        self,
        runtime: Any,
        target_device: int,
        relay_gpus: Iterable[int],
    ) -> None:
        initializer = getattr(runtime, "init", None)
        if not callable(initializer):
            raise RuntimeError("native runtime does not support initialization")
        initializer(int(target_device), [int(gpu) for gpu in relay_gpus])

    def make_ranges(
        self,
        ranges: Iterable,
        source_bytes: int,
        destination_bytes: int,
    ) -> list:
        return self._runtime_engine._native_ranges(ranges, source_bytes, destination_bytes)

    def make_transfer_plan(self, plan: Any) -> Any:
        return self._runtime_engine._native_transfer_plan(plan)

    def register_host_memory(self, host_ptr: int, bytes_: int) -> None:
        ptr = int(host_ptr)
        size_bytes = int(bytes_)
        if ptr <= 0:
            raise ValueError("host_ptr must be positive")
        if size_bytes <= 0:
            raise ValueError("bytes must be positive")
        self.require_available()
        registrar = getattr(self._runtime_engine._turbobus, "register_host_memory", None)
        if not callable(registrar):
            raise RuntimeError("native runtime does not support host memory registration")
        registrar(ptr, size_bytes)

    def unregister_host_memory(self, host_ptr: int) -> None:
        ptr = int(host_ptr)
        if ptr <= 0:
            raise ValueError("host_ptr must be positive")
        self.require_available()
        unregister = getattr(
            self._runtime_engine._turbobus,
            "unregister_host_memory",
            None,
        )
        if not callable(unregister):
            raise RuntimeError("native runtime does not support host memory registration")
        unregister(ptr)

    def export_device_ipc_handle(self, device_ptr: int) -> bytes:
        ptr = int(device_ptr)
        if ptr <= 0:
            raise ValueError("device_ptr must be positive")
        self.require_available()
        exporter = getattr(
            self._runtime_engine._turbobus,
            "export_device_ipc_handle",
            None,
        )
        if not callable(exporter):
            raise RuntimeError("native runtime does not support CUDA IPC handles")
        return bytes(exporter(ptr))

    def open_device_ipc_handle(self, cuda_ipc_handle: bytes | bytearray | str) -> int:
        handle = _coerce_cuda_ipc_handle(cuda_ipc_handle)
        self.require_available()
        opener = getattr(
            self._runtime_engine._turbobus,
            "open_device_ipc_handle",
            None,
        )
        if not callable(opener):
            raise RuntimeError("native runtime does not support CUDA IPC handles")
        ptr = int(opener(handle))
        if ptr <= 0:
            raise RuntimeError("native runtime returned an invalid CUDA IPC pointer")
        return ptr

    def close_device_ipc_handle(self, device_ptr: int) -> None:
        ptr = int(device_ptr)
        if ptr <= 0:
            raise ValueError("device_ptr must be positive")
        self.require_available()
        closer = getattr(
            self._runtime_engine._turbobus,
            "close_device_ipc_handle",
            None,
        )
        if not callable(closer):
            raise RuntimeError("native runtime does not support CUDA IPC handles")
        closer(ptr)

    def fetch_plan_to_gpu(
        self,
        runtime: Any,
        host_ptr: int,
        host_bytes: int,
        target_ptr: int,
        target_bytes: int,
        plan: Any,
    ) -> Any:
        submitter = getattr(runtime, "fetch_plan_to_gpu", None)
        if not callable(submitter):
            raise RuntimeError("native runtime does not support exact transfer plans")
        return submitter(host_ptr, host_bytes, target_ptr, target_bytes, plan)

    def offload_plan_to_cpu(
        self,
        runtime: Any,
        target_ptr: int,
        target_bytes: int,
        host_ptr: int,
        host_bytes: int,
        plan: Any,
    ) -> Any:
        submitter = getattr(runtime, "offload_plan_to_cpu", None)
        if not callable(submitter):
            raise RuntimeError("native runtime does not support exact transfer plans")
        return submitter(target_ptr, target_bytes, host_ptr, host_bytes, plan)

    def wait(self, runtime: Any, handle: Any) -> None:
        waiter = getattr(runtime, "wait", None)
        if not callable(waiter):
            raise RuntimeError("native runtime does not support transfer waiting")
        waiter(handle)

    def stats(self, runtime: Any, handle: Any) -> Any:
        statter = getattr(runtime, "stats", None)
        if not callable(statter):
            raise RuntimeError("native runtime does not support transfer stats")
        return statter(handle)


default_cuda_backend = CudaNativeBackend()


def _coerce_cuda_ipc_handle(handle: bytes | bytearray | str) -> bytes:
    if isinstance(handle, str):
        try:
            return bytes.fromhex(handle)
        except ValueError as exc:
            raise ValueError("cuda_ipc_handle string must be hex encoded") from exc
    return bytes(handle)


__all__ = ["CudaNativeBackend", "default_cuda_backend"]
