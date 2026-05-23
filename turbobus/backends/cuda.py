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

    def transfer_mode_value(self, mode: TransferMode | str) -> Any:
        return self._runtime_engine._runtime_transfer_mode_value(mode)

    def create_runtime(self, options: Any) -> Any:
        self.require_available()
        return self._runtime_engine._turbobus.Runtime(options.to_native())

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


default_cuda_backend = CudaNativeBackend()


__all__ = ["CudaNativeBackend", "default_cuda_backend"]
