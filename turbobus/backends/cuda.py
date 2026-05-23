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


default_cuda_backend = CudaNativeBackend()


__all__ = ["CudaNativeBackend", "default_cuda_backend"]
