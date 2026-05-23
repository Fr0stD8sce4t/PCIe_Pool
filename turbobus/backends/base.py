from __future__ import annotations

from typing import Any, Iterable, Protocol

from ..schema import TransferMode


class TransferBackend(Protocol):
    def bind_runtime(self, native_module: Any, torch_module: Any) -> None:
        ...

    def require_available(self) -> None:
        ...

    def require_torch(self) -> None:
        ...

    def transfer_mode_value(self, mode: TransferMode | str) -> Any:
        ...

    def create_runtime(self, options: Any) -> Any:
        ...

    def initialize_runtime(
        self,
        runtime: Any,
        target_device: int,
        relay_gpus: Iterable[int],
    ) -> None:
        ...

    def make_ranges(
        self,
        ranges: Iterable,
        source_bytes: int,
        destination_bytes: int,
    ) -> list:
        ...

    def make_transfer_plan(self, plan: Any) -> Any:
        ...

    def fetch_plan_to_gpu(
        self,
        runtime: Any,
        host_ptr: int,
        host_bytes: int,
        target_ptr: int,
        target_bytes: int,
        plan: Any,
    ) -> Any:
        ...

    def offload_plan_to_cpu(
        self,
        runtime: Any,
        target_ptr: int,
        target_bytes: int,
        host_ptr: int,
        host_bytes: int,
        plan: Any,
    ) -> Any:
        ...

    def wait(self, runtime: Any, handle: Any) -> None:
        ...

    def stats(self, runtime: Any, handle: Any) -> Any:
        ...


__all__ = ["TransferBackend"]
