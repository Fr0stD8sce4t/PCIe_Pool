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

    def make_ranges(
        self,
        ranges: Iterable,
        source_bytes: int,
        destination_bytes: int,
    ) -> list:
        ...


__all__ = ["TransferBackend"]
