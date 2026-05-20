from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

try:
    import torch
except ImportError:  # pragma: no cover - import-time convenience only
    torch = None


class BlockState(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    PREFETCHING = "prefetching"
    EVICTING = "evicting"
    UNKNOWN = "unknown"


@dataclass
class OffloadBlock:
    name: str
    cpu_tensor: object
    gpu_tensor: object
    block_id: object | None = None
    cpu_slot: object | None = None
    gpu_slot: object | None = None
    state: BlockState = BlockState.CPU
    last_prefetch: object | None = None
    last_evict: object | None = None
    last_handle: object | None = None
    last_operation: str | None = None

    def __post_init__(self) -> None:
        if self.block_id is None:
            self.block_id = self.name

    @property
    def bytes(self) -> int:
        return int(self.cpu_tensor.numel() * self.cpu_tensor.element_size())

    @property
    def last_stats(self):
        if self.last_handle is None:
            return None
        return self.last_handle.stats


class OffloadStore:
    """Connector-shaped named-block layer over Runtime H2D/D2H transfers."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._blocks: dict[str, OffloadBlock] = {}

    def add(
        self,
        name: str,
        cpu_tensor,
        gpu_tensor=None,
        *,
        block_id=None,
        cpu_slot=None,
        gpu_slot=None,
    ) -> OffloadBlock:
        self._validate_name(name)
        if name in self._blocks:
            raise ValueError(f"offload block already exists: {name}")
        if gpu_tensor is None:
            gpu_tensor = self._make_gpu_tensor(cpu_tensor)
        block = OffloadBlock(
            name=name,
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            block_id=block_id,
            cpu_slot=cpu_slot,
            gpu_slot=gpu_slot,
        )
        self._blocks[name] = block
        return block

    def remove(self, name: str) -> OffloadBlock:
        return self._blocks.pop(name)

    def block(self, name: str) -> OffloadBlock:
        try:
            return self._blocks[name]
        except KeyError as exc:
            raise KeyError(f"unknown offload block: {name}") from exc

    def names(self) -> list[str]:
        return list(self._blocks)

    def blocks(self) -> Iterable[OffloadBlock]:
        return self._blocks.values()

    def prefetch(self, name: str):
        block = self.block(name)
        handle = self.runtime.fetch_to_gpu(block.cpu_tensor, block.gpu_tensor)
        block.last_prefetch = handle
        block.last_handle = handle
        block.last_operation = "prefetch"
        block.state = BlockState.PREFETCHING
        return handle

    def evict(self, name: str):
        block = self.block(name)
        handle = self.runtime.offload_to_cpu(block.gpu_tensor, block.cpu_tensor)
        block.last_evict = handle
        block.last_handle = handle
        block.last_operation = "evict"
        block.state = BlockState.EVICTING
        return handle

    def prefetch_many(self, names: Iterable[str]) -> list:
        return [self.prefetch(name) for name in names]

    def evict_many(self, names: Iterable[str]) -> list:
        return [self.evict(name) for name in names]

    def wait(self, name: str) -> None:
        block = self.block(name)
        if block.last_handle is None:
            return
        block.last_handle.wait()
        self._mark_waited(block)

    def wait_many(self, names: Iterable[str]) -> None:
        for name in names:
            self.wait(name)

    def stats(self, name: str):
        return self.block(name).last_stats

    def _mark_waited(self, block: OffloadBlock) -> None:
        if block.last_operation == "prefetch":
            block.state = BlockState.GPU
        elif block.last_operation == "evict":
            block.state = BlockState.CPU
        else:
            block.state = BlockState.UNKNOWN

    def _make_gpu_tensor(self, cpu_tensor):
        if torch is None:
            raise RuntimeError("PyTorch is required to allocate OffloadStore GPU tensors")
        return torch.empty_like(cpu_tensor, device=f"cuda:{self.runtime.target_gpu}")

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("offload block name must be a non-empty string")


OffloadManager = OffloadStore
KVBlockStore = OffloadStore
