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


@dataclass(frozen=True)
class TransferStats:
    bytes: int = 0
    direct_chunks: int = 0
    relay_chunks: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "bytes": self.bytes,
            "direct_chunks": self.direct_chunks,
            "relay_chunks": self.relay_chunks,
        }


@dataclass
class OffloadBlock:
    name: str
    cpu_tensor: object
    gpu_tensor: object
    block_id: object | None = None
    cpu_slot: object | None = None
    gpu_slot: object | None = None
    cpu_offset: int = 0
    gpu_offset: int = 0
    byte_count: int | None = None
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
        if self.byte_count is not None:
            return int(self.byte_count)
        return int(self.cpu_tensor.numel() * self.cpu_tensor.element_size())

    @property
    def last_stats(self):
        if self.last_handle is None:
            return None
        return self.last_handle.stats

    @property
    def last_transfer_stats(self) -> TransferStats | None:
        if self.last_handle is None:
            return None
        return summarize_transfer_handles([self.last_handle])


def summarize_transfer_handles(handles: Iterable) -> TransferStats:
    unique = []
    seen = set()
    for handle in handles:
        if id(handle) in seen:
            continue
        stats = getattr(handle, "stats", None)
        if stats is None:
            continue
        seen.add(id(handle))
        unique.append(stats)
    return TransferStats(
        bytes=sum(_stat_value(stats, "bytes") for stats in unique),
        direct_chunks=sum(_stat_value(stats, "direct_chunks") for stats in unique),
        relay_chunks=sum(_stat_value(stats, "relay_chunks") for stats in unique),
    )


def _stat_value(stats, name: str) -> int:
    if isinstance(stats, dict):
        return int(stats.get(name, 0) or 0)
    return int(getattr(stats, name, 0) or 0)


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
        cpu_offset: int = 0,
        gpu_offset: int = 0,
        byte_count: int | None = None,
    ) -> OffloadBlock:
        self._validate_name(name)
        self._validate_range_fields(cpu_offset, gpu_offset, byte_count)
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
            cpu_offset=int(cpu_offset),
            gpu_offset=int(gpu_offset),
            byte_count=int(byte_count) if byte_count is not None else None,
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
        blocks = [self.block(name) for name in names]
        if not blocks:
            return []
        if self._can_use_range_batch(blocks):
            ranges = self._ranges(blocks, "prefetch")
            handle = self.runtime.fetch_ranges_to_gpu(
                blocks[0].cpu_tensor,
                blocks[0].gpu_tensor,
                ranges,
            )
            self._record_many(blocks, handle, "prefetch", BlockState.PREFETCHING)
            return [handle for _ in blocks]
        return [self.prefetch(block.name) for block in blocks]

    def evict_many(self, names: Iterable[str]) -> list:
        blocks = [self.block(name) for name in names]
        if not blocks:
            return []
        if self._can_use_range_batch(blocks):
            ranges = self._ranges(blocks, "evict")
            handle = self.runtime.offload_ranges_to_cpu(
                blocks[0].gpu_tensor,
                blocks[0].cpu_tensor,
                ranges,
            )
            self._record_many(blocks, handle, "evict", BlockState.EVICTING)
            return [handle for _ in blocks]
        return [self.evict(block.name) for block in blocks]

    def wait(self, name: str) -> None:
        block = self.block(name)
        if block.last_handle is None:
            return
        block.last_handle.wait()
        self._mark_waited(block)

    def wait_many(self, names: Iterable[str]) -> None:
        waited = set()
        for name in names:
            block = self.block(name)
            handle_key = id(block.last_handle)
            if block.last_handle is not None and handle_key not in waited:
                block.last_handle.wait()
                waited.add(handle_key)
            self._mark_waited(block)

    def stats(self, name: str):
        return self.block(name).last_stats

    def transfer_stats(self, name: str) -> TransferStats | None:
        return self.block(name).last_transfer_stats

    def _mark_waited(self, block: OffloadBlock) -> None:
        if block.last_operation == "prefetch":
            block.state = BlockState.GPU
        elif block.last_operation == "evict":
            block.state = BlockState.CPU
        else:
            block.state = BlockState.UNKNOWN

    @staticmethod
    def _can_use_range_batch(blocks: list[OffloadBlock]) -> bool:
        first = blocks[0]
        if first.byte_count is None:
            return False
        return all(
            block.cpu_tensor is first.cpu_tensor
            and block.gpu_tensor is first.gpu_tensor
            and block.byte_count is not None
            for block in blocks
        )

    @staticmethod
    def _ranges(blocks: list[OffloadBlock], operation: str) -> list[dict]:
        ranges = []
        for block in blocks:
            if operation == "prefetch":
                src_offset = block.cpu_offset
                dst_offset = block.gpu_offset
            elif operation == "evict":
                src_offset = block.gpu_offset
                dst_offset = block.cpu_offset
            else:
                raise ValueError(f"unknown offload operation: {operation}")
            ranges.append(
                {
                    "src_offset": src_offset,
                    "dst_offset": dst_offset,
                    "bytes": block.bytes,
                }
            )
        return ranges

    @staticmethod
    def _record_many(
        blocks: list[OffloadBlock],
        handle,
        operation: str,
        state: BlockState,
    ) -> None:
        for block in blocks:
            if operation == "prefetch":
                block.last_prefetch = handle
            elif operation == "evict":
                block.last_evict = handle
            else:
                raise ValueError(f"unknown offload operation: {operation}")
            block.last_handle = handle
            block.last_operation = operation
            block.state = state

    def _make_gpu_tensor(self, cpu_tensor):
        if torch is None:
            raise RuntimeError("PyTorch is required to allocate OffloadStore GPU tensors")
        return torch.empty_like(cpu_tensor, device=f"cuda:{self.runtime.target_gpu}")

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("offload block name must be a non-empty string")

    @staticmethod
    def _validate_range_fields(
        cpu_offset: int,
        gpu_offset: int,
        byte_count: int | None,
    ) -> None:
        if cpu_offset < 0 or gpu_offset < 0:
            raise ValueError("block offsets must be non-negative")
        if byte_count is not None and byte_count <= 0:
            raise ValueError("byte_count must be positive")


OffloadManager = OffloadStore
KVBlockStore = OffloadStore
