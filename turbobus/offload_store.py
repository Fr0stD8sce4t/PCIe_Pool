from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Protocol
import uuid

from .schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind


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


class TransferIntentClient(Protocol):
    def submit_transfer_intent(self, intent: TransferIntent) -> TransferReceipt:
        ...

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> TransferReceipt:
        ...


@dataclass(frozen=True)
class AdapterTransferContext:
    job_id: str
    session_id: str
    cpu_buffer_id: str
    gpu_buffer_id: str
    workload_kind: WorkloadKind | str = WorkloadKind.GENERIC
    priority: int = 0
    policy_hints: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    intent_prefix: str | None = None
    wait_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _require_non_empty(self.job_id, "job_id"))
        object.__setattr__(
            self,
            "session_id",
            _require_non_empty(self.session_id, "session_id"),
        )
        object.__setattr__(
            self,
            "cpu_buffer_id",
            _require_non_empty(self.cpu_buffer_id, "cpu_buffer_id"),
        )
        object.__setattr__(
            self,
            "gpu_buffer_id",
            _require_non_empty(self.gpu_buffer_id, "gpu_buffer_id"),
        )
        object.__setattr__(self, "workload_kind", WorkloadKind(self.workload_kind))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(
            self,
            "policy_hints",
            _validate_policy_hints_no_physical(self.policy_hints),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        prefix = self.intent_prefix or f"adapter-{uuid.uuid4()}"
        object.__setattr__(
            self,
            "intent_prefix",
            _require_non_empty(prefix, "intent_prefix"),
        )
        if self.wait_timeout_seconds is not None:
            timeout = float(self.wait_timeout_seconds)
            if timeout < 0:
                raise ValueError("wait_timeout_seconds must be non-negative")
            object.__setattr__(self, "wait_timeout_seconds", timeout)


@dataclass
class ReceiptTransferHandle:
    client: TransferIntentClient
    intent: TransferIntent
    receipt: TransferReceipt
    wait_timeout_seconds: float | None = None
    wait_calls: int = 0
    _waited: bool = field(default=False, init=False, repr=False)

    @property
    def stats(self) -> TransferStats:
        return transfer_stats_from_receipt(self.receipt)

    def wait(self) -> TransferReceipt:
        if self._waited:
            return self.receipt
        self.receipt = self.client.wait_transfer_receipt(
            self.intent.intent_id,
            timeout_seconds=self.wait_timeout_seconds,
        )
        if not isinstance(self.receipt, TransferReceipt):
            raise TypeError("wait_transfer_receipt must return a TransferReceipt")
        if self.receipt.intent_id != self.intent.intent_id:
            raise ValueError("receipt intent_id does not match transfer intent")
        self.wait_calls += 1
        self._waited = True
        state = TransferStatusState(self.receipt.state)
        if state in {TransferStatusState.FAILED, TransferStatusState.CANCELED}:
            raise RuntimeError(self.receipt.error or f"transfer {state.value}")
        return self.receipt


@dataclass(frozen=True)
class OffloadBlockInfo:
    name: str
    block_id: object
    cpu_slot: object | None
    gpu_slot: object | None
    cpu_offset: int
    gpu_offset: int
    bytes: int
    state: BlockState
    last_operation: str | None
    transfer_stats: TransferStats | None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "block_id": self.block_id,
            "cpu_slot": self.cpu_slot,
            "gpu_slot": self.gpu_slot,
            "cpu_offset": self.cpu_offset,
            "gpu_offset": self.gpu_offset,
            "bytes": self.bytes,
            "state": self.state.value,
            "last_operation": self.last_operation,
            "transfer_stats": (
                self.transfer_stats.as_dict()
                if self.transfer_stats is not None
                else None
            ),
        }


@dataclass(frozen=True)
class OffloadBatch:
    operation: str
    names: tuple[str, ...]
    handles: tuple[object, ...]
    store: "OffloadStore" = field(repr=False, compare=False)

    def wait(self) -> None:
        self.store.wait_many(self.names)

    def transfer_stats(self) -> TransferStats:
        return self.store.transfer_stats_many(self.names)

    def block_infos(self) -> list[OffloadBlockInfo]:
        return self.store.block_infos(self.names)

    def as_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "names": list(self.names),
            "transfer_stats": self.transfer_stats().as_dict(),
            "blocks": [info.as_dict() for info in self.block_infos()],
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

    def info(self) -> OffloadBlockInfo:
        return OffloadBlockInfo(
            name=self.name,
            block_id=self.block_id,
            cpu_slot=self.cpu_slot,
            gpu_slot=self.gpu_slot,
            cpu_offset=self.cpu_offset,
            gpu_offset=self.gpu_offset,
            bytes=self.bytes,
            state=self.state,
            last_operation=self.last_operation,
            transfer_stats=self.last_transfer_stats,
        )


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


def transfer_stats_from_receipt(receipt: TransferReceipt) -> TransferStats:
    direct_bytes = 0
    relay_bytes = 0
    direct_chunks = 0
    relay_chunks = 0
    for path in receipt.path_stats:
        bytes_count = int(path.get("bytes", 0) or 0)
        chunk_count = int(path.get("chunk_count", path.get("chunks", 0)) or 0)
        if str(path.get("kind", "")).lower() == "relay":
            relay_bytes += bytes_count
            relay_chunks += chunk_count
        else:
            direct_bytes += bytes_count
            direct_chunks += chunk_count
    return TransferStats(
        bytes=direct_bytes + relay_bytes,
        direct_chunks=direct_chunks,
        relay_chunks=relay_chunks,
    )


def _stat_value(stats, name: str) -> int:
    if isinstance(stats, dict):
        return int(stats.get(name, 0) or 0)
    return int(getattr(stats, name, 0) or 0)


class OffloadStore:
    """Connector-shaped named-block layer over daemon transfer intent."""

    def __init__(
        self,
        client: TransferIntentClient,
        transfer_context: AdapterTransferContext,
    ) -> None:
        if not isinstance(transfer_context, AdapterTransferContext):
            raise TypeError("transfer_context must be an AdapterTransferContext")
        self.client = client
        self.transfer_context = transfer_context
        self._blocks: dict[str, OffloadBlock] = {}
        self._intent_counter = 0

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
            gpu_tensor = object()
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

    def add_block(
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
        return self.add(
            name,
            cpu_tensor,
            gpu_tensor,
            block_id=block_id,
            cpu_slot=cpu_slot,
            gpu_slot=gpu_slot,
            cpu_offset=cpu_offset,
            gpu_offset=gpu_offset,
            byte_count=byte_count,
        )

    def remove(self, name: str) -> OffloadBlock:
        return self._blocks.pop(name)

    def remove_block(self, name: str) -> OffloadBlock:
        return self.remove(name)

    def block(self, name: str) -> OffloadBlock:
        try:
            return self._blocks[name]
        except KeyError as exc:
            raise KeyError(f"unknown offload block: {name}") from exc

    def get_block(self, name: str) -> OffloadBlock:
        return self.block(name)

    def names(self) -> list[str]:
        return list(self._blocks)

    def block_ids(self) -> list[object]:
        return [block.block_id for block in self._blocks.values()]

    def blocks(self) -> Iterable[OffloadBlock]:
        return self._blocks.values()

    def block_info(self, name: str) -> OffloadBlockInfo:
        return self.block(name).info()

    def block_infos(self, names: Iterable[str] | None = None) -> list[OffloadBlockInfo]:
        if names is None:
            return [block.info() for block in self._blocks.values()]
        return [self.block(name).info() for name in names]

    def prefetch(self, name: str):
        return self.submit_prefetch_many([name]).handles[0]

    def evict(self, name: str):
        return self.submit_evict_many([name]).handles[0]

    def prefetch_many(self, names: Iterable[str]) -> list:
        return list(self.submit_prefetch_many(names).handles)

    def submit_prefetch_many(self, names: Iterable[str]) -> OffloadBatch:
        return self._submit_many(names, "prefetch")

    def evict_many(self, names: Iterable[str]) -> list:
        return list(self.submit_evict_many(names).handles)

    def submit_evict_many(self, names: Iterable[str]) -> OffloadBatch:
        return self._submit_many(names, "evict")

    def _submit_many(self, names: Iterable[str], operation: str) -> OffloadBatch:
        blocks = [self.block(name) for name in names]
        if not blocks:
            return OffloadBatch(operation, (), (), self)
        if self._can_use_range_batch(blocks):
            ranges = self._ranges(blocks, operation)
            if operation == "prefetch":
                handle = self._submit_transfer(blocks, "prefetch", ranges)
                state = BlockState.PREFETCHING
            elif operation == "evict":
                handle = self._submit_transfer(blocks, "evict", ranges)
                state = BlockState.EVICTING
            else:
                raise ValueError(f"unknown offload operation: {operation}")
            self._record_many(blocks, handle, operation, state)
            handles = tuple(handle for _ in blocks)
        else:
            if operation == "prefetch":
                state = BlockState.PREFETCHING
            elif operation == "evict":
                state = BlockState.EVICTING
            else:
                raise ValueError(f"unknown offload operation: {operation}")
            handles = tuple(
                self._submit_transfer([block], operation, self._ranges([block], operation))
                for block in blocks
            )
            for block, handle in zip(blocks, handles):
                self._record_many([block], handle, operation, state)
        return OffloadBatch(operation, tuple(block.name for block in blocks), handles, self)

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

    def transfer_stats_many(self, names: Iterable[str]) -> TransferStats:
        return summarize_transfer_handles(
            block.last_handle
            for block in (self.block(name) for name in names)
            if block.last_handle is not None
        )

    def set_block_state(
        self,
        name: str,
        state: BlockState,
        *,
        clear_transfer_state: bool = False,
    ) -> OffloadBlock:
        block = self.block(name)
        block.state = state
        if clear_transfer_state:
            self.clear_block_transfer_state(name)
        return block

    def clear_block_transfer_state(self, name: str) -> OffloadBlock:
        block = self.block(name)
        block.last_prefetch = None
        block.last_evict = None
        block.last_handle = None
        block.last_operation = None
        return block

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

    def _submit_transfer(
        self,
        blocks: list[OffloadBlock],
        operation: str,
        ranges: Iterable[dict[str, int]],
    ) -> ReceiptTransferHandle:
        direction = _direction_for_operation(operation)
        ranges_tuple = tuple(dict(item) for item in ranges)
        total_bytes = sum(item["bytes"] for item in ranges_tuple)
        if direction == "h2d":
            source_buffer_id = self.transfer_context.cpu_buffer_id
            destination_buffer_id = self.transfer_context.gpu_buffer_id
        else:
            source_buffer_id = self.transfer_context.gpu_buffer_id
            destination_buffer_id = self.transfer_context.cpu_buffer_id
        metadata = {
            **self.transfer_context.metadata,
            "operation": operation,
            "block_names": [block.name for block in blocks],
        }
        intent = TransferIntent(
            intent_id=self._next_intent_id(operation),
            job_id=self.transfer_context.job_id,
            session_id=self.transfer_context.session_id,
            source_buffer_id=source_buffer_id,
            destination_buffer_id=destination_buffer_id,
            direction=direction,
            total_bytes=total_bytes,
            ranges=ranges_tuple,
            workload_kind=self.transfer_context.workload_kind,
            priority=self.transfer_context.priority,
            policy_hints=self.transfer_context.policy_hints,
            metadata=metadata,
        )
        receipt = self.client.submit_transfer_intent(intent)
        if not isinstance(receipt, TransferReceipt):
            raise TypeError("submit_transfer_intent must return a TransferReceipt")
        if receipt.intent_id != intent.intent_id:
            raise ValueError("receipt intent_id does not match transfer intent")
        return ReceiptTransferHandle(
            client=self.client,
            intent=intent,
            receipt=receipt,
            wait_timeout_seconds=self.transfer_context.wait_timeout_seconds,
        )

    def _next_intent_id(self, operation: str) -> str:
        self._intent_counter += 1
        return f"{self.transfer_context.intent_prefix}-{operation}-{self._intent_counter}"

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


def _direction_for_operation(operation: str) -> str:
    if operation == "prefetch":
        return "h2d"
    if operation == "evict":
        return "d2h"
    raise ValueError(f"unknown offload operation: {operation}")


def _require_non_empty(value: object, field_name: str) -> str:
    normalized = str(value)
    if not normalized.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_policy_hints_no_physical(value: Mapping[str, object]) -> dict[str, object]:
    policy_hints = dict(value)
    forbidden_keys = {
        "mode",
        "path",
        "paths",
        "route",
        "routes",
        "relay",
        "relays",
        "relay_gpu",
        "relay_gpus",
        "target_device",
        "target_gpu",
    }
    invalid_keys = sorted(
        key for key in policy_hints if str(key).lower() in forbidden_keys
    )
    if invalid_keys:
        raise ValueError(
            "policy_hints must not choose physical paths: "
            + ", ".join(str(key) for key in invalid_keys)
        )
    return policy_hints


OffloadManager = OffloadStore
KVBlockStore = OffloadStore
