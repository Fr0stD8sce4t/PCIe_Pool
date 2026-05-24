from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..offload_store import AdapterTransferContext, OffloadBatch, OffloadStore, TransferStats


@dataclass(frozen=True)
class InferenceKVSlot:
    """One inference-framework-owned KV block slot."""

    name: str
    block_id: object
    cpu_offset: int
    gpu_offset: int
    byte_count: int
    cpu_slot: object | None = None
    gpu_slot: object | None = None


class InferenceKVSlotAdapter(OffloadStore):
    """Register framework KV slots and restore/save them through TurboBus."""

    def __init__(
        self,
        client,
        transfer_context: AdapterTransferContext,
        cpu_backing,
        gpu_kv_backing,
    ) -> None:
        super().__init__(client, transfer_context)
        self.cpu_backing = cpu_backing
        self.gpu_kv_backing = gpu_kv_backing

    def register_slots(self, slots: Iterable[InferenceKVSlot]) -> None:
        for slot in slots:
            self.add(
                slot.name,
                self.cpu_backing,
                self.gpu_kv_backing,
                block_id=slot.block_id,
                cpu_slot=slot.cpu_slot,
                gpu_slot=slot.gpu_slot,
                cpu_offset=slot.cpu_offset,
                gpu_offset=slot.gpu_offset,
                byte_count=slot.byte_count,
            )

    def restore_prefix(self, names: Iterable[str]) -> list:
        return self._run_prefix_transfer(names, self.submit_restore_prefix)

    def save_prefix(self, names: Iterable[str]) -> list:
        return self._run_prefix_transfer(names, self.submit_save_prefix)

    def submit_restore_prefix(self, names: Iterable[str]) -> tuple[list[str], list]:
        names = list(names)
        return names, self.prefetch_many(names)

    def restore_batch(self, names: Iterable[str]) -> OffloadBatch:
        names = list(names)
        return OffloadBatch(
            "restore",
            tuple(names),
            tuple(self.prefetch_many(names)),
            self,
        )

    def submit_save_prefix(self, names: Iterable[str]) -> tuple[list[str], list]:
        names = list(names)
        return names, self.evict_many(names)

    def save_batch(self, names: Iterable[str]) -> OffloadBatch:
        names = list(names)
        return OffloadBatch(
            "save",
            tuple(names),
            tuple(self.evict_many(names)),
            self,
        )

    def wait_prefix(self, names: Iterable[str]) -> None:
        self.wait_many(names)

    def transfer_stats(self, names: Iterable[str]) -> TransferStats:
        return self.transfer_stats_many(names)

    def _run_prefix_transfer(self, names: Iterable[str], submit) -> list:
        names, handles = submit(names)
        self.wait_prefix(names)
        return handles


def make_contiguous_kv_slots(
    prefix: str,
    count: int,
    block_bytes: int,
) -> list[InferenceKVSlot]:
    return [
        InferenceKVSlot(
            name=f"{prefix}{index}",
            block_id=index,
            cpu_slot=index,
            gpu_slot=index,
            cpu_offset=index * block_bytes,
            gpu_offset=index * block_bytes,
            byte_count=block_bytes,
        )
        for index in range(count)
    ]


FrameworkKVSlot = InferenceKVSlot
FrameworkKVSlotAdapter = InferenceKVSlotAdapter
