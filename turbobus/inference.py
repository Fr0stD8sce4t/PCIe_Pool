from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .offload_store import OffloadStore, TransferStats
from .runtime import Runtime


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
        runtime: Runtime,
        cpu_backing,
        gpu_kv_backing,
    ) -> None:
        super().__init__(runtime)
        self.manager = self
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
        names, handles = self.submit_restore_prefix(names)
        self.wait_prefix(names)
        return handles

    def save_prefix(self, names: Iterable[str]) -> list:
        names, handles = self.submit_save_prefix(names)
        self.wait_prefix(names)
        return handles

    def submit_restore_prefix(self, names: Iterable[str]) -> tuple[list[str], list]:
        names = list(names)
        return names, self.prefetch_many(names)

    def submit_save_prefix(self, names: Iterable[str]) -> tuple[list[str], list]:
        names = list(names)
        return names, self.evict_many(names)

    def wait_prefix(self, names: Iterable[str]) -> None:
        self.wait_many(names)

    def transfer_stats(self, names: Iterable[str]) -> TransferStats:
        return self.transfer_stats_many(names)


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
