from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .offload_store import OffloadManager
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


class InferenceKVSlotAdapter:
    """Register framework KV slots and restore/save them through TurboBus."""

    def __init__(
        self,
        runtime: Runtime,
        cpu_backing,
        gpu_kv_backing,
    ) -> None:
        self.manager = OffloadManager(runtime)
        self.cpu_backing = cpu_backing
        self.gpu_kv_backing = gpu_kv_backing

    def register_slots(self, slots: Iterable[InferenceKVSlot]) -> None:
        for slot in slots:
            self.manager.add(
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
        names = list(names)
        handles = self.manager.prefetch_many(names)
        self.manager.wait_many(names)
        return handles

    def save_prefix(self, names: Iterable[str]) -> list:
        names = list(names)
        handles = self.manager.evict_many(names)
        self.manager.wait_many(names)
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
