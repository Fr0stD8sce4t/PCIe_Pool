from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import turbobus
from framework_kv_slot_adapter import FrameworkKVSlot, FrameworkKVSlotAdapter


@dataclass(frozen=True)
class VllmKVBlockRef:
    """One vLLM KV block mapped to saved CPU backing and a GPU KV slot."""

    request_id: str
    group_id: int
    block_id: int
    cpu_slot: int
    gpu_slot: int


@dataclass(frozen=True)
class VllmKVGroup:
    """KV backing tensors and block size for one vLLM KV cache group."""

    group_id: int
    cpu_backing: object
    gpu_kv_backing: object
    block_bytes: int


class VllmKVSlotAdapter:
    """vLLM-shaped wrapper around the generic TurboBus KV slot adapter.

    This class intentionally does not import vLLM. A vLLM patch should extract
    block ids and KV cache tensors from the local vLLM version, then pass them
    into this adapter as plain Python objects.
    """

    def __init__(
        self,
        runtime: turbobus.Runtime,
        groups: Iterable[VllmKVGroup],
    ) -> None:
        self.runtime = runtime
        self.groups: dict[int, VllmKVGroup] = {group.group_id: group for group in groups}
        self.adapters = {
            group.group_id: FrameworkKVSlotAdapter(
                runtime,
                group.cpu_backing,
                group.gpu_kv_backing,
            )
            for group in self.groups.values()
        }

    def register_blocks(self, refs: Iterable[VllmKVBlockRef]) -> list[str]:
        slots_by_group: dict[int, list[FrameworkKVSlot]] = {}
        names = []
        for ref in refs:
            group = self.groups[ref.group_id]
            name = block_name(ref)
            names.append(name)
            slots_by_group.setdefault(ref.group_id, []).append(
                FrameworkKVSlot(
                    name=name,
                    block_id=(ref.request_id, ref.group_id, ref.block_id),
                    cpu_slot=ref.cpu_slot,
                    gpu_slot=ref.gpu_slot,
                    cpu_offset=ref.cpu_slot * group.block_bytes,
                    gpu_offset=ref.gpu_slot * group.block_bytes,
                    byte_count=group.block_bytes,
                )
            )

        for group_id, slots in slots_by_group.items():
            self.adapters[group_id].register_slots(slots)
        return names

    def restore_prefix(self, refs: Iterable[VllmKVBlockRef]) -> None:
        names_by_group = self._register_and_group(refs)
        for group_id, names in names_by_group.items():
            self.adapters[group_id].restore_prefix(names)

    def save_prefix(self, refs: Iterable[VllmKVBlockRef]) -> None:
        names_by_group = self._register_and_group(refs)
        for group_id, names in names_by_group.items():
            self.adapters[group_id].save_prefix(names)

    def _register_and_group(
        self,
        refs: Iterable[VllmKVBlockRef],
    ) -> Mapping[int, list[str]]:
        refs = list(refs)
        self.register_blocks(refs)
        names_by_group: dict[int, list[str]] = {}
        for ref in refs:
            names_by_group.setdefault(ref.group_id, []).append(block_name(ref))
        return names_by_group


def block_name(ref: VllmKVBlockRef) -> str:
    return f"{ref.request_id}:g{ref.group_id}:b{ref.block_id}"


def make_block_refs_from_ids(
    request_id: str,
    group_id: int,
    block_ids: Iterable[int],
    cpu_slot_start: int = 0,
) -> list[VllmKVBlockRef]:
    refs = []
    for index, block_id in enumerate(block_ids):
        refs.append(
            VllmKVBlockRef(
                request_id=request_id,
                group_id=group_id,
                block_id=int(block_id),
                cpu_slot=cpu_slot_start + index,
                gpu_slot=int(block_id),
            )
        )
    return refs


def main() -> None:
    raise SystemExit(
        "This is a vLLM adapter sketch. Import VllmKVSlotAdapter from a "
        "vLLM-specific POC after extracting vLLM KV cache tensors and block ids."
    )


if __name__ == "__main__":
    main()
