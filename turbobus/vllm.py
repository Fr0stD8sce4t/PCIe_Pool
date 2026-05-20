from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .inference import InferenceKVSlot, InferenceKVSlotAdapter
from .runtime import Runtime


@dataclass(frozen=True)
class VllmKVBlockRef:
    """One vLLM KV block mapped to saved CPU backing and a GPU KV slot."""

    request_id: str
    group_id: int
    block_id: int
    cpu_slot: int
    gpu_slot: int
    lane_id: int | None = None
    cpu_offset: int | None = None
    gpu_offset: int | None = None


@dataclass(frozen=True)
class VllmKVGroup:
    """KV backing tensors and block size for one vLLM KV cache group."""

    group_id: int
    cpu_backing: object
    gpu_kv_backing: object
    block_bytes: int
    layer_id: int | None = None


class VllmKVSlotAdapter:
    """vLLM-shaped wrapper around TurboBus inference KV slot adapters."""

    def __init__(
        self,
        runtime: Runtime,
        groups: Iterable[VllmKVGroup],
    ) -> None:
        self.runtime = runtime
        self.groups: dict[int, VllmKVGroup] = {group.group_id: group for group in groups}
        self.adapters = {
            group.group_id: InferenceKVSlotAdapter(
                runtime,
                group.cpu_backing,
                group.gpu_kv_backing,
            )
            for group in self.groups.values()
        }
        self._registered_names: set[str] = set()

    def register_blocks(self, refs: Iterable[VllmKVBlockRef]) -> list[str]:
        slots_by_group: dict[int, list[InferenceKVSlot]] = {}
        names = []
        for ref in refs:
            group = self.groups[ref.group_id]
            name = vllm_block_name(ref)
            names.append(name)
            if name in self._registered_names:
                continue
            slots_by_group.setdefault(ref.group_id, []).append(
                InferenceKVSlot(
                    name=name,
                    block_id=(ref.request_id, ref.group_id, ref.block_id),
                    cpu_slot=ref.cpu_slot,
                    gpu_slot=ref.gpu_slot,
                    cpu_offset=(
                        ref.cpu_offset
                        if ref.cpu_offset is not None
                        else ref.cpu_slot * group.block_bytes
                    ),
                    gpu_offset=(
                        ref.gpu_offset
                        if ref.gpu_offset is not None
                        else ref.gpu_slot * group.block_bytes
                    ),
                    byte_count=group.block_bytes,
                )
            )

        for group_id, slots in slots_by_group.items():
            self.adapters[group_id].register_slots(slots)
            for slot in slots:
                self._registered_names.add(slot.name)
        return names

    def restore_prefix(self, refs: Iterable[VllmKVBlockRef]) -> list:
        names_by_group = self._register_and_group(refs)
        handles = []
        for group_id, names in names_by_group.items():
            handles.extend(self.adapters[group_id].restore_prefix(names))
        return handles

    def save_prefix(self, refs: Iterable[VllmKVBlockRef]) -> list:
        names_by_group = self._register_and_group(refs)
        handles = []
        for group_id, names in names_by_group.items():
            handles.extend(self.adapters[group_id].save_prefix(names))
        return handles

    def _register_and_group(
        self,
        refs: Iterable[VllmKVBlockRef],
    ) -> Mapping[int, list[str]]:
        refs = list(refs)
        self.register_blocks(refs)
        names_by_group: dict[int, list[str]] = {}
        for ref in refs:
            names_by_group.setdefault(ref.group_id, []).append(vllm_block_name(ref))
        return names_by_group


def vllm_block_name(ref: VllmKVBlockRef) -> str:
    lane = "" if ref.lane_id is None else f":l{ref.lane_id}"
    return f"{ref.request_id}:g{ref.group_id}:b{ref.block_id}{lane}"


def make_vllm_block_refs_from_ids(
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


def block_bytes_from_vllm_kv_tensor(tensor) -> int:
    """Return bytes for one vLLM KV block in a tensor shaped like [*, blocks, ...]."""

    if len(tensor.shape) < 2:
        raise ValueError("vLLM KV tensor must have at least two dimensions")
    return int(tensor.stride(1) * tensor.element_size())


def make_vllm_layer_groups_from_kv_caches(
    cpu_backings: Iterable,
    kv_caches: Iterable,
    *,
    group_id_start: int = 0,
) -> list[VllmKVGroup]:
    """Create one TurboBus group for each vLLM layer KV cache tensor."""

    groups = []
    for layer_offset, (cpu_backing, kv_cache) in enumerate(zip(cpu_backings, kv_caches)):
        groups.append(
            VllmKVGroup(
                group_id=group_id_start + layer_offset,
                layer_id=layer_offset,
                cpu_backing=cpu_backing,
                gpu_kv_backing=kv_cache,
                block_bytes=block_bytes_from_vllm_kv_tensor(kv_cache),
            )
        )
    return groups


def make_vllm_layer_block_refs_from_ids(
    request_id: str,
    block_ids: Iterable[int],
    layer_count: int,
    cpu_slot_start: int = 0,
) -> list[VllmKVBlockRef]:
    refs = []
    block_ids = [int(block_id) for block_id in block_ids]
    for layer_id in range(layer_count):
        for index, block_id in enumerate(block_ids):
            refs.append(
                VllmKVBlockRef(
                    request_id=request_id,
                    group_id=layer_id,
                    block_id=block_id,
                    cpu_slot=cpu_slot_start + index,
                    gpu_slot=block_id,
                )
            )
    return refs


def make_vllm_layer_range_refs_from_ids(
    request_id: str,
    block_ids: Iterable[int],
    kv_caches: Iterable,
    cpu_slot_start: int = 0,
) -> list[VllmKVBlockRef]:
    """Create byte-range refs for vLLM tensors shaped like [kv, blocks, ...].

    vLLM commonly stores K and V in dimension 0 and block id in dimension 1.
    The K and V ranges for the same block are not necessarily contiguous, so a
    logical KV block can become more than one TurboBus byte range.
    """

    refs = []
    block_ids = [int(block_id) for block_id in block_ids]
    for layer_id, kv_cache in enumerate(kv_caches):
        lane_count = int(kv_cache.shape[0]) if len(kv_cache.shape) >= 3 else 1
        block_bytes = block_bytes_from_vllm_kv_tensor(kv_cache)
        for index, block_id in enumerate(block_ids):
            for lane_id in range(lane_count):
                cpu_slot = cpu_slot_start + index * lane_count + lane_id
                if lane_count == 1:
                    gpu_offset = block_id * block_bytes
                    lane = None
                else:
                    gpu_offset = int(
                        (lane_id * kv_cache.stride(0) + block_id * kv_cache.stride(1))
                        * kv_cache.element_size()
                    )
                    lane = lane_id
                refs.append(
                    VllmKVBlockRef(
                        request_id=request_id,
                        group_id=layer_id,
                        block_id=block_id,
                        cpu_slot=cpu_slot,
                        gpu_slot=block_id,
                        lane_id=lane,
                        gpu_offset=gpu_offset,
                    )
                )
    return refs


block_name = vllm_block_name
make_block_refs_from_ids = make_vllm_block_refs_from_ids
make_layer_groups_from_kv_caches = make_vllm_layer_groups_from_kv_caches
make_layer_block_refs_from_ids = make_vllm_layer_block_refs_from_ids
make_layer_range_refs_from_ids = make_vllm_layer_range_refs_from_ids
