from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .runtime import Runtime
from .vllm import (
    VllmKVBlockRef,
    VllmKVSlotAdapter,
    block_bytes_from_vllm_kv_tensor,
    make_vllm_layer_block_refs_from_ids,
    make_vllm_layer_groups_from_kv_caches,
)


@dataclass(frozen=True)
class VllmAllocationEvent:
    """Block ids that vLLM allocated for one request."""

    request_id: str
    block_ids_by_group: tuple[tuple[int, ...], ...]
    event_count: int = 1

    @property
    def block_ids(self) -> tuple[int, ...]:
        seen = set()
        ordered = []
        for group_ids in self.block_ids_by_group:
            for block_id in group_ids:
                if block_id not in seen:
                    seen.add(block_id)
                    ordered.append(block_id)
        return tuple(ordered)

    def merge(self, other: "VllmAllocationEvent") -> "VllmAllocationEvent":
        if other.request_id != self.request_id:
            raise ValueError("cannot merge allocation events for different requests")
        group_count = max(len(self.block_ids_by_group), len(other.block_ids_by_group))
        merged = []
        for group_index in range(group_count):
            left = (
                self.block_ids_by_group[group_index]
                if group_index < len(self.block_ids_by_group)
                else tuple()
            )
            right = (
                other.block_ids_by_group[group_index]
                if group_index < len(other.block_ids_by_group)
                else tuple()
            )
            seen = set()
            ordered = []
            for block_id in (*left, *right):
                if block_id not in seen:
                    seen.add(block_id)
                    ordered.append(block_id)
            merged.append(tuple(ordered))
        return VllmAllocationEvent(
            self.request_id,
            tuple(merged),
            event_count=self.event_count + other.event_count,
        )


AllocationCallback = Callable[
    ["VllmTurboBusIntegration", object, object, VllmAllocationEvent],
    None,
]


@dataclass
class VllmIntegrationState:
    """Runtime state observed from a real vLLM process."""

    kv_cache_config: object | None = None
    kv_caches: list[object] = field(default_factory=list)
    allocations: dict[str, VllmAllocationEvent] = field(default_factory=dict)
    adapter: VllmKVSlotAdapter | None = None


class VllmTurboBusIntegration:
    """Narrow TurboBus data-path hook for vLLM-owned KV cache slots.

    vLLM still owns scheduling, request state, and GPU KV allocation. This hook
    observes the real vLLM KV tensors and block ids, then maps those slots to
    TurboBus restore/save operations.
    """

    def __init__(
        self,
        runtime: Runtime,
        cpu_backings: Iterable | None = None,
    ) -> None:
        self.runtime = runtime
        self.state = VllmIntegrationState()
        self._cpu_backings = list(cpu_backings) if cpu_backings is not None else None
        self._allocation_callback: AllocationCallback | None = None

    def install(self) -> None:
        """Install hooks into the imported vLLM V1 classes."""

        from vllm.v1.core import kv_cache_manager as manager_module
        from vllm.v1.worker import gpu_model_runner as runner_module

        self.install_on_classes(
            runner_module.GPUModelRunner,
            manager_module.KVCacheManager,
        )

    def install_on_classes(self, runner_cls, manager_cls) -> None:
        """Install hooks on explicit classes.

        This method exists so tests and version-specific integration code can
        patch the exact classes used by the active vLLM build.
        """

        runner_cls._turbobus_integration = self
        manager_cls._turbobus_integration = self

        if not hasattr(runner_cls, "_turbobus_original_initialize_kv_cache"):
            runner_cls._turbobus_original_initialize_kv_cache = runner_cls.initialize_kv_cache
            original_initialize = runner_cls.initialize_kv_cache

            @functools.wraps(original_initialize)
            def wrapped_initialize(runner, kv_cache_config, *args, **kwargs):
                result = original_initialize(runner, kv_cache_config, *args, **kwargs)
                integration = getattr(type(runner), "_turbobus_integration", None)
                if integration is not None:
                    integration.bind_runner(runner, kv_cache_config)
                return result

            runner_cls.initialize_kv_cache = wrapped_initialize

        if not hasattr(manager_cls, "_turbobus_original_allocate_slots"):
            manager_cls._turbobus_original_allocate_slots = manager_cls.allocate_slots
            original_allocate = manager_cls.allocate_slots

            @functools.wraps(original_allocate)
            def wrapped_allocate(manager, request, *args, **kwargs):
                result = original_allocate(manager, request, *args, **kwargs)
                integration = getattr(type(manager), "_turbobus_integration", None)
                if integration is not None:
                    integration.handle_allocation(request, result)
                return result

            manager_cls.allocate_slots = wrapped_allocate

    def bind_runner(self, runner, kv_cache_config=None) -> None:
        kv_caches = list(getattr(runner, "kv_caches", []) or [])
        self.bind_kv_caches(kv_caches, kv_cache_config)

    def bind_kv_caches(self, kv_caches: Iterable, kv_cache_config=None) -> None:
        self.state.kv_cache_config = kv_cache_config
        self.state.kv_caches = list(kv_caches)
        self._refresh_adapter()

    def set_cpu_backings(self, cpu_backings: Iterable) -> None:
        self._cpu_backings = list(cpu_backings)
        self._refresh_adapter()

    def set_allocation_callback(self, callback: AllocationCallback | None) -> None:
        self._allocation_callback = callback

    def allocate_cpu_backings(self, slots_per_layer: int, *, pin_memory: bool = True) -> list:
        """Allocate pinned CPU byte buffers for the observed vLLM layer caches."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - import-time convenience only
            raise RuntimeError("PyTorch is required to allocate vLLM CPU backings") from exc

        backings = []
        for kv_cache in self.state.kv_caches:
            block_bytes = block_bytes_from_vllm_kv_tensor(kv_cache)
            backings.append(
                torch.empty(
                    slots_per_layer * block_bytes,
                    dtype=torch.uint8,
                    pin_memory=pin_memory,
                )
            )
        self.set_cpu_backings(backings)
        return backings

    def record_allocation(self, request, blocks) -> VllmAllocationEvent | None:
        request_id = str(getattr(request, "request_id", "unknown"))
        block_ids_by_group = extract_vllm_block_ids(blocks)
        if not block_ids_by_group:
            return None
        event = VllmAllocationEvent(request_id, block_ids_by_group)
        previous = self.state.allocations.get(request_id)
        if previous is not None:
            event = previous.merge(event)
        self.state.allocations[request_id] = event
        return event

    def handle_allocation(self, request, blocks) -> VllmAllocationEvent | None:
        event = self.record_allocation(request, blocks)
        if event is not None and self._allocation_callback is not None:
            self._allocation_callback(self, request, blocks, event)
        return event

    def block_ids_for_request(self, request_id: str) -> tuple[int, ...]:
        event = self.state.allocations[str(request_id)]
        return event.block_ids

    def make_refs_for_request(
        self,
        request_id: str,
        *,
        cpu_slot_start: int = 0,
    ) -> list[VllmKVBlockRef]:
        block_ids = self.block_ids_for_request(request_id)
        return make_vllm_layer_block_refs_from_ids(
            str(request_id),
            block_ids,
            layer_count=len(self.state.kv_caches),
            cpu_slot_start=cpu_slot_start,
        )

    def restore_request_prefix(self, request_id: str, *, cpu_slot_start: int = 0) -> list:
        adapter = self.require_adapter()
        return adapter.restore_prefix(
            self.make_refs_for_request(request_id, cpu_slot_start=cpu_slot_start)
        )

    def save_request_prefix(self, request_id: str, *, cpu_slot_start: int = 0) -> list:
        adapter = self.require_adapter()
        return adapter.save_prefix(
            self.make_refs_for_request(request_id, cpu_slot_start=cpu_slot_start)
        )

    def _refresh_adapter(self) -> None:
        if not self.state.kv_caches or self._cpu_backings is None:
            self.state.adapter = None
            return
        if len(self._cpu_backings) != len(self.state.kv_caches):
            raise ValueError("cpu_backings must match the number of vLLM KV cache tensors")
        groups = make_vllm_layer_groups_from_kv_caches(
            self._cpu_backings,
            self.state.kv_caches,
        )
        self.state.adapter = VllmKVSlotAdapter(self.runtime, groups)

    def require_adapter(self) -> VllmKVSlotAdapter:
        if self.state.adapter is None:
            raise RuntimeError("vLLM KV caches and CPU backings must be bound before restore/save")
        return self.state.adapter

    _require_adapter = require_adapter


def extract_vllm_block_ids(blocks) -> tuple[tuple[int, ...], ...]:
    if blocks is None:
        return tuple()
    get_block_ids = getattr(blocks, "get_block_ids", None)
    if get_block_ids is None:
        return tuple()
    try:
        raw = get_block_ids(allow_none=True)
    except TypeError:
        raw = get_block_ids()
    if raw is None:
        return tuple()
    groups = []
    for group_ids in raw:
        if group_ids is None:
            groups.append(tuple())
        else:
            groups.append(tuple(int(block_id) for block_id in group_ids if block_id is not None))
    return tuple(groups)
