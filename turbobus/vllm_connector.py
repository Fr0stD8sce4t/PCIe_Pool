from __future__ import annotations

from dataclasses import dataclass, field
import time

from .vllm import make_vllm_layer_range_refs_from_ids
from .vllm_integration import VllmAllocationEvent, VllmTurboBusIntegration


@dataclass
class VllmConnectorEvent:
    request_id: str
    operation: str
    block_count: int
    elapsed_ms: float
    direct_chunks: int
    relay_chunks: int
    bytes: int


@dataclass
class VllmTurboBusConnector:
    """Run TurboBus restore/save from real vLLM allocation hooks.

    vLLM still owns scheduling and block allocation. This connector supplies a
    narrow data path: save selected KV blocks into pinned CPU backing after a
    request, then restore those bytes inside a later vLLM `allocate_slots()`
    call for the next real request.
    """

    integration: VllmTurboBusIntegration
    events: list[VllmConnectorEvent] = field(default_factory=list)
    _stored_block_count: int = 0
    _restore_next_blocks: int = 0

    def install(self) -> None:
        self.integration.set_allocation_callback(self._on_allocation)

    def allocate_cpu_backings_for_blocks(self, block_count: int) -> list:
        if not self.integration.state.kv_caches:
            raise RuntimeError("vLLM KV caches must be bound before allocating CPU backing")
        slots_per_layer = max(1, block_count * self._max_lanes_per_layer())
        return self.integration.allocate_cpu_backings(slots_per_layer)

    def save_request(self, request_id: str, block_count: int) -> VllmConnectorEvent:
        refs = self._refs_for_request(request_id, block_count)
        start = time.perf_counter()
        handles = self.integration.require_adapter().save_prefix(refs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        event = self._event(request_id, "save", block_count, elapsed_ms, handles)
        self.events.append(event)
        self._stored_block_count = block_count
        return event

    def restore_next_allocation(self, block_count: int | None = None) -> None:
        if self._stored_block_count <= 0:
            raise RuntimeError("save_request() must run before restore_next_allocation()")
        requested = block_count if block_count is not None else self._stored_block_count
        self._restore_next_blocks = min(int(requested), self._stored_block_count)

    def _on_allocation(
        self,
        integration: VllmTurboBusIntegration,
        request,
        blocks,
        event: VllmAllocationEvent,
    ) -> None:
        if self._restore_next_blocks <= 0:
            return
        block_ids = event.block_ids[: self._restore_next_blocks]
        if len(block_ids) < self._restore_next_blocks:
            return
        refs = make_vllm_layer_range_refs_from_ids(
            event.request_id,
            block_ids,
            integration.state.kv_caches,
        )
        start = time.perf_counter()
        handles = integration.require_adapter().restore_prefix(refs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.events.append(
            self._event(event.request_id, "restore", len(block_ids), elapsed_ms, handles)
        )
        self._restore_next_blocks = 0

    def _refs_for_request(self, request_id: str, block_count: int):
        block_ids = self.integration.block_ids_for_request(request_id)[:block_count]
        if len(block_ids) < block_count:
            raise RuntimeError(
                f"request {request_id} has {len(block_ids)} blocks, need {block_count}"
            )
        return make_vllm_layer_range_refs_from_ids(
            request_id,
            block_ids,
            self.integration.state.kv_caches,
        )

    def _max_lanes_per_layer(self) -> int:
        return max(
            (
                int(kv_cache.shape[0]) if len(kv_cache.shape) >= 3 else 1
                for kv_cache in self.integration.state.kv_caches
            ),
            default=1,
        )

    @staticmethod
    def _event(
        request_id: str,
        operation: str,
        block_count: int,
        elapsed_ms: float,
        handles: list,
    ) -> VllmConnectorEvent:
        unique = []
        seen = set()
        for handle in handles:
            if id(handle) in seen or handle.stats is None:
                continue
            seen.add(id(handle))
            unique.append(handle.stats)
        return VllmConnectorEvent(
            request_id=request_id,
            operation=operation,
            block_count=block_count,
            elapsed_ms=elapsed_ms,
            direct_chunks=sum(stats.direct_chunks for stats in unique),
            relay_chunks=sum(stats.relay_chunks for stats in unique),
            bytes=sum(stats.bytes for stats in unique),
        )
