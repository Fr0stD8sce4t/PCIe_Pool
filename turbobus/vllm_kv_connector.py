from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any

from .runtime import Runtime, RuntimeOptions
from .vllm import make_vllm_layer_range_refs_from_ids
from .vllm_integration import extract_vllm_block_ids

try:  # pragma: no cover - depends on an installed vLLM build
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
    except ImportError:
        SupportsHMA = object
except ImportError:  # pragma: no cover - lets unit tests import without vLLM
    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._vllm_config = vllm_config
            self._role = role
            self._connector_metadata = None

    class KVConnectorRole:
        SCHEDULER = "scheduler"
        WORKER = "worker"

    SupportsHMA = object


@dataclass
class TurboBusRequestMetadata:
    request_id: str
    prefix_key: str
    block_ids: tuple[int, ...]
    matched_tokens: int
    block_count: int
    cpu_slot_start: int = 0


@dataclass
class TurboBusSavedPrefix:
    key: str
    cpu_backings: list[Any]
    block_count: int
    matched_tokens: int
    source_request_id: str = ""
    bytes: int = 0
    elapsed_ms: float = 0.0
    direct_chunks: int = 0
    relay_chunks: int = 0


@dataclass
class _ScheduledRequestView:
    req_id: str
    new_block_ids: Any
    kv_transfer_params: dict[str, Any]


class TurboBusConnectorMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TurboBusRequestMetadata] = []
        self.save_requests: list[TurboBusRequestMetadata] = []

    def add_request(self, request: TurboBusRequestMetadata) -> None:
        self.requests.append(request)

    def add_save_request(self, request: TurboBusRequestMetadata) -> None:
        self.save_requests.append(request)

    def __len__(self) -> int:
        return len(self.requests) + len(self.save_requests)


@dataclass
class TurboBusKVConnectorState:
    kv_caches: dict[str, Any] = field(default_factory=dict)
    pending_loads: dict[str, TurboBusRequestMetadata] = field(default_factory=dict)
    pending_saves: dict[str, TurboBusRequestMetadata] = field(default_factory=dict)
    save_request_ids: set[str] = field(default_factory=set)
    saved_request_ids: set[str] = field(default_factory=set)
    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    events: list[dict[str, Any]] = field(default_factory=list)


_SAVED_PREFIXES: dict[str, TurboBusSavedPrefix] = {}


def register_saved_prefix(
    key: str,
    cpu_backings: list[Any],
    *,
    block_count: int,
    matched_tokens: int,
    source_request_id: str = "",
    bytes: int = 0,
    elapsed_ms: float = 0.0,
    direct_chunks: int = 0,
    relay_chunks: int = 0,
) -> None:
    if not key:
        raise ValueError("prefix key must not be empty")
    _SAVED_PREFIXES[str(key)] = TurboBusSavedPrefix(
        key=str(key),
        cpu_backings=list(cpu_backings),
        block_count=int(block_count),
        matched_tokens=int(matched_tokens),
        source_request_id=str(source_request_id),
        bytes=int(bytes),
        elapsed_ms=float(elapsed_ms),
        direct_chunks=int(direct_chunks),
        relay_chunks=int(relay_chunks),
    )
    _emit_event(
        "register_saved_prefix",
        prefix_key=str(key),
        block_count=int(block_count),
        matched_tokens=int(matched_tokens),
        source_request_id=str(source_request_id),
        layers=len(cpu_backings),
    )


def clear_saved_prefixes() -> None:
    _SAVED_PREFIXES.clear()


def get_saved_prefix(key: str) -> TurboBusSavedPrefix | None:
    return _SAVED_PREFIXES.get(str(key))


class TurboBusConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM KV connector entry point for TurboBus prefix restore.

    This uses vLLM's KV-transfer connector lifecycle instead of replacing the
    scheduler. A request opts in with `kv_transfer_params`:

    {
      "turbobus.do_restore": true,
      "turbobus.matched_tokens": 128
    }
    """

    def __init__(
        self,
        vllm_config,
        role: KVConnectorRole,
        kv_cache_config=None,
    ) -> None:
        try:
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        except TypeError:
            super().__init__(vllm_config=vllm_config, role=role)
        self.state = TurboBusKVConnectorState()
        self.vllm_block_size = int(getattr(vllm_config.cache_config, "block_size", 16))
        self.restore_block_limit = _extra_config_int(
            vllm_config,
            "turbobus.restore_block_limit",
            int(os.environ.get("TURBOBUS_RESTORE_BLOCK_LIMIT", "0") or 0),
        )
        self.restore_enabled = _extra_config_bool(
            vllm_config,
            "turbobus.restore_enabled",
            os.environ.get("TURBOBUS_RESTORE_ENABLED", "0") == "1",
        )
        self.runtime = None
        self._adapters_by_prefix: dict[str, Any] = {}
        _emit_event(
            "init",
            role=str(role),
            restore_enabled=self.restore_enabled,
            restore_block_limit=self.restore_block_limit,
        )

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.state.kv_caches = dict(kv_caches)
        self.state.events.append(
            {
                "event": "register_kv_caches",
                "layers": len(self.state.kv_caches),
            }
        )
        _emit_event("register_kv_caches", layers=len(self.state.kv_caches))

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        params = _request_params(request)
        if not params.get("turbobus.do_restore"):
            return 0, False
        if not self.restore_enabled:
            self.state.events.append(
                {
                    "event": "match_skipped",
                    "request_id": str(getattr(request, "request_id", "unknown")),
                    "restore_enabled": False,
                }
            )
            _emit_event(
                "match_skipped",
                request_id=str(getattr(request, "request_id", "unknown")),
                restore_enabled=False,
            )
            return 0, False
        prefix_key = _request_prefix_key(params)
        saved = get_saved_prefix(prefix_key)
        if saved is None:
            self.state.events.append(
                {
                    "event": "match_miss",
                    "request_id": str(getattr(request, "request_id", "unknown")),
                    "prefix_key": prefix_key,
                }
            )
            _emit_event(
                "match_miss",
                request_id=str(getattr(request, "request_id", "unknown")),
                prefix_key=prefix_key,
            )
            return 0, False
        matched_tokens = int(params.get("turbobus.matched_tokens", 0))
        if matched_tokens <= 0:
            matched_tokens = saved.matched_tokens
        matched_tokens = min(matched_tokens, saved.matched_tokens)
        if matched_tokens <= num_computed_tokens:
            return 0, False
        available = matched_tokens - int(num_computed_tokens)
        if available == matched_tokens and available == int(getattr(request, "num_tokens", 0)):
            available -= 1
        self.state.events.append(
            {
                "event": "match",
                "request_id": str(getattr(request, "request_id", "unknown")),
                "prefix_key": prefix_key,
                "matched_tokens": matched_tokens,
                "num_computed_tokens": int(num_computed_tokens),
                "available_tokens": max(0, available),
            }
        )
        _emit_event(
            "match",
            request_id=str(getattr(request, "request_id", "unknown")),
            prefix_key=prefix_key,
            matched_tokens=matched_tokens,
            num_computed_tokens=int(num_computed_tokens),
            available_tokens=max(0, available),
        )
        return max(0, available), available > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        params = _request_params(request)
        if params.get("turbobus.do_save"):
            self._update_save_state_after_alloc(request, blocks, params)
        if num_external_tokens <= 0:
            return
        prefix_key = _request_prefix_key(params)
        saved = get_saved_prefix(prefix_key)
        if saved is None:
            _emit_event(
                "alloc_miss",
                request_id=str(getattr(request, "request_id", "unknown")),
                prefix_key=prefix_key,
            )
            return
        block_ids = _flatten_block_ids(extract_vllm_block_ids(blocks))
        if not block_ids:
            return
        block_count = _block_count_for_tokens(num_external_tokens, self.vllm_block_size)
        if self.restore_block_limit > 0:
            block_count = min(block_count, self.restore_block_limit)
        block_count = min(block_count, saved.block_count)
        block_ids = block_ids[:block_count]
        request_id = str(getattr(request, "request_id", "unknown"))
        meta = TurboBusRequestMetadata(
            request_id=request_id,
            prefix_key=prefix_key,
            block_ids=tuple(block_ids),
            matched_tokens=int(num_external_tokens),
            block_count=len(block_ids),
        )
        self.state.pending_loads[request_id] = meta
        self.state.events.append(
            {
                "event": "alloc",
                "request_id": request_id,
                "prefix_key": prefix_key,
                "matched_tokens": int(num_external_tokens),
                "block_count": len(block_ids),
            }
        )
        _emit_event(
            "alloc",
            request_id=request_id,
            prefix_key=prefix_key,
            matched_tokens=int(num_external_tokens),
            block_count=len(block_ids),
        )

    def build_connector_meta(self, scheduler_output) -> TurboBusConnectorMetadata:
        self._collect_save_requests_from_scheduler_output(scheduler_output)
        metadata = TurboBusConnectorMetadata()
        for request_id in sorted(self.state.pending_loads):
            metadata.add_request(self.state.pending_loads[request_id])
        for request_id in sorted(self.state.pending_saves):
            metadata.add_save_request(self.state.pending_saves[request_id])
        if len(metadata) > 0:
            _emit_event(
                "build_connector_meta",
                requests=len(metadata),
                loads=len(metadata.requests),
                saves=len(metadata.save_requests),
            )
        self.state.pending_loads.clear()
        self.state.pending_saves.clear()
        return metadata

    def start_load_kv(self, forward_context, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, TurboBusConnectorMetadata) or not metadata.requests:
            return
        start = time.perf_counter()
        if not self.restore_enabled:
            for request in metadata.requests:
                self.state.finished_recving.add(request.request_id)
                self.state.events.append(
                    {
                        "event": "load_ready",
                        "request_id": request.request_id,
                        "block_count": len(request.block_ids),
                        "restore_enabled": False,
                    }
                )
                _emit_event(
                    "load_ready",
                    request_id=request.request_id,
                    block_count=len(request.block_ids),
                    restore_enabled=False,
                )
        else:
            for request in metadata.requests:
                self._restore_request(request)
                self.state.finished_recving.add(request.request_id)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _emit_event(
            "start_load_done",
            requests=len(metadata.requests),
            restore_enabled=self.restore_enabled,
            elapsed_ms=f"{elapsed_ms:.3f}",
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        return None

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, TurboBusConnectorMetadata) or not metadata.save_requests:
            return None
        for request in metadata.save_requests:
            self._save_request(request)
        return None

    def get_finished(self, finished_req_ids: set[str]):
        finished_sending = self.state.finished_sending
        if finished_req_ids:
            finished_sending = finished_sending | (
                self.state.saved_request_ids & set(finished_req_ids)
            )
            self.state.saved_request_ids -= finished_sending
        self.state.finished_sending -= finished_sending
        finished_recving = self.state.finished_recving
        self.state.finished_recving = set()
        return finished_sending or None, finished_recving or None

    def request_finished(self, request, block_ids: list[int]):
        params = _request_params(request)
        if not params.get("turbobus.do_save"):
            return False, None
        request_id = str(getattr(request, "request_id", "unknown"))
        if request_id not in self.state.save_request_ids:
            return False, None
        prefix_key = _request_prefix_key(params)
        matched_tokens = _matched_tokens_for_save(params, self.vllm_block_size)
        return True, {
            "turbobus.prefix_key": prefix_key,
            "turbobus.matched_tokens": matched_tokens,
        }

    def request_finished_all_groups(self, request, block_ids):
        flat_block_ids = [
            block_id
            for group_block_ids in block_ids
            for block_id in group_block_ids
        ]
        return self.request_finished(request, flat_block_ids)

    def _get_connector_metadata(self):
        return getattr(self, "_connector_metadata", None)

    def _update_save_state_after_alloc(
        self,
        request,
        blocks,
        params: dict[str, Any],
    ) -> None:
        request_id = str(getattr(request, "request_id", "unknown"))
        if request_id in self.state.save_request_ids:
            return
        block_ids = _flatten_block_ids(extract_vllm_block_ids(blocks))
        if not block_ids:
            return
        requested_blocks = _save_block_count(params, self.vllm_block_size)
        if requested_blocks <= 0:
            requested_blocks = len(block_ids)
        if len(block_ids) < requested_blocks:
            self.state.events.append(
                {
                    "event": "save_waiting",
                    "request_id": request_id,
                    "available_blocks": len(block_ids),
                    "requested_blocks": requested_blocks,
                }
            )
            _emit_event(
                "save_waiting",
                request_id=request_id,
                available_blocks=len(block_ids),
                requested_blocks=requested_blocks,
            )
            return
        block_ids = block_ids[:requested_blocks]
        meta = TurboBusRequestMetadata(
            request_id=request_id,
            prefix_key=_request_prefix_key(params),
            block_ids=tuple(block_ids),
            matched_tokens=_matched_tokens_for_save(params, self.vllm_block_size),
            block_count=len(block_ids),
        )
        self.state.pending_saves[request_id] = meta
        self.state.save_request_ids.add(request_id)
        self.state.events.append(
            {
                "event": "save_alloc",
                "request_id": request_id,
                "prefix_key": meta.prefix_key,
                "matched_tokens": meta.matched_tokens,
                "block_count": meta.block_count,
            }
        )
        _emit_event(
            "save_alloc",
            request_id=request_id,
            prefix_key=meta.prefix_key,
            matched_tokens=meta.matched_tokens,
            block_count=meta.block_count,
        )

    def _collect_save_requests_from_scheduler_output(self, scheduler_output) -> None:
        for request in _iter_scheduled_requests(scheduler_output):
            params = _request_params(request)
            if not params.get("turbobus.do_save"):
                continue
            request_id = _scheduled_request_id(request)
            if request_id in self.state.save_request_ids:
                continue
            block_ids = _scheduled_request_block_ids(request)
            if not block_ids:
                continue
            requested_blocks = _save_block_count(params, self.vllm_block_size)
            if requested_blocks <= 0:
                requested_blocks = len(block_ids)
            if len(block_ids) < requested_blocks:
                continue
            block_ids = block_ids[:requested_blocks]
            meta = TurboBusRequestMetadata(
                request_id=request_id,
                prefix_key=_request_prefix_key(params),
                block_ids=tuple(block_ids),
                matched_tokens=_matched_tokens_for_save(params, self.vllm_block_size),
                block_count=len(block_ids),
            )
            self.state.pending_saves[request_id] = meta
            self.state.save_request_ids.add(request_id)
            self.state.events.append(
                {
                    "event": "save_schedule",
                    "request_id": request_id,
                    "prefix_key": meta.prefix_key,
                    "matched_tokens": meta.matched_tokens,
                    "block_count": meta.block_count,
                }
            )
            _emit_event(
                "save_schedule",
                request_id=request_id,
                prefix_key=meta.prefix_key,
                matched_tokens=meta.matched_tokens,
                block_count=meta.block_count,
            )

    def _adapter_for_saved_prefix(self, saved: TurboBusSavedPrefix):
        adapter = self._adapters_by_prefix.get(saved.key)
        if adapter is not None:
            return adapter
        if self.runtime is None:
            self.runtime = _make_runtime_from_config(self._vllm_config)
        if not self.state.kv_caches:
            raise RuntimeError("vLLM did not register KV caches for TurboBus")
        from .vllm import VllmKVSlotAdapter
        from .vllm import make_vllm_layer_groups_from_kv_caches

        kv_caches = list(self.state.kv_caches.values())
        if len(saved.cpu_backings) != len(kv_caches):
            raise RuntimeError(
                f"saved prefix {saved.key!r} has {len(saved.cpu_backings)} backing tensors, "
                f"but vLLM registered {len(kv_caches)} KV cache tensors"
            )
        groups = make_vllm_layer_groups_from_kv_caches(saved.cpu_backings, kv_caches)
        adapter = VllmKVSlotAdapter(self.runtime, groups)
        self._adapters_by_prefix[saved.key] = adapter
        return adapter

    def _restore_request(self, request: TurboBusRequestMetadata) -> None:
        total_start = time.perf_counter()
        saved = get_saved_prefix(request.prefix_key)
        if saved is None:
            raise RuntimeError(f"saved prefix {request.prefix_key!r} is not registered")
        prepare_start = time.perf_counter()
        adapter = self._adapter_for_saved_prefix(saved)
        kv_caches = list(self.state.kv_caches.values())
        refs = make_vllm_layer_range_refs_from_ids(
            request.request_id,
            request.block_ids,
            kv_caches,
            cpu_slot_start=request.cpu_slot_start,
        )
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        transfer_start = time.perf_counter()
        handles = adapter.restore_prefix(refs)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        stats = _summarize_handles(handles)
        self.state.events.append(
            {
                "event": "restore",
                "request_id": request.request_id,
                "prefix_key": request.prefix_key,
                "block_count": len(request.block_ids),
                "matched_tokens": request.matched_tokens,
                "elapsed_ms": transfer_ms,
                "prepare_ms": prepare_ms,
                "transfer_ms": transfer_ms,
                "total_ms": total_ms,
                "layers": len(kv_caches),
                "ranges": len(refs),
                **stats,
            }
        )
        _emit_event(
            "restore",
            request_id=request.request_id,
            prefix_key=request.prefix_key,
            block_count=len(request.block_ids),
            matched_tokens=request.matched_tokens,
            elapsed_ms=f"{transfer_ms:.3f}",
            prepare_ms=f"{prepare_ms:.3f}",
            transfer_ms=f"{transfer_ms:.3f}",
            total_ms=f"{total_ms:.3f}",
            layers=len(kv_caches),
            ranges=len(refs),
            **stats,
        )

    def _save_request(self, request: TurboBusRequestMetadata) -> None:
        total_start = time.perf_counter()
        if self.runtime is None:
            self.runtime = _make_runtime_from_config(self._vllm_config)
        if not self.state.kv_caches:
            raise RuntimeError("vLLM did not register KV caches for TurboBus")
        from .vllm import VllmKVSlotAdapter
        from .vllm import make_vllm_layer_groups_from_kv_caches

        prepare_start = time.perf_counter()
        kv_caches = list(self.state.kv_caches.values())
        cpu_backings = self._allocate_cpu_backings(request.block_count, kv_caches)
        groups = make_vllm_layer_groups_from_kv_caches(cpu_backings, kv_caches)
        adapter = VllmKVSlotAdapter(self.runtime, groups)
        refs = make_vllm_layer_range_refs_from_ids(
            request.request_id,
            request.block_ids,
            kv_caches,
            cpu_slot_start=request.cpu_slot_start,
        )
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        transfer_start = time.perf_counter()
        handles = adapter.save_prefix(refs)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        stats = _summarize_handles(handles)
        register_saved_prefix(
            request.prefix_key,
            cpu_backings,
            block_count=request.block_count,
            matched_tokens=request.matched_tokens,
            source_request_id=request.request_id,
            elapsed_ms=transfer_ms,
            **stats,
        )
        self._adapters_by_prefix[request.prefix_key] = adapter
        self.state.saved_request_ids.add(request.request_id)
        self.state.events.append(
            {
                "event": "save",
                "request_id": request.request_id,
                "prefix_key": request.prefix_key,
                "block_count": len(request.block_ids),
                "matched_tokens": request.matched_tokens,
                "elapsed_ms": transfer_ms,
                "prepare_ms": prepare_ms,
                "transfer_ms": transfer_ms,
                "total_ms": total_ms,
                "layers": len(kv_caches),
                "ranges": len(refs),
                **stats,
            }
        )
        _emit_event(
            "save",
            request_id=request.request_id,
            prefix_key=request.prefix_key,
            block_count=len(request.block_ids),
            matched_tokens=request.matched_tokens,
            elapsed_ms=f"{transfer_ms:.3f}",
            prepare_ms=f"{prepare_ms:.3f}",
            transfer_ms=f"{transfer_ms:.3f}",
            total_ms=f"{total_ms:.3f}",
            layers=len(kv_caches),
            ranges=len(refs),
            **stats,
        )

    def _allocate_cpu_backings(self, block_count: int, kv_caches: list[Any]) -> list[Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - import-time convenience only
            raise RuntimeError("PyTorch is required to allocate vLLM CPU backings") from exc

        slots_per_layer = max(1, int(block_count) * _max_lanes_per_layer(kv_caches))
        backings = []
        for kv_cache in kv_caches:
            from .vllm import block_bytes_from_vllm_kv_tensor

            block_bytes = block_bytes_from_vllm_kv_tensor(kv_cache)
            backings.append(
                torch.empty(
                    slots_per_layer * block_bytes,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
            )
        return backings


def _request_params(request) -> dict[str, Any]:
    params = getattr(request, "kv_transfer_params", None)
    if isinstance(params, dict):
        return params
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if isinstance(extra_args, dict):
        params = extra_args.get("kv_transfer_params")
        if isinstance(params, dict):
            return params
    return {}


def _request_prefix_key(params: dict[str, Any]) -> str:
    return str(params.get("turbobus.prefix_key", "default"))


def _flatten_block_ids(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    seen = set()
    ordered = []
    for group in groups:
        for block_id in group:
            if block_id not in seen:
                seen.add(block_id)
                ordered.append(block_id)
    return tuple(ordered)


def _iter_scheduled_requests(scheduler_output) -> list[Any]:
    requests = []
    for request in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        requests.append(request)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if isinstance(cached, list):
        requests.extend(cached)
    elif cached is not None:
        req_ids = list(getattr(cached, "req_ids", []) or [])
        new_block_ids = list(getattr(cached, "new_block_ids", []) or [])
        for index, req_id in enumerate(req_ids):
            request = getattr(cached, "requests", {}).get(req_id, None)
            params = _request_params(request) if request is not None else {}
            requests.append(
                _ScheduledRequestView(
                    req_id=str(req_id),
                    new_block_ids=new_block_ids[index] if index < len(new_block_ids) else [],
                    kv_transfer_params=params,
                )
            )
    return requests


def _scheduled_request_id(request) -> str:
    return str(
        getattr(
            request,
            "request_id",
            getattr(request, "req_id", "unknown"),
        )
    )


def _scheduled_request_block_ids(request) -> tuple[int, ...]:
    raw = getattr(request, "new_block_ids", None)
    if raw is None:
        raw = getattr(request, "block_ids", None)
    return _flatten_block_ids(_normalize_block_id_groups(raw))


def _normalize_block_id_groups(raw) -> tuple[tuple[int, ...], ...]:
    if raw is None:
        return tuple()
    if hasattr(raw, "get_block_ids"):
        return extract_vllm_block_ids(raw)
    if isinstance(raw, tuple):
        return tuple(tuple(int(block_id) for block_id in group) for group in raw)
    if isinstance(raw, list):
        if not raw:
            return tuple()
        if all(isinstance(item, int) for item in raw):
            return (tuple(int(item) for item in raw),)
        groups = []
        for group in raw:
            if group is None:
                groups.append(tuple())
            elif isinstance(group, int):
                groups.append((int(group),))
            else:
                groups.append(tuple(int(block_id) for block_id in group))
        return tuple(groups)
    return tuple()


def _block_count_for_tokens(token_count: int, block_size: int) -> int:
    if token_count <= 0:
        return 0
    return (int(token_count) + int(block_size) - 1) // int(block_size)


def _save_block_count(params: dict[str, Any], block_size: int) -> int:
    if "turbobus.save_blocks" in params:
        return int(params.get("turbobus.save_blocks", 0) or 0)
    return _block_count_for_tokens(
        int(params.get("turbobus.matched_tokens", 0) or 0),
        block_size,
    )


def _matched_tokens_for_save(params: dict[str, Any], block_size: int) -> int:
    matched_tokens = int(params.get("turbobus.matched_tokens", 0) or 0)
    if matched_tokens > 0:
        return matched_tokens
    return _save_block_count(params, block_size) * int(block_size)


def _extra_config_int(vllm_config, key: str, default: int) -> int:
    config = getattr(vllm_config, "kv_transfer_config", None)
    getter = getattr(config, "get_from_extra_config", None)
    if getter is None:
        return default
    value = getter(key, default)
    return int(value)


def _extra_config_bool(vllm_config, key: str, default: bool) -> bool:
    config = getattr(vllm_config, "kv_transfer_config", None)
    getter = getattr(config, "get_from_extra_config", None)
    if getter is None:
        return default
    value = getter(key, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _make_runtime_from_config(vllm_config) -> Runtime:
    config = getattr(vllm_config, "kv_transfer_config", None)
    getter = getattr(config, "get_from_extra_config", None)

    def get(name: str, default):
        if getter is None:
            return default
        return getter(name, default)

    target_gpu = int(get("turbobus.target_gpu", os.environ.get("TURBOBUS_TARGET_GPU", "0")))
    relay_value = str(get("turbobus.relay_gpus", os.environ.get("TURBOBUS_RELAY_GPUS", "")))
    relay_gpus = [int(item) for item in relay_value.split(",") if item.strip()]
    options = RuntimeOptions(
        chunk_bytes=int(get("turbobus.chunk_bytes", os.environ.get("TURBOBUS_CHUNK_BYTES", 4 * 1024 * 1024))),
        profile_bytes=int(get("turbobus.profile_bytes", os.environ.get("TURBOBUS_PROFILE_BYTES", 16 * 1024 * 1024))),
        transfer_mode=str(get("turbobus.mode", os.environ.get("TURBOBUS_MODE", "pool"))),
    )
    return Runtime(target_gpu=target_gpu, relay_gpus=relay_gpus, options=options)


def _max_lanes_per_layer(kv_caches: list[Any]) -> int:
    return max(
        (
            int(kv_cache.shape[0]) if len(getattr(kv_cache, "shape", ())) >= 3 else 1
            for kv_cache in kv_caches
        ),
        default=1,
    )


def _summarize_handles(handles: list) -> dict[str, int]:
    unique = []
    seen = set()
    for handle in handles:
        if id(handle) in seen or getattr(handle, "stats", None) is None:
            continue
        seen.add(id(handle))
        unique.append(handle.stats)
    return {
        "bytes": sum(stats.bytes for stats in unique),
        "direct_chunks": sum(stats.direct_chunks for stats in unique),
        "relay_chunks": sum(stats.relay_chunks for stats in unique),
    }


def _emit_event(event: str, **fields) -> None:
    parts = ["turbobus_kv_connector_event", f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


__all__ = [
    "TurboBusConnector",
    "TurboBusConnectorMetadata",
    "TurboBusRequestMetadata",
    "TurboBusSavedPrefix",
    "clear_saved_prefixes",
    "get_saved_prefix",
    "register_saved_prefix",
]
