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


class TurboBusConnectorMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TurboBusRequestMetadata] = []

    def add_request(self, request: TurboBusRequestMetadata) -> None:
        self.requests.append(request)

    def __len__(self) -> int:
        return len(self.requests)


@dataclass
class TurboBusKVConnectorState:
    kv_caches: dict[str, Any] = field(default_factory=dict)
    pending_loads: dict[str, TurboBusRequestMetadata] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


_SAVED_PREFIXES: dict[str, TurboBusSavedPrefix] = {}


def register_saved_prefix(
    key: str,
    cpu_backings: list[Any],
    *,
    block_count: int,
    matched_tokens: int,
) -> None:
    if not key:
        raise ValueError("prefix key must not be empty")
    _SAVED_PREFIXES[str(key)] = TurboBusSavedPrefix(
        key=str(key),
        cpu_backings=list(cpu_backings),
        block_count=int(block_count),
        matched_tokens=int(matched_tokens),
    )
    _emit_event(
        "register_saved_prefix",
        prefix_key=str(key),
        block_count=int(block_count),
        matched_tokens=int(matched_tokens),
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
        if num_external_tokens <= 0:
            return
        params = _request_params(request)
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
        metadata = TurboBusConnectorMetadata()
        for request_id in sorted(self.state.pending_loads):
            metadata.add_request(self.state.pending_loads[request_id])
        if len(metadata) > 0:
            _emit_event("build_connector_meta", requests=len(metadata))
        self.state.pending_loads.clear()
        return metadata

    def start_load_kv(self, forward_context, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, TurboBusConnectorMetadata) or len(metadata) == 0:
            return
        if not self.restore_enabled:
            for request in metadata.requests:
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
            return
        for request in metadata.requests:
            self._restore_request(request)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def get_finished(self, finished_req_ids: set[str]):
        return None, None

    def request_finished(self, request, block_ids: list[int]):
        return False, None

    def _get_connector_metadata(self):
        return getattr(self, "_connector_metadata", None)

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
        saved = get_saved_prefix(request.prefix_key)
        if saved is None:
            raise RuntimeError(f"saved prefix {request.prefix_key!r} is not registered")
        adapter = self._adapter_for_saved_prefix(saved)
        kv_caches = list(self.state.kv_caches.values())
        refs = make_vllm_layer_range_refs_from_ids(
            request.request_id,
            request.block_ids,
            kv_caches,
            cpu_slot_start=request.cpu_slot_start,
        )
        start = time.perf_counter()
        handles = adapter.restore_prefix(refs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats = _summarize_handles(handles)
        self.state.events.append(
            {
                "event": "restore",
                "request_id": request.request_id,
                "prefix_key": request.prefix_key,
                "block_count": len(request.block_ids),
                "matched_tokens": request.matched_tokens,
                "elapsed_ms": elapsed_ms,
                **stats,
            }
        )
        _emit_event(
            "restore",
            request_id=request.request_id,
            prefix_key=request.prefix_key,
            block_count=len(request.block_ids),
            matched_tokens=request.matched_tokens,
            elapsed_ms=f"{elapsed_ms:.3f}",
            **stats,
        )


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


def _block_count_for_tokens(token_count: int, block_size: int) -> int:
    if token_count <= 0:
        return 0
    return (int(token_count) + int(block_size) - 1) // int(block_size)


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
