from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

from .daemon import TurboBusDaemonClient
from .plan_trace import transfer_plan_to_dict
from .transfer_selector import (
    AutoTransferDecision,
    AutoTransferSelector,
    TransferMode,
)

try:
    import torch
except ImportError:  # pragma: no cover - import-time convenience only
    torch = None

try:
    from . import _turbobus
except ImportError as exc:  # pragma: no cover - depends on local build
    _turbobus = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _native_transfer_mode(mode: TransferMode | str):
    _require_extension()
    if not isinstance(mode, TransferMode):
        mode = TransferMode(mode)
    if mode is TransferMode.POOL:
        return _turbobus.TransferMode.Pool
    if mode is TransferMode.DIRECT:
        return _turbobus.TransferMode.DirectOnly
    if mode is TransferMode.RELAY:
        return _turbobus.TransferMode.RelayOnly
    raise ValueError(f"unsupported transfer mode: {mode}")


def _runtime_transfer_mode_value(mode: TransferMode | str):
    if _turbobus is None:
        return TransferMode(mode)
    return _native_transfer_mode(mode)


@dataclass
class RuntimeOptions:
    chunk_bytes: int = 16 * 1024 * 1024
    staging_slots: int = 2
    enable_peer_access: bool = True
    profile_bytes: int = 256 * 1024 * 1024
    profile_on_first_transfer: bool = True
    profile_cache_enabled: bool = True
    transfer_mode: TransferMode | str = TransferMode.POOL
    min_chunks_for_relay: int = 2
    min_pool_bytes: int = 12 * 1024 * 1024
    relay_min_effective_bw_gbps: float = 0.0
    relay_min_direct_ratio: float = 0.0
    enable_dynamic_weights: bool = False
    dynamic_weight_alpha: float = 0.25
    daemon_socket_path: str | None = None
    daemon_max_inflight_chunks: int = 8
    daemon_profile_max_age_seconds: float = 3600.0

    @classmethod
    def from_tuning_json(cls, path: str | Path) -> "RuntimeOptions":
        data = _read_json(path)
        best = data.get("best")
        if not isinstance(best, dict):
            raise ValueError("tuning JSON does not contain a 'best' object")
        chunk_bytes = int(best["chunk_bytes"])
        staging_slots = int(best["staging_slots"])
        return cls(chunk_bytes=chunk_bytes, staging_slots=staging_slots)

    @classmethod
    def from_profile_json(cls, path: str | Path) -> "RuntimeOptions":
        data = _read_json(path)
        config = data.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("profile JSON contains an invalid 'config' object")
        defaults = cls()
        return cls(
            chunk_bytes=int(config.get("chunk_bytes", defaults.chunk_bytes)),
            staging_slots=int(config.get("staging_slots", defaults.staging_slots)),
            profile_bytes=int(config.get("profile_bytes", defaults.profile_bytes)),
        )

    def to_native(self):
        _require_extension()
        options = _turbobus.RuntimeOptions()
        options.chunk_bytes = self.chunk_bytes
        options.staging_slots = self.staging_slots
        options.enable_peer_access = self.enable_peer_access
        options.profile_bytes = self.profile_bytes
        options.profile_on_first_transfer = self.profile_on_first_transfer
        options.profile_cache_enabled = self.profile_cache_enabled
        mode = TransferMode(self.transfer_mode)
        options.transfer_mode = (
            _turbobus.TransferMode.Pool
            if mode is TransferMode.AUTO
            else _native_transfer_mode(mode)
        )
        options.min_chunks_for_relay = self.min_chunks_for_relay
        options.relay_min_effective_bw_gbps = self.relay_min_effective_bw_gbps
        options.relay_min_direct_ratio = self.relay_min_direct_ratio
        options.enable_dynamic_weights = self.enable_dynamic_weights
        options.dynamic_weight_alpha = self.dynamic_weight_alpha
        return options


def _require_extension() -> None:
    if _turbobus is None:
        raise RuntimeError(
            "turbobus native extension is not available. Build cpp/_turbobus "
            "before using the runtime."
        ) from _IMPORT_ERROR


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor based TurboBus APIs")


class Runtime:
    def __init__(
        self,
        target_gpu: int,
        relay_gpus: Iterable[int] | None = None,
        options: RuntimeOptions | None = None,
    ) -> None:
        _require_extension()
        self.target_gpu = int(target_gpu)
        self.relay_gpus = [int(gpu) for gpu in (relay_gpus or [])]
        self.options = options or RuntimeOptions()
        self._daemon_client = None
        self._daemon_session_id: str | None = None
        self._daemon_profile = None
        self._last_daemon_reservation: dict[str, object] = {}
        self._last_daemon_profile: dict[str, object] = {}
        self._last_resolved_transfer_mode = TransferMode.POOL
        self._last_auto_decision: AutoTransferDecision | None = None
        self._forced_transfer_mode: TransferMode | None = None
        if TransferMode(self.options.transfer_mode) is TransferMode.AUTO:
            self._last_resolved_transfer_mode = TransferMode.AUTO
        self._runtime = _turbobus.Runtime(self.options.to_native())
        self._runtime.init(self.target_gpu, self.relay_gpus)
        self._init_daemon_session()

    def _init_daemon_session(self) -> None:
        if not self.options.daemon_socket_path:
            return
        client = TurboBusDaemonClient(self.options.daemon_socket_path)
        response = client.register_session(
            target_gpu=self.target_gpu,
            relay_gpus=self.relay_gpus,
            max_inflight_chunks=self.options.daemon_max_inflight_chunks,
        )
        if not response.ok:
            raise RuntimeError(response.error or "daemon session registration failed")
        self._daemon_client = client
        self._daemon_session_id = str(response.payload["session"]["session_id"])
        self._load_daemon_profile()

    def close(self) -> None:
        if self._daemon_client is not None and self._daemon_session_id is not None:
            self._daemon_client.close_session(self._daemon_session_id)
            self._daemon_session_id = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def profile(self, bytes: int = 256 * 1024 * 1024, force: bool = False):
        profile = self._runtime.profile(int(bytes), bool(force))
        self._daemon_profile = profile
        self._publish_daemon_profile(profile, int(bytes))
        return profile

    def cached_profile(self):
        return self._runtime.cached_profile()

    def planner_profile(self):
        return self._runtime.planner_profile()

    def last_plan(self):
        return self._runtime.last_plan()

    def last_plan_dict(self) -> dict:
        plan = transfer_plan_to_dict(self.last_plan())
        requested_mode = TransferMode(self.options.transfer_mode)
        plan["requested_transfer_mode"] = requested_mode.value
        plan["resolved_transfer_mode"] = self._last_resolved_transfer_mode.value
        return plan

    def set_transfer_mode(self, mode: TransferMode | str) -> None:
        self.options.transfer_mode = TransferMode(mode)
        if self.options.transfer_mode is TransferMode.AUTO:
            self._last_resolved_transfer_mode = TransferMode.AUTO
            self._runtime.set_transfer_mode(_runtime_transfer_mode_value(TransferMode.POOL))
            return
        self._last_resolved_transfer_mode = self.options.transfer_mode
        self._runtime.set_transfer_mode(_runtime_transfer_mode_value(self.options.transfer_mode))

    def resolve_transfer_mode(
        self,
        bytes: int,
        direction: str = "h2d",
        range_count: int | None = None,
    ) -> AutoTransferDecision:
        requested_mode = TransferMode(self.options.transfer_mode)
        forced_mode = getattr(self, "_forced_transfer_mode", None)
        if forced_mode is not None:
            return self._explicit_transfer_decision(
                forced_mode,
                bytes,
                range_count,
                reason="batch resolved transfer mode",
                clear_auto_decision=False,
            )
        if requested_mode is not TransferMode.AUTO:
            return self._explicit_transfer_decision(
                requested_mode,
                bytes,
                range_count,
                reason="explicit transfer mode",
                clear_auto_decision=True,
            )

        decision = self._auto_transfer_decision(bytes, direction, range_count)
        self._last_resolved_transfer_mode = decision.resolved_mode
        self._last_auto_decision = decision
        self._runtime.set_transfer_mode(_runtime_transfer_mode_value(decision.resolved_mode))
        return decision

    def _explicit_transfer_decision(
        self,
        mode: TransferMode,
        bytes: int,
        range_count: int | None,
        *,
        reason: str,
        clear_auto_decision: bool,
    ) -> AutoTransferDecision:
        request_chunks = max(
            1,
            int(range_count)
            if range_count is not None
            else math.ceil(max(0, int(bytes)) / max(1, int(self.options.chunk_bytes))),
        )
        decision = AutoTransferDecision(
            requested_mode=mode,
            resolved_mode=mode,
            request_bytes=max(0, int(bytes)),
            request_chunks=request_chunks,
            direct_h2d_bw_gbps=0.0,
            relay_effective_bw_gbps=0.0,
            eligible_relay_devices=tuple(self.relay_gpus),
            reason=reason,
        )
        self._last_resolved_transfer_mode = mode
        if clear_auto_decision:
            self._last_auto_decision = None
        self._runtime.set_transfer_mode(_runtime_transfer_mode_value(mode))
        return decision

    def _auto_transfer_decision(
        self,
        bytes: int,
        direction: str,
        range_count: int | None,
    ) -> AutoTransferDecision:
        plan_profile = self._auto_profile()
        direct_attr = "direct_h2d_bw_gbps" if direction == "h2d" else "direct_d2h_bw_gbps"
        direct_bw = getattr(plan_profile, direct_attr, 0.0)
        if direction != "h2d" and direct_bw <= 0.0:
            direct_bw = getattr(plan_profile, "direct_h2d_bw_gbps", 0.0)
        missing_direct_profile = direct_bw <= 0.0
        missing_relay_profile = bool(self.relay_gpus) and not plan_profile.relays
        if missing_direct_profile or missing_relay_profile:
            self.profile(self.options.profile_bytes, force=missing_relay_profile)
        selector = AutoTransferSelector(
            min_chunks_for_relay=self.options.min_chunks_for_relay,
            min_pool_bytes=self.options.min_pool_bytes,
            relay_min_effective_bw_gbps=self.options.relay_min_effective_bw_gbps,
            relay_min_direct_ratio=self.options.relay_min_direct_ratio,
        )
        plan_profile = self._auto_profile()
        decision = selector.choose(
            plan_profile,
            request_bytes=bytes,
            chunk_bytes=self.options.chunk_bytes,
            request_chunks=range_count,
            direction=direction,
        )
        return decision

    def _auto_profile(self):
        daemon_profile = getattr(self, "_daemon_profile", None)
        if daemon_profile is not None and not self.options.enable_dynamic_weights:
            return daemon_profile
        return self.planner_profile() if self.options.enable_dynamic_weights else self.cached_profile()

    def _load_daemon_profile(self) -> None:
        if self._daemon_client is None:
            return
        getter = getattr(self._daemon_client, "get_profile", None)
        if not callable(getter):
            return
        try:
            response = getter(self.target_gpu, self.relay_gpus)
            if not response.ok:
                self._daemon_profile = None
                self._last_daemon_profile = {
                    "daemon_profile_status": "miss",
                    "daemon_profile_error": response.error or "",
                }
                return
            entry = response.payload.get("profile")
            if not entry:
                self._daemon_profile = None
                self._last_daemon_profile = {"daemon_profile_status": "miss"}
                return
            if not _daemon_profile_is_fresh(
                entry,
                max_age_seconds=self.options.daemon_profile_max_age_seconds,
            ):
                self._daemon_profile = None
                self._last_daemon_profile = {"daemon_profile_status": "stale"}
                return
            profile = _profile_from_daemon_entry(entry, self.target_gpu)
        except Exception as exc:
            self._daemon_profile = None
            self._last_daemon_profile = {
                "daemon_profile_status": "invalid",
                "daemon_profile_error": str(exc),
            }
            return

        self._daemon_profile = profile
        self._set_native_cached_profile(profile)
        self._last_daemon_profile = {
            "daemon_profile_status": "hit",
            "daemon_profile_updated_at": entry.get("updated_at", 0.0),
            "daemon_profile_bytes": entry.get("profile_bytes", 0),
        }

    def _publish_daemon_profile(self, profile, profile_bytes: int) -> None:
        daemon_client = getattr(self, "_daemon_client", None)
        if daemon_client is None:
            return
        publisher = getattr(daemon_client, "put_profile", None)
        if not callable(publisher):
            return
        try:
            response = publisher(
                self.target_gpu,
                self.relay_gpus,
                _profile_to_daemon_dict(profile),
                profile_bytes=int(profile_bytes),
            )
        except Exception as exc:
            self._last_daemon_profile = {
                "daemon_profile_status": "publish_failed",
                "daemon_profile_error": str(exc),
            }
            return
        if response.ok:
            self._last_daemon_profile = {
                "daemon_profile_status": "published",
                "daemon_profile_bytes": int(profile_bytes),
            }
        else:
            self._last_daemon_profile = {
                "daemon_profile_status": "publish_failed",
                "daemon_profile_error": response.error or "",
            }

    def _set_native_cached_profile(self, profile) -> None:
        setter = getattr(self._runtime, "set_cached_profile", None)
        if callable(setter):
            setter(profile)

    def last_daemon_profile_dict(self) -> dict[str, object]:
        return dict(getattr(self, "_last_daemon_profile", {}))

    @contextmanager
    def batch_transfer_mode(
        self,
        bytes: int,
        direction: str,
        range_count: int | None = None,
    ):
        decision = self.resolve_transfer_mode(bytes, direction=direction, range_count=range_count)
        previous = self._forced_transfer_mode
        self._forced_transfer_mode = decision.resolved_mode
        try:
            yield decision
        finally:
            self._forced_transfer_mode = previous

    def last_transfer_mode(self) -> TransferMode:
        return self._last_resolved_transfer_mode

    def last_auto_decision_dict(self) -> dict[str, object]:
        decision = self._last_auto_decision
        if decision is None:
            return {}
        return {
            "auto_resolved_mode": decision.resolved_mode.value,
            "auto_reason": decision.reason.replace(" ", "_"),
            "auto_request_bytes": decision.request_bytes,
            "auto_request_chunks": decision.request_chunks,
            "auto_direct_bw_gbps": f"{decision.direct_h2d_bw_gbps:.3f}",
            "auto_relay_bw_gbps": f"{decision.relay_effective_bw_gbps:.3f}",
            "auto_eligible_relays": ",".join(
                str(device) for device in decision.eligible_relay_devices
            ),
        }

    def last_daemon_reservation_dict(self) -> dict[str, object]:
        return dict(self._last_daemon_reservation)

    def fetch_to_gpu(self, cpu_tensor, gpu_tensor):
        _require_torch()
        bytes_to_copy = _validate_transfer_tensors(
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            target_gpu=self.target_gpu,
            direction="h2d",
        )
        reservations = self._resolve_transfer_with_daemon(
            bytes_to_copy,
            direction="h2d",
        )

        try:
            handle = self._runtime.fetch_to_gpu(
                int(cpu_tensor.data_ptr()),
                int(gpu_tensor.data_ptr()),
                int(bytes_to_copy),
            )
        except Exception:
            self._release_daemon_reservations(reservations)
            raise
        return TransferHandle(self, handle, reservations)

    def offload_to_cpu(self, gpu_tensor, cpu_tensor):
        _require_torch()
        bytes_to_copy = _validate_transfer_tensors(
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            target_gpu=self.target_gpu,
            direction="d2h",
        )
        reservations = self._resolve_transfer_with_daemon(
            bytes_to_copy,
            direction="d2h",
        )

        try:
            handle = self._runtime.offload_to_cpu(
                int(gpu_tensor.data_ptr()),
                int(cpu_tensor.data_ptr()),
                int(bytes_to_copy),
            )
        except Exception:
            self._release_daemon_reservations(reservations)
            raise
        return TransferHandle(self, handle, reservations)

    def fetch_ranges_to_gpu(self, cpu_tensor, gpu_tensor, ranges: Iterable):
        _require_torch()
        source_bytes, destination_bytes = _validate_range_tensors(
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            target_gpu=self.target_gpu,
            direction="h2d",
        )
        range_items = list(ranges)
        native_ranges = _native_ranges(range_items, source_bytes, destination_bytes)
        range_fields = [_range_fields(item) for item in range_items]
        transfer_bytes = sum(int(bytes_) for _, _, bytes_ in range_fields)
        range_count = sum(
            max(1, math.ceil(int(bytes_) / max(1, int(self.options.chunk_bytes))))
            for _, _, bytes_ in range_fields
        )
        reservations = self._resolve_transfer_with_daemon(
            transfer_bytes,
            direction="h2d",
            range_count=range_count,
        )
        try:
            handle = self._runtime.fetch_ranges_to_gpu(
                int(cpu_tensor.data_ptr()),
                int(source_bytes),
                int(gpu_tensor.data_ptr()),
                int(destination_bytes),
                native_ranges,
            )
        except Exception:
            self._release_daemon_reservations(reservations)
            raise
        return TransferHandle(self, handle, reservations)

    def offload_ranges_to_cpu(self, gpu_tensor, cpu_tensor, ranges: Iterable):
        _require_torch()
        source_bytes, destination_bytes = _validate_range_tensors(
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            target_gpu=self.target_gpu,
            direction="d2h",
        )
        range_items = list(ranges)
        native_ranges = _native_ranges(range_items, source_bytes, destination_bytes)
        range_fields = [_range_fields(item) for item in range_items]
        transfer_bytes = sum(int(bytes_) for _, _, bytes_ in range_fields)
        range_count = sum(
            max(1, math.ceil(int(bytes_) / max(1, int(self.options.chunk_bytes))))
            for _, _, bytes_ in range_fields
        )
        reservations = self._resolve_transfer_with_daemon(
            transfer_bytes,
            direction="d2h",
            range_count=range_count,
        )
        try:
            handle = self._runtime.offload_ranges_to_cpu(
                int(gpu_tensor.data_ptr()),
                int(source_bytes),
                int(cpu_tensor.data_ptr()),
                int(destination_bytes),
                native_ranges,
            )
        except Exception:
            self._release_daemon_reservations(reservations)
            raise
        return TransferHandle(self, handle, reservations)

    def wait(self, handle: "TransferHandle") -> None:
        try:
            self._runtime.wait(handle.native)
            handle._status = "complete"
            handle._stats = self.stats(handle)
        finally:
            self._release_daemon_reservations(handle._daemon_reservations)
            handle._daemon_reservations = []

    def stats(self, handle: "TransferHandle"):
        stats = self._runtime.stats(handle.native)
        if handle.daemon_reservation_info:
            return _attach_daemon_stats(stats, handle.daemon_reservation_info)
        return stats

    def run_dummy_compute(self, tensor, iterations: int):
        _require_torch()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if tensor.device.type != "cuda":
            raise ValueError("tensor must be on CUDA")
        if tensor.device.index != self.target_gpu:
            raise ValueError("tensor must be on the runtime target_gpu")
        if tensor.dtype != torch.float32:
            raise ValueError("tensor must be torch.float32")
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")
        return self._runtime.run_dummy_compute(
            int(tensor.data_ptr()),
            int(tensor.numel()),
            int(iterations),
        )

    def _resolve_transfer_with_daemon(
        self,
        bytes: int,
        direction: str,
        range_count: int | None = None,
    ) -> list[str]:
        decision = self.resolve_transfer_mode(bytes, direction=direction, range_count=range_count)
        return self._reserve_daemon_transfer(decision, direction)

    def _reserve_daemon_transfer(
        self,
        decision: AutoTransferDecision,
        direction: str,
    ) -> list[str]:
        if (
            self._daemon_client is None
            or self._daemon_session_id is None
            or decision.resolved_mode is TransferMode.DIRECT
        ):
            self._last_daemon_reservation = {}
            return []

        relay_devices = tuple(decision.eligible_relay_devices or self.relay_gpus)
        if not relay_devices:
            self._last_daemon_reservation = {
                "daemon_session_id": self._daemon_session_id,
                "daemon_reservation_status": "skipped",
                "daemon_reservation_reason": "no relay devices",
            }
            return []

        divisor = len(relay_devices)
        if decision.resolved_mode is TransferMode.POOL:
            divisor += 1
        chunks_per_relay = max(1, math.ceil(decision.request_chunks / divisor))
        bytes_per_relay = (
            math.ceil(decision.request_bytes / divisor)
            if decision.request_bytes > 0
            else 0
        )
        reservations: list[str] = []
        try:
            for relay_gpu in relay_devices:
                response = self._daemon_client.reserve_transfer(
                    self._daemon_session_id,
                    relay_gpu=int(relay_gpu),
                    chunks=chunks_per_relay,
                    bytes_=bytes_per_relay,
                    direction=direction,
                )
                if not response.ok:
                    raise RuntimeError(response.error or "daemon reservation denied")
                reservations.append(str(response.payload["reservation"]["reservation_id"]))
        except Exception as exc:
            self._release_daemon_reservations(reservations)
            self._last_daemon_reservation = {
                "daemon_session_id": self._daemon_session_id,
                "daemon_reservation_status": "denied",
                "daemon_reservation_error": str(exc),
            }
            fallback = AutoTransferDecision(
                requested_mode=decision.requested_mode,
                resolved_mode=TransferMode.DIRECT,
                request_bytes=decision.request_bytes,
                request_chunks=decision.request_chunks,
                direct_h2d_bw_gbps=decision.direct_h2d_bw_gbps,
                relay_effective_bw_gbps=decision.relay_effective_bw_gbps,
                eligible_relay_devices=(),
                reason=f"daemon reservation denied: {exc}",
            )
            self._last_resolved_transfer_mode = TransferMode.DIRECT
            if decision.requested_mode is TransferMode.AUTO:
                self._last_auto_decision = fallback
            self._runtime.set_transfer_mode(_runtime_transfer_mode_value(TransferMode.DIRECT))
            return []

        self._last_daemon_reservation = {
            "daemon_session_id": self._daemon_session_id,
            "daemon_reservation_status": "granted",
            "daemon_reservation_ids": ",".join(reservations),
            "daemon_reserved_relays": ",".join(str(gpu) for gpu in relay_devices),
            "daemon_reserved_chunks_per_relay": chunks_per_relay,
            "daemon_reserved_bytes_per_relay": bytes_per_relay,
            "daemon_reserved_direction": direction,
        }
        return reservations

    def _release_daemon_reservations(self, reservations: list[str]) -> None:
        if self._daemon_client is None:
            return
        for reservation_id in list(reservations):
            self._daemon_client.release_transfer(reservation_id)


def _validate_transfer_tensors(cpu_tensor, gpu_tensor, target_gpu: int, direction: str) -> int:
    if direction not in {"h2d", "d2h"}:
        raise ValueError(f"unsupported transfer direction: {direction}")
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor based TurboBus APIs")
    if not isinstance(cpu_tensor, torch.Tensor):
        raise TypeError("cpu_tensor must be a torch.Tensor")
    if not isinstance(gpu_tensor, torch.Tensor):
        raise TypeError("gpu_tensor must be a torch.Tensor")
    if cpu_tensor.device.type != "cpu":
        raise ValueError("cpu_tensor must be on CPU")
    if not cpu_tensor.is_pinned():
        raise ValueError("cpu_tensor must be pinned memory")
    if gpu_tensor.device.type != "cuda":
        raise ValueError("gpu_tensor must be on CUDA")
    if gpu_tensor.device.index != target_gpu:
        raise ValueError("gpu_tensor must be on the runtime target_gpu")
    if not cpu_tensor.is_contiguous() or not gpu_tensor.is_contiguous():
        raise ValueError("cpu_tensor and gpu_tensor must be contiguous")

    if direction == "h2d":
        bytes_to_copy = cpu_tensor.numel() * cpu_tensor.element_size()
        if gpu_tensor.numel() * gpu_tensor.element_size() < bytes_to_copy:
            raise ValueError("gpu_tensor is smaller than cpu_tensor")
    else:
        bytes_to_copy = gpu_tensor.numel() * gpu_tensor.element_size()
        if cpu_tensor.numel() * cpu_tensor.element_size() < bytes_to_copy:
            raise ValueError("cpu_tensor is smaller than gpu_tensor")
    return bytes_to_copy


def _validate_range_tensors(
    cpu_tensor,
    gpu_tensor,
    target_gpu: int,
    direction: str,
) -> tuple[int, int]:
    _validate_tensor_pair(cpu_tensor, gpu_tensor, target_gpu)
    cpu_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
    gpu_bytes = gpu_tensor.numel() * gpu_tensor.element_size()
    if direction == "h2d":
        return cpu_bytes, gpu_bytes
    if direction != "d2h":
        raise ValueError(f"unsupported transfer direction: {direction}")
    return gpu_bytes, cpu_bytes


def _validate_tensor_pair(cpu_tensor, gpu_tensor, target_gpu: int) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor based TurboBus APIs")
    if not isinstance(cpu_tensor, torch.Tensor):
        raise TypeError("cpu_tensor must be a torch.Tensor")
    if not isinstance(gpu_tensor, torch.Tensor):
        raise TypeError("gpu_tensor must be a torch.Tensor")
    if cpu_tensor.device.type != "cpu":
        raise ValueError("cpu_tensor must be on CPU")
    if not cpu_tensor.is_pinned():
        raise ValueError("cpu_tensor must be pinned memory")
    if gpu_tensor.device.type != "cuda":
        raise ValueError("gpu_tensor must be on CUDA")
    if gpu_tensor.device.index != target_gpu:
        raise ValueError("gpu_tensor must be on the runtime target_gpu")
    if not cpu_tensor.is_contiguous() or not gpu_tensor.is_contiguous():
        raise ValueError("cpu_tensor and gpu_tensor must be contiguous")


def _native_ranges(
    ranges: Iterable,
    source_bytes: int,
    destination_bytes: int,
) -> list:
    _require_extension()
    native = []
    for item in ranges:
        src_offset, dst_offset, bytes_ = _range_fields(item)
        if src_offset < 0 or dst_offset < 0 or bytes_ <= 0:
            raise ValueError("range offsets must be non-negative and bytes must be positive")
        if src_offset + bytes_ > source_bytes:
            raise ValueError("range source extends past source tensor")
        if dst_offset + bytes_ > destination_bytes:
            raise ValueError("range destination extends past destination tensor")
        transfer_range = _turbobus.TransferRange()
        transfer_range.src_offset = int(src_offset)
        transfer_range.dst_offset = int(dst_offset)
        transfer_range.bytes = int(bytes_)
        native.append(transfer_range)
    if not native:
        raise ValueError("at least one non-empty range is required")
    return native


def _range_fields(item) -> tuple[int, int, int]:
    if isinstance(item, Mapping):
        return int(item["src_offset"]), int(item["dst_offset"]), int(item["bytes"])
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        if len(item) != 3:
            raise ValueError("range tuples must be (src_offset, dst_offset, bytes)")
        return int(item[0]), int(item[1]), int(item[2])
    src_offset = getattr(item, "src_offset")
    dst_offset = getattr(item, "dst_offset")
    bytes_ = getattr(item, "bytes")
    return int(src_offset), int(dst_offset), int(bytes_)


def _profile_to_daemon_dict(profile) -> dict[str, object]:
    return {
        "target_device": int(getattr(profile, "target_device", 0)),
        "direct_h2d_bw_gbps": float(getattr(profile, "direct_h2d_bw_gbps", 0.0) or 0.0),
        "direct_d2h_bw_gbps": float(getattr(profile, "direct_d2h_bw_gbps", 0.0) or 0.0),
        "relays": [
            {
                "relay_device": int(getattr(relay, "relay_device")),
                "target_device": int(getattr(relay, "target_device", 0)),
                "h2d_bw_gbps": float(getattr(relay, "h2d_bw_gbps", 0.0) or 0.0),
                "d2h_bw_gbps": float(getattr(relay, "d2h_bw_gbps", 0.0) or 0.0),
                "p2p_bw_gbps": float(getattr(relay, "p2p_bw_gbps", 0.0) or 0.0),
                "effective_bw_gbps": float(
                    getattr(relay, "effective_bw_gbps", 0.0) or 0.0
                ),
                "effective_d2h_bw_gbps": float(
                    getattr(relay, "effective_d2h_bw_gbps", 0.0) or 0.0
                ),
                "p2p_enabled": bool(getattr(relay, "p2p_enabled", False)),
            }
            for relay in getattr(profile, "relays", []) or []
        ],
    }


def _profile_from_daemon_entry(entry: Mapping, target_gpu: int):
    profile = entry.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("daemon profile entry has no profile object")
    direct_h2d = float(profile.get("direct_h2d_bw_gbps", 0.0) or 0.0)
    if direct_h2d <= 0.0:
        raise ValueError("daemon profile direct_h2d_bw_gbps must be positive")
    use_native_profile = _turbobus is not None and hasattr(_turbobus, "ProfileResult")
    if use_native_profile:
        profile_obj = _turbobus.ProfileResult()
        profile_obj.target_device = int(profile.get("target_device", target_gpu))
        profile_obj.direct_h2d_bw_gbps = direct_h2d
        profile_obj.direct_d2h_bw_gbps = float(profile.get("direct_d2h_bw_gbps", 0.0) or 0.0)
        profile_relays = []
    else:
        profile_relays = []
    for relay in profile.get("relays", []) or []:
        if not isinstance(relay, Mapping):
            raise ValueError("daemon profile relay must be an object")
        relay_obj = {
            "relay_device": int(relay["relay_device"]),
            "target_device": int(relay.get("target_device", target_gpu)),
            "h2d_bw_gbps": float(relay.get("h2d_bw_gbps", 0.0) or 0.0),
            "d2h_bw_gbps": float(relay.get("d2h_bw_gbps", 0.0) or 0.0),
            "p2p_bw_gbps": float(relay.get("p2p_bw_gbps", 0.0) or 0.0),
            "effective_bw_gbps": float(relay.get("effective_bw_gbps", 0.0) or 0.0),
            "effective_d2h_bw_gbps": float(
                relay.get("effective_d2h_bw_gbps", 0.0) or 0.0
            ),
            "p2p_enabled": bool(relay.get("p2p_enabled", False)),
        }
        if use_native_profile:
            native_relay = _turbobus.RelayProfile()
            native_relay.relay_device = relay_obj["relay_device"]
            native_relay.target_device = relay_obj["target_device"]
            native_relay.h2d_bw_gbps = relay_obj["h2d_bw_gbps"]
            native_relay.d2h_bw_gbps = relay_obj["d2h_bw_gbps"]
            native_relay.p2p_bw_gbps = relay_obj["p2p_bw_gbps"]
            native_relay.effective_bw_gbps = relay_obj["effective_bw_gbps"]
            native_relay.effective_d2h_bw_gbps = relay_obj["effective_d2h_bw_gbps"]
            native_relay.p2p_enabled = relay_obj["p2p_enabled"]
            profile_relays.append(native_relay)
        else:
            profile_relays.append(
                SimpleProfileRelay(
                    relay_device=relay_obj["relay_device"],
                    target_device=relay_obj["target_device"],
                    h2d_bw_gbps=relay_obj["h2d_bw_gbps"],
                    d2h_bw_gbps=relay_obj["d2h_bw_gbps"],
                    p2p_bw_gbps=relay_obj["p2p_bw_gbps"],
                    effective_bw_gbps=relay_obj["effective_bw_gbps"],
                    effective_d2h_bw_gbps=relay_obj["effective_d2h_bw_gbps"],
                    p2p_enabled=relay_obj["p2p_enabled"],
                )
            )
    if use_native_profile:
        profile_obj.relays = profile_relays
        return profile_obj
    return SimpleProfileResult(
        target_device=int(profile.get("target_device", target_gpu)),
        direct_h2d_bw_gbps=direct_h2d,
        direct_d2h_bw_gbps=float(profile.get("direct_d2h_bw_gbps", 0.0) or 0.0),
        relays=profile_relays,
    )


def _daemon_profile_is_fresh(entry: Mapping, max_age_seconds: float) -> bool:
    if max_age_seconds <= 0:
        return True
    updated_at = float(entry.get("updated_at", 0.0) or 0.0)
    if updated_at <= 0.0:
        return False
    return (time.time() - updated_at) <= float(max_age_seconds)


@dataclass(frozen=True)
class SimpleProfileRelay:
    relay_device: int
    target_device: int
    h2d_bw_gbps: float
    d2h_bw_gbps: float
    p2p_bw_gbps: float
    effective_bw_gbps: float
    effective_d2h_bw_gbps: float
    p2p_enabled: bool


@dataclass(frozen=True)
class SimpleProfileResult:
    target_device: int
    direct_h2d_bw_gbps: float
    direct_d2h_bw_gbps: float
    relays: list[SimpleProfileRelay]


class _TransferStatsWithDaemon:
    def __init__(self, stats, daemon_info: dict[str, object]) -> None:
        self._stats = stats
        self.daemon_reservation_info = dict(daemon_info)
        for key, value in self.daemon_reservation_info.items():
            setattr(self, key, value)

    def __getattr__(self, name: str):
        return getattr(self._stats, name)


def _attach_daemon_stats(stats, daemon_info: dict[str, object]):
    if isinstance(stats, dict):
        return {**stats, **daemon_info}
    return _TransferStatsWithDaemon(stats, daemon_info)


class TransferHandle:
    def __init__(
        self,
        runtime: Runtime,
        native_handle,
        daemon_reservations: list[str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.native = native_handle
        self._daemon_reservations = list(daemon_reservations or [])
        last_daemon_reservation = getattr(runtime, "last_daemon_reservation_dict", None)
        self.daemon_reservation_info = (
            last_daemon_reservation() if callable(last_daemon_reservation) else {}
        )
        self._status = "submitted"
        self._stats = None
        self.error = ""

    @property
    def id(self) -> int:
        return self.native.id

    @property
    def status(self) -> str:
        return self._status

    @property
    def done(self) -> bool:
        return self._status == "complete"

    @property
    def stats(self):
        return self._stats

    def wait(self) -> None:
        if self.done:
            return
        try:
            self.runtime.wait(self)
        except Exception as exc:  # pragma: no cover - error path depends on CUDA
            self._status = "failed"
            self.error = str(exc)
            raise
        else:
            self._status = "complete"

    def __repr__(self) -> str:
        return f"TransferHandle(id={self.id}, status={self.status})"
