from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable

from . import runtime_engine as _runtime_engine
from .daemon import TurboBusDaemonClient
from .plan_trace import transfer_plan_to_dict
from .runtime_engine import (
    RuntimeOptions,
    TransferHandle,
)
from .schema import AutoTransferDecision, TransferMode
from .transfer_selector import AutoTransferSelector

try:
    import torch
except ImportError:  # pragma: no cover - import-time convenience only
    torch = None

try:
    from . import _turbobus
except ImportError:  # pragma: no cover - depends on local build
    _turbobus = None


def _sync_runtime_engine() -> None:
    _runtime_engine._turbobus = _turbobus
    _runtime_engine.torch = torch


def _require_extension() -> None:
    _sync_runtime_engine()
    _runtime_engine._require_extension()


def _require_torch() -> None:
    _sync_runtime_engine()
    _runtime_engine._require_torch()


def _runtime_transfer_mode_value(mode: TransferMode | str):
    _sync_runtime_engine()
    return _runtime_engine._runtime_transfer_mode_value(mode)


def _native_ranges(
    ranges: Iterable,
    source_bytes: int,
    destination_bytes: int,
) -> list:
    _sync_runtime_engine()
    return _runtime_engine._native_ranges(ranges, source_bytes, destination_bytes)


def _validate_range_tensors(
    cpu_tensor,
    gpu_tensor,
    target_gpu: int,
    direction: str,
) -> tuple[int, int]:
    _sync_runtime_engine()
    return _runtime_engine._validate_range_tensors(cpu_tensor, gpu_tensor, target_gpu, direction)


def _validate_tensor_pair(cpu_tensor, gpu_tensor, target_gpu: int) -> None:
    _sync_runtime_engine()
    return _runtime_engine._validate_tensor_pair(cpu_tensor, gpu_tensor, target_gpu)


def _profile_to_daemon_dict(profile) -> dict[str, object]:
    return _runtime_engine._profile_to_daemon_dict(profile)


def _profile_from_daemon_entry(entry, target_gpu: int):
    return _runtime_engine._profile_from_daemon_entry(entry, target_gpu)


def _daemon_profile_is_fresh(entry, max_age_seconds: float) -> bool:
    return _runtime_engine._daemon_profile_is_fresh(entry, max_age_seconds)


def _attach_daemon_stats(stats, daemon_info: dict[str, object]):
    return _runtime_engine._attach_daemon_stats(stats, daemon_info)


def _mode_from_daemon_stats(stats, default: TransferMode) -> TransferMode:
    if not isinstance(stats, dict):
        return default
    value = stats.get("resolved_mode")
    if value is None:
        return default
    try:
        return TransferMode(str(value))
    except ValueError:
        try:
            return TransferMode[str(value).upper()]
        except KeyError:
            return default


def _daemon_lease_summary(leases: list) -> dict[str, object]:
    relay_devices = []
    chunk_limits = []
    byte_limits = []
    for lease in leases:
        if not isinstance(lease, dict):
            continue
        relay_devices.append(str(lease.get("relay_device", "")))
        chunk_limits.append(str(lease.get("chunk_limit", "")))
        byte_limits.append(str(lease.get("bytes_limit", "")))
    return {
        "daemon_reserved_relays": ",".join(item for item in relay_devices if item),
        "daemon_reserved_chunks_per_relay": ",".join(
            item for item in chunk_limits if item
        ),
        "daemon_reserved_bytes_per_relay": ",".join(
            item for item in byte_limits if item
        ),
    }


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
        self._last_auto_decision_profile = None
        self._last_auto_decision_key: tuple[object, ...] | None = None
        self._forced_transfer_mode: TransferMode | None = None
        if TransferMode(self.options.transfer_mode) is TransferMode.AUTO:
            self._last_resolved_transfer_mode = TransferMode.AUTO
        self._runtime = _turbobus.Runtime(self.options.to_native())
        self._runtime.init(self.target_gpu, self.relay_gpus)
        self._last_native_transfer_mode = (
            TransferMode.POOL
            if TransferMode(self.options.transfer_mode) is TransferMode.AUTO
            else TransferMode(self.options.transfer_mode)
        )
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
            self._set_native_transfer_mode(TransferMode.POOL)
            return
        self._last_resolved_transfer_mode = self.options.transfer_mode
        self._set_native_transfer_mode(self.options.transfer_mode)

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
        self._set_native_transfer_mode(decision.resolved_mode)
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
        self._set_native_transfer_mode(mode)
        return decision

    def _set_native_transfer_mode(self, mode: TransferMode | str) -> None:
        native_mode = TransferMode(mode)
        last_native = getattr(self, "_last_native_transfer_mode", None)
        if last_native is native_mode:
            return
        self._runtime.set_transfer_mode(_runtime_transfer_mode_value(native_mode))
        self._last_native_transfer_mode = native_mode

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
            plan_profile = self._auto_profile()
            direct_bw = getattr(plan_profile, direct_attr, 0.0)
            if direction != "h2d" and direct_bw <= 0.0:
                direct_bw = getattr(plan_profile, "direct_h2d_bw_gbps", 0.0)
            missing_direct_profile = direct_bw <= 0.0
            missing_relay_profile = bool(self.relay_gpus) and not plan_profile.relays
        cache_key = self._auto_decision_key(bytes, direction, range_count)
        last_profile = getattr(self, "_last_auto_decision_profile", None)
        last_key = getattr(self, "_last_auto_decision_key", None)
        if (
            plan_profile is last_profile
            and cache_key == last_key
            and self._last_auto_decision is not None
        ):
            return self._last_auto_decision
        selector = AutoTransferSelector(
            min_chunks_for_relay=self.options.min_chunks_for_relay,
            min_pool_bytes=self.options.min_pool_bytes,
            relay_min_effective_bw_gbps=self.options.relay_min_effective_bw_gbps,
            relay_min_direct_ratio=self.options.relay_min_direct_ratio,
        )
        decision = selector.choose(
            plan_profile,
            request_bytes=bytes,
            chunk_bytes=self.options.chunk_bytes,
            request_chunks=range_count,
            direction=direction,
        )
        self._last_auto_decision_profile = plan_profile
        self._last_auto_decision_key = cache_key
        return decision

    def _auto_profile(self):
        daemon_profile = getattr(self, "_daemon_profile", None)
        if daemon_profile is not None and not self.options.enable_dynamic_weights:
            return daemon_profile
        return self.planner_profile() if self.options.enable_dynamic_weights else self.cached_profile()

    def _auto_decision_key(
        self,
        bytes: int,
        direction: str,
        range_count: int | None,
    ) -> tuple[object, ...]:
        return (
            int(bytes),
            str(direction),
            None if range_count is None else int(range_count),
            int(self.options.chunk_bytes),
            int(self.options.min_chunks_for_relay),
            int(self.options.min_pool_bytes),
            float(self.options.relay_min_effective_bw_gbps),
            float(self.options.relay_min_direct_ratio),
            bool(self.options.enable_dynamic_weights),
        )

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

        planned = self._plan_daemon_transfer(decision, direction)
        if planned is not None:
            return planned

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

    def _plan_daemon_transfer(
        self,
        decision: AutoTransferDecision,
        direction: str,
    ) -> list[str] | None:
        planner = getattr(self._daemon_client, "plan_transfer", None)
        if not callable(planner):
            return None
        try:
            response = planner(
                self._daemon_session_id,
                total_bytes=decision.request_bytes,
                chunk_bytes=self.options.chunk_bytes,
                mode=decision.resolved_mode.value,
                direction=direction,
            )
        except Exception as exc:
            self._apply_daemon_plan_fallback(decision, str(exc))
            return []

        if not response.ok:
            error = response.error or "daemon plan denied"
            if "unsupported" in error or "not a valid RequestType" in error:
                return None
            self._apply_daemon_plan_fallback(decision, error)
            return []

        payload = response.payload or {}
        stats = payload.get("stats", {})
        fallback_reason = (
            stats.get("fallback_reason") if isinstance(stats, dict) else None
        )
        resolved_mode = _mode_from_daemon_stats(stats, decision.resolved_mode)
        self._last_resolved_transfer_mode = resolved_mode
        self._set_native_transfer_mode(resolved_mode)
        if fallback_reason and decision.requested_mode is TransferMode.AUTO:
            self._last_auto_decision = AutoTransferDecision(
                requested_mode=decision.requested_mode,
                resolved_mode=resolved_mode,
                request_bytes=decision.request_bytes,
                request_chunks=decision.request_chunks,
                direct_h2d_bw_gbps=decision.direct_h2d_bw_gbps,
                relay_effective_bw_gbps=decision.relay_effective_bw_gbps,
                eligible_relay_devices=(),
                reason=f"daemon reservation denied: {fallback_reason}",
            )

        reservations = [
            str(reservation["reservation_id"])
            for reservation in (payload.get("reservations") or [])
        ]
        leases = payload.get("leases") or []
        status = "granted" if reservations else "planned"
        if fallback_reason:
            status = "denied"
        info = {
            "daemon_session_id": self._daemon_session_id,
            "daemon_reservation_status": status,
            "daemon_plan_resolved_mode": resolved_mode.value,
            "daemon_reserved_direction": direction,
        }
        if reservations:
            info["daemon_reservation_ids"] = ",".join(reservations)
        if fallback_reason:
            info["daemon_reservation_error"] = str(fallback_reason)
        if leases:
            info.update(_daemon_lease_summary(leases))
        self._last_daemon_reservation = info
        return reservations

    def _apply_daemon_plan_fallback(
        self,
        decision: AutoTransferDecision,
        reason: str,
    ) -> None:
        self._last_daemon_reservation = {
            "daemon_session_id": self._daemon_session_id,
            "daemon_reservation_status": "denied",
            "daemon_reservation_error": str(reason),
        }
        fallback = AutoTransferDecision(
            requested_mode=decision.requested_mode,
            resolved_mode=TransferMode.DIRECT,
            request_bytes=decision.request_bytes,
            request_chunks=decision.request_chunks,
            direct_h2d_bw_gbps=decision.direct_h2d_bw_gbps,
            relay_effective_bw_gbps=decision.relay_effective_bw_gbps,
            eligible_relay_devices=(),
            reason=f"daemon reservation denied: {reason}",
        )
        self._last_resolved_transfer_mode = TransferMode.DIRECT
        if decision.requested_mode is TransferMode.AUTO:
            self._last_auto_decision = fallback
        self._set_native_transfer_mode(TransferMode.DIRECT)

    def _release_daemon_reservations(self, reservations: list[str]) -> None:
        if self._daemon_client is None:
            return
        for reservation_id in list(reservations):
            self._daemon_client.release_transfer(reservation_id)
