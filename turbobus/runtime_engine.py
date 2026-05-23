from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .schema import AutoTransferDecision, TransferMode

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
            "effective_d2h_bw_gbps": float(relay.get("effective_d2h_bw_gbps", 0.0) or 0.0),
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


def _validate_transfer_tensors(cpu_tensor, gpu_tensor, target_gpu: int, direction: str) -> int:
    if direction not in {"h2d", "d2h"}:
        raise ValueError(f"unsupported transfer direction: {direction}")
    _require_torch()
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
    _require_torch()
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
        runtime,
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


__all__ = [
    "RuntimeOptions",
    "SimpleProfileRelay",
    "SimpleProfileResult",
    "TransferHandle",
    "_daemon_profile_is_fresh",
    "_profile_from_daemon_entry",
    "_profile_to_daemon_dict",
    "_range_fields",
    "_runtime_transfer_mode_value",
    "_native_ranges",
    "_validate_range_tensors",
    "_validate_tensor_pair",
    "_validate_transfer_tensors",
]
