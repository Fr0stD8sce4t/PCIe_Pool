from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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


class TransferMode(str, Enum):
    AUTO = "auto"
    POOL = "pool"
    DIRECT = "direct"
    RELAY = "relay"


@dataclass(frozen=True)
class AutoTransferDecision:
    requested_mode: TransferMode
    resolved_mode: TransferMode
    request_bytes: int
    request_chunks: int
    direct_h2d_bw_gbps: float
    relay_effective_bw_gbps: float
    eligible_relay_devices: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class AutoTransferSelector:
    min_chunks_for_relay: int = 2
    min_pool_bytes: int = 16 * 1024 * 1024
    relay_min_effective_bw_gbps: float = 0.0
    relay_min_direct_ratio: float = 0.0
    min_pool_speedup: float = 1.15
    min_relay_speedup: float = 1.05

    def choose(
        self,
        profile,
        request_bytes: int,
        chunk_bytes: int,
        request_chunks: int | None = None,
        direction: str = "h2d",
    ) -> AutoTransferDecision:
        request_bytes = max(0, int(request_bytes))
        chunk_bytes = max(1, int(chunk_bytes))
        if request_chunks is None:
            request_chunks = max(1, math.ceil(request_bytes / chunk_bytes)) if request_bytes else 0
        else:
            request_chunks = max(0, int(request_chunks))
        direct_bw = max(0.0, float(getattr(profile, "direct_h2d_bw_gbps", 0.0) or 0.0))
        eligible_relays = self._eligible_relays(profile, direct_bw)
        relay_bw = sum(float(getattr(relay, "effective_bw_gbps", 0.0) or 0.0) for relay in eligible_relays)

        if request_bytes == 0:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.DIRECT,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} request has no bytes",
            )
        if request_chunks < self.min_chunks_for_relay:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.DIRECT,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} request has only {request_chunks} chunk(s)",
            )
        if direct_bw <= 0.0 and relay_bw > 0.0:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.RELAY,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} direct bandwidth is unavailable",
            )
        if relay_bw <= 0.0:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.DIRECT,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} has no eligible relay paths",
            )

        direct_ms = self._transfer_ms(request_bytes, direct_bw)
        relay_ms = self._transfer_ms(request_bytes, relay_bw)
        pool_ms = self._transfer_ms(request_bytes, direct_bw + relay_bw)
        best_single_ms = min(direct_ms, relay_ms)
        pool_speedup = best_single_ms / pool_ms if pool_ms > 0.0 else 0.0
        relay_speedup = direct_ms / relay_ms if direct_ms > 0.0 and relay_ms > 0.0 else 0.0

        if request_bytes >= self.min_pool_bytes and pool_speedup >= self.min_pool_speedup:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.POOL,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"pool speedup {pool_speedup:.3f} >= {self.min_pool_speedup:.3f}",
            )
        if relay_speedup >= self.min_relay_speedup and relay_ms < direct_ms:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.RELAY,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"relay speedup {relay_speedup:.3f} >= {self.min_relay_speedup:.3f}",
            )
        return self._decision(
            TransferMode.AUTO,
            TransferMode.DIRECT,
            request_bytes,
            request_chunks,
            direct_bw,
            relay_bw,
            eligible_relays,
            "direct is the best single path",
        )

    @staticmethod
    def _transfer_ms(bytes_: int, bandwidth_gbps: float) -> float:
        if bytes_ <= 0 or bandwidth_gbps <= 0.0:
            return 0.0
        return (float(bytes_) / (bandwidth_gbps * 1e9)) * 1000.0

    def _eligible_relays(self, profile, direct_bw: float):
        relays = []
        for relay in getattr(profile, "relays", []) or []:
            effective_bw = max(0.0, float(getattr(relay, "effective_bw_gbps", 0.0) or 0.0))
            if not getattr(relay, "p2p_enabled", False) or effective_bw <= 0.0:
                continue
            if effective_bw < self.relay_min_effective_bw_gbps:
                continue
            if (
                direct_bw > 0.0
                and self.relay_min_direct_ratio > 0.0
                and effective_bw < direct_bw * self.relay_min_direct_ratio
            ):
                continue
            relays.append(relay)
        return relays

    @staticmethod
    def _decision(
        requested_mode: TransferMode,
        resolved_mode: TransferMode,
        request_bytes: int,
        request_chunks: int,
        direct_bw: float,
        relay_bw: float,
        eligible_relays,
        reason: str,
    ) -> AutoTransferDecision:
        return AutoTransferDecision(
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            request_bytes=int(request_bytes),
            request_chunks=int(request_chunks),
            direct_h2d_bw_gbps=float(direct_bw),
            relay_effective_bw_gbps=float(relay_bw),
            eligible_relay_devices=tuple(
                int(getattr(relay, "relay_device", -1)) for relay in eligible_relays
            ),
            reason=str(reason),
        )


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
    min_pool_bytes: int = 16 * 1024 * 1024
    relay_min_effective_bw_gbps: float = 0.0
    relay_min_direct_ratio: float = 0.0
    enable_dynamic_weights: bool = False
    dynamic_weight_alpha: float = 0.25

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
        self._last_resolved_transfer_mode = TransferMode.POOL
        if TransferMode(self.options.transfer_mode) is TransferMode.AUTO:
            self._last_resolved_transfer_mode = TransferMode.AUTO
        self._runtime = _turbobus.Runtime(self.options.to_native())
        self._runtime.init(self.target_gpu, self.relay_gpus)

    def profile(self, bytes: int = 256 * 1024 * 1024, force: bool = False):
        return self._runtime.profile(int(bytes), bool(force))

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
        if requested_mode is not TransferMode.AUTO:
            request_chunks = max(
                1,
                int(range_count)
                if range_count is not None
                else math.ceil(max(0, int(bytes)) / max(1, int(self.options.chunk_bytes))),
            )
            decision = AutoTransferDecision(
                requested_mode=requested_mode,
                resolved_mode=requested_mode,
                request_bytes=max(0, int(bytes)),
                request_chunks=request_chunks,
                direct_h2d_bw_gbps=0.0,
                relay_effective_bw_gbps=0.0,
                eligible_relay_devices=tuple(self.relay_gpus),
                reason="explicit transfer mode",
            )
            self._last_resolved_transfer_mode = requested_mode
            self._runtime.set_transfer_mode(_runtime_transfer_mode_value(requested_mode))
            return decision

        plan_profile = (
            self.planner_profile()
            if self.options.enable_dynamic_weights
            else self.cached_profile()
        )
        missing_direct_profile = plan_profile.direct_h2d_bw_gbps <= 0.0
        missing_relay_profile = bool(self.relay_gpus) and not plan_profile.relays
        if missing_direct_profile or missing_relay_profile:
            self.profile(self.options.profile_bytes, force=missing_relay_profile)
        selector = AutoTransferSelector(
            min_chunks_for_relay=self.options.min_chunks_for_relay,
            min_pool_bytes=self.options.min_pool_bytes,
            relay_min_effective_bw_gbps=self.options.relay_min_effective_bw_gbps,
            relay_min_direct_ratio=self.options.relay_min_direct_ratio,
        )
        plan_profile = (
            self.planner_profile()
            if self.options.enable_dynamic_weights
            else self.cached_profile()
        )
        decision = selector.choose(
            plan_profile,
            request_bytes=bytes,
            chunk_bytes=self.options.chunk_bytes,
            request_chunks=range_count,
            direction=direction,
        )
        self._last_resolved_transfer_mode = decision.resolved_mode
        self._runtime.set_transfer_mode(_runtime_transfer_mode_value(decision.resolved_mode))
        return decision

    def last_transfer_mode(self) -> TransferMode:
        return self._last_resolved_transfer_mode

    def fetch_to_gpu(self, cpu_tensor, gpu_tensor):
        _require_torch()
        bytes_to_copy = _validate_transfer_tensors(
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            target_gpu=self.target_gpu,
            direction="h2d",
        )
        self.resolve_transfer_mode(bytes_to_copy, direction="h2d")

        handle = self._runtime.fetch_to_gpu(
            int(cpu_tensor.data_ptr()),
            int(gpu_tensor.data_ptr()),
            int(bytes_to_copy),
        )
        return TransferHandle(self, handle)

    def offload_to_cpu(self, gpu_tensor, cpu_tensor):
        _require_torch()
        bytes_to_copy = _validate_transfer_tensors(
            cpu_tensor=cpu_tensor,
            gpu_tensor=gpu_tensor,
            target_gpu=self.target_gpu,
            direction="d2h",
        )
        self.resolve_transfer_mode(bytes_to_copy, direction="d2h")

        handle = self._runtime.offload_to_cpu(
            int(gpu_tensor.data_ptr()),
            int(cpu_tensor.data_ptr()),
            int(bytes_to_copy),
        )
        return TransferHandle(self, handle)

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
        self.resolve_transfer_mode(transfer_bytes, direction="h2d", range_count=range_count)
        handle = self._runtime.fetch_ranges_to_gpu(
            int(cpu_tensor.data_ptr()),
            int(source_bytes),
            int(gpu_tensor.data_ptr()),
            int(destination_bytes),
            native_ranges,
        )
        return TransferHandle(self, handle)

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
        self.resolve_transfer_mode(transfer_bytes, direction="d2h", range_count=range_count)
        handle = self._runtime.offload_ranges_to_cpu(
            int(gpu_tensor.data_ptr()),
            int(source_bytes),
            int(cpu_tensor.data_ptr()),
            int(destination_bytes),
            native_ranges,
        )
        return TransferHandle(self, handle)

    def wait(self, handle: "TransferHandle") -> None:
        self._runtime.wait(handle.native)
        handle._status = "complete"
        handle._stats = self._runtime.stats(handle.native)

    def stats(self, handle: "TransferHandle"):
        return self._runtime.stats(handle.native)

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


class TransferHandle:
    def __init__(self, runtime: Runtime, native_handle) -> None:
        self.runtime = runtime
        self.native = native_handle
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


def transfer_plan_to_dict(plan) -> dict:
    assignments = []
    for assignment in plan.assignments:
        path = assignment.path
        chunks = [
            {
                "src_offset": chunk.src_offset,
                "dst_offset": chunk.dst_offset,
                "bytes": chunk.bytes,
            }
            for chunk in assignment.chunks
        ]
        assignments.append(
            {
                "path": {
                    "kind": path.kind,
                    "direction": path.direction,
                    "target_device": path.target_device,
                    "relay_device": path.relay_device,
                    "h2d_bw_gbps": path.h2d_bw_gbps,
                    "p2p_bw_gbps": path.p2p_bw_gbps,
                    "effective_bw_gbps": path.effective_bw_gbps,
                    "enabled": path.enabled,
                },
                "chunks": chunks,
                "bytes": sum(chunk["bytes"] for chunk in chunks),
                "chunk_count": len(chunks),
            }
        )
    return {
        "total_bytes": plan.total_bytes,
        "chunk_bytes": plan.chunk_bytes,
        "assignments": assignments,
    }
