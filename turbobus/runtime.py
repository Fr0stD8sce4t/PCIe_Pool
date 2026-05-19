from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


@dataclass
class RuntimeOptions:
    chunk_bytes: int = 16 * 1024 * 1024
    staging_slots: int = 2
    enable_peer_access: bool = True

    def to_native(self):
        _require_extension()
        options = _turbobus.RuntimeOptions()
        options.chunk_bytes = self.chunk_bytes
        options.staging_slots = self.staging_slots
        options.enable_peer_access = self.enable_peer_access
        return options


def _require_extension() -> None:
    if _turbobus is None:
        raise RuntimeError(
            "turbobus native extension is not available. Build cpp/_turbobus "
            "before using the runtime."
        ) from _IMPORT_ERROR


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor based TurboBus APIs")


class Runtime:
    def __init__(
        self,
        target_gpu: int,
        relay_gpus: Iterable[int],
        options: RuntimeOptions | None = None,
    ) -> None:
        _require_extension()
        self.target_gpu = int(target_gpu)
        self.relay_gpus = [int(gpu) for gpu in relay_gpus]
        self.options = options or RuntimeOptions()
        self._runtime = _turbobus.Runtime(self.options.to_native())
        self._runtime.init(self.target_gpu, self.relay_gpus)

    def profile(self, bytes: int = 256 * 1024 * 1024):
        return self._runtime.profile(int(bytes))

    def fetch_to_gpu(self, cpu_tensor, gpu_tensor):
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
        if gpu_tensor.device.index != self.target_gpu:
            raise ValueError("gpu_tensor must be on the runtime target_gpu")
        if not cpu_tensor.is_contiguous() or not gpu_tensor.is_contiguous():
            raise ValueError("cpu_tensor and gpu_tensor must be contiguous")

        bytes_to_copy = cpu_tensor.numel() * cpu_tensor.element_size()
        if gpu_tensor.numel() * gpu_tensor.element_size() < bytes_to_copy:
            raise ValueError("gpu_tensor is smaller than cpu_tensor")

        handle = self._runtime.fetch_to_gpu(
            int(cpu_tensor.data_ptr()),
            int(gpu_tensor.data_ptr()),
            int(bytes_to_copy),
        )
        return TransferHandle(self, handle)

    def wait(self, handle: "TransferHandle") -> None:
        self._runtime.wait(handle.native)
        handle._status = "complete"


class TransferHandle:
    def __init__(self, runtime: Runtime, native_handle) -> None:
        self.runtime = runtime
        self.native = native_handle
        self._status = "submitted"
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
