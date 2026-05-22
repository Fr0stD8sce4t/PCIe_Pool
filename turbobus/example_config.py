from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CudaRuntimeMapping:
    physical_target_gpu: int
    physical_relay_gpus: tuple[int, ...]
    runtime_target_gpu: int
    runtime_relay_gpus: tuple[int, ...]
    cuda_visible_devices: str


def parse_gpu_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def configure_cuda_runtime_mapping(
    target_gpu: int,
    relay_gpus: str,
    *,
    map_physical_gpus: bool = True,
) -> CudaRuntimeMapping:
    physical_relays = tuple(parse_gpu_list(relay_gpus))
    visible = (int(target_gpu), *physical_relays)
    if map_physical_gpus and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        cuda_visible_devices = ",".join(str(gpu) for gpu in visible)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        return CudaRuntimeMapping(
            physical_target_gpu=int(target_gpu),
            physical_relay_gpus=physical_relays,
            runtime_target_gpu=0,
            runtime_relay_gpus=tuple(range(1, len(visible))),
            cuda_visible_devices=cuda_visible_devices,
        )
    return CudaRuntimeMapping(
        physical_target_gpu=int(target_gpu),
        physical_relay_gpus=physical_relays,
        runtime_target_gpu=int(target_gpu),
        runtime_relay_gpus=physical_relays,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
