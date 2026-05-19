from __future__ import annotations

import torch

import turbobus


def main() -> None:
    target_gpu = 0
    relay_gpus = [1]
    bytes_to_copy = 256 * 1024 * 1024

    cpu_tensor = torch.empty(bytes_to_copy, dtype=torch.uint8, pin_memory=True)
    gpu_tensor = torch.empty(bytes_to_copy, dtype=torch.uint8, device=f"cuda:{target_gpu}")

    runtime = turbobus.Runtime(target_gpu=target_gpu, relay_gpus=relay_gpus)
    runtime.profile()
    handle = runtime.fetch_to_gpu(cpu_tensor, gpu_tensor)
    handle.wait()


if __name__ == "__main__":
    main()

