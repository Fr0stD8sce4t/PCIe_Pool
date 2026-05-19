# TurboBus

TurboBus is an experimental single-node PCIe bandwidth pooling runtime for LLM
offload-style transfers.

The current MVP focuses on:

- `CPU pinned memory -> target GPU`
- `CPU pinned memory -> relay GPU -> target GPU`
- chunk/block transfer planning
- CUDA stream/event execution
- path profiling
- a thin PyTorch API
- a minimal daemon state skeleton for relay quota control

The daemon first version only manages session and relay quota state through a
local Unix socket. It does not move GPU pointers across processes.

Out of scope for this first version:

- HMC integration
- RDMA
- cross-node transfer
- vLLM/SGLang deep integration
- a full KV cache state machine

## Layout

```text
cpp/
  include/turbobus/     C++ public headers
  src/                  CUDA/C++ implementation
turbobus/               Python wrapper and daemon skeleton
benchmarks/             Benchmark entry points, not run by default
examples/               Minimal usage examples
docs/                   Design notes
references/             Cloned reference repositories
```

Initial server benchmark notes are recorded in `docs/benchmark_notes.md`.

## Minimal Python Shape

```python
import torch
import turbobus

target_gpu = 0
relay_gpus = [1]

cpu_tensor = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{target_gpu}")

rt = turbobus.Runtime(target_gpu=target_gpu, relay_gpus=relay_gpus)
rt.profile()
handle = rt.fetch_to_gpu(cpu_tensor, gpu_tensor)
handle.wait()
```

This repository has not been built or tested yet in the current environment.
