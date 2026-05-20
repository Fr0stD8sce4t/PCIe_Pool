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

## Build Python Extension

```bash
pip install -e .
```

This invokes CMake with `TURBOBUS_BUILD_PYTHON=ON` and builds
`turbobus._turbobus`.

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
print(handle.stats.gib_per_second)
for path in handle.stats.path_stats:
    print(path.kind, path.direction, path.relay_device, path.gib_per_second)
print(rt.last_plan_dict())
```

Pass `relay_gpus=None` or omit it to let the runtime scan CUDA P2P-capable
relay GPUs for the target device.

## Python Benchmark

The Python benchmark can compare direct-only, relay-only, and pooled transfer
from the same API path:

```bash
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify \
  --json-output benchmarks/results/gpu6_relay5.json
```

Add `--dynamic-weights` to let repeated benchmark iterations update planner
weights from completed H2D `path_stats`.

Use physical CUDA device IDs. Avoid setting `CUDA_VISIBLE_DEVICES` in a way that
renumbers GPUs unless the runtime and PyTorch tensors use the same remapped IDs.

To sweep chunk sizes and staging slot counts:

```bash
python benchmarks/tune_transfer.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --profile-bytes 16777216 \
  --chunk-mib 4,8,16,32,64 \
  --staging-slots 2,3,4 \
  --warmup 1 \
  --iterations 5 \
  --json-output benchmarks/results/tune_gpu6_relay5.json
```

Use a tuner result directly:

```python
opts = turbobus.RuntimeOptions.from_tuning_json(
    "benchmarks/results/tune_gpu6_relay5.json"
)
rt = turbobus.Runtime(target_gpu=6, relay_gpus=[5], options=opts)
```

To skip weak relay paths, set conservative planner thresholds:

```python
opts.relay_min_direct_ratio = 0.8
opts.relay_min_effective_bw_gbps = 6.0
```

To let the planner adapt to recent per-path transfer timings, enable dynamic
weights explicitly:

```python
opts.enable_dynamic_weights = True
opts.dynamic_weight_alpha = 0.25
```

This is off by default, so existing transfer behavior stays unchanged.

For GPU-to-CPU offload into pinned host memory:

```python
handle = rt.offload_to_cpu(gpu_tensor, cpu_tensor)
handle.wait()
```

For block-style transfers inside the same CPU/GPU backing buffers, submit
multiple byte ranges in one call:

```python
ranges = [
    {"src_offset": 0, "dst_offset": 0, "bytes": 4 * 1024 * 1024},
    {"src_offset": 16 * 1024 * 1024, "dst_offset": 8 * 1024 * 1024, "bytes": 4 * 1024 * 1024},
]
handle = rt.fetch_ranges_to_gpu(cpu_tensor, gpu_tensor, ranges)
handle.wait()
```

Use `offload_ranges_to_cpu(gpu_tensor, cpu_tensor, ranges)` for the D2H
direction. This is the first batched API; it batches ranges within one source
buffer and one destination buffer.

## Named Offload Blocks

`OffloadStore` is a small Python layer for LLM-style named blocks. It keeps a
pinned CPU tensor and target-GPU tensor together, then submits async prefetch
and evict transfers through the runtime:

```python
store = turbobus.OffloadStore(rt)
store.add("kv-layer0-block0", cpu_tensor, gpu_tensor)

handle = store.prefetch("kv-layer0-block0")
handle.wait()

handle = store.evict("kv-layer0-block0")
handle.wait()
print(store.stats("kv-layer0-block0"))
```

This is intentionally thin: it does not implement a full KV-cache state
machine, but it gives the next benchmarks a named block API instead of raw
tensor copies only.
