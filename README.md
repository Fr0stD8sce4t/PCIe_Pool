# TurboBus

TurboBus is an experimental single-node PCIe bandwidth pooling system for real
large-model memory offload.

TurboBus uses fast GPU scale-up fabric such as NVLink, NVSwitch, or Infinity
Fabric to use neighboring GPUs as relays. A target GPU can borrow otherwise idle
PCIe links from relay GPUs, so several independent PCIe links behave like a
shared transfer layer instead of a single GPU-private link.

The current implementation focuses on:

- `CPU pinned memory -> target GPU`
- `CPU pinned memory -> relay GPU -> target GPU`
- chunk/block transfer planning
- CUDA stream/event execution
- path profiling
- a thin PyTorch API
- real framework KV slot APIs
- a per-node daemon state skeleton for relay quota control

The daemon manages session and relay quota state through a local Unix socket.
It is the control-plane boundary for cross-job bandwidth sharing. Client
processes still execute their own CUDA transfers, but daemon policy decides
which relay GPUs and relay capacity they may use.

Out of scope for this first version:

- HMC integration
- RDMA
- cross-node transfer
- broad vLLM/SGLang scheduler rewrites
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
The real framework connector boundary is recorded in
`docs/real_framework_connector.md`.
The first real framework target is vLLM; its integration plan is recorded in
`docs/vllm_integration.md`.

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
  --json-output benchmarks/results/gpu6_relay5.json \
  --summary-output benchmarks/results/gpu6_relay5_summary.txt
```

JSON output is compact by default: it records per-sample stats and a
`last_plan_summary` instead of dumping every chunk. Add `--include-plan` only
when debugging exact chunk placement.

At the end of each run, the benchmark prints a `COPY_SUMMARY_BEGIN` /
`COPY_SUMMARY_END` block. Copy only that block when sharing results in chat.
Use `--summary-output` to save the same block to a standalone text file.
For an existing JSON file:

```bash
python benchmarks/summarize_result.py benchmarks/results/gpu6_relay5.json
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

## KV Block Offload Benchmark

`benchmarks/kv_offload.py` validates named KV-cache shaped blocks moving
between pinned CPU memory and the target GPU. It is a transfer-layer check, not
a replacement for real vLLM testing:

```bash
python benchmarks/kv_offload.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --num-blocks 8 \
  --active-blocks 4 \
  --storage-layout packed \
  --block-bytes 16777216 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify \
  --dynamic-weights \
  --json-output benchmarks/results/kv_gpu6_relay5.json \
  --summary-output benchmarks/results/kv_gpu6_relay5_summary.txt
```

Copy only the final `COPY_SUMMARY_BEGIN` / `COPY_SUMMARY_END` block when
sharing results. In `kv_op` lines, `batch_gib_s` is the main decode-step style
throughput metric; `block_gib_s` keeps the per-block submit-to-complete view and
can include queueing time when several blocks are submitted together. Use
`--storage-layout packed` to place all KV blocks in shared CPU/GPU backing
tensors and exercise the range-batched manager path; use `separate` to keep one
tensor per block.

## Real Inference Integration

`turbobus.inference` is the framework-facing KV slot API. A framework owns the
KV cache allocation and passes TurboBus a CPU backing tensor, a GPU KV backing
tensor, and per-block byte offsets. TurboBus restores or saves those registered
slots through pooled PCIe transfer:

```python
from turbobus.inference import InferenceKVSlotAdapter

adapter = InferenceKVSlotAdapter(rt, cpu_backing, gpu_kv_backing)
adapter.register_slots(slots)
adapter.restore_prefix(["prefix0", "prefix1"])
```

`turbobus.vllm` maps vLLM V1 `GPUModelRunner.kv_caches` tensors and
`KVCacheManager` block ids into TurboBus KV slots. `turbobus.vllm_integration`
installs a narrow hook in a real vLLM process: vLLM still owns scheduling and
KV allocation, while TurboBus observes the allocated block ids and restores or
saves those real slots through pooled PCIe transfer.

On the current Qwen3-0.6B server run, vLLM exposes 28 layer KV tensors shaped
`(2, 9944, 16, 8, 128)` in `bfloat16`, so TurboBus treats each layer tensor as
one relay-poolable group.

Use `examples/vllm_introspect.py` and `examples/vllm_probe.py` only to inspect
version-specific vLLM internals before wiring a hook. They are discovery tools,
not the runtime data path.

Minimal integration shape:

```python
import turbobus
from turbobus.vllm_integration import VllmTurboBusIntegration

rt = turbobus.Runtime(target_gpu=6, relay_gpus=[5])
integration = VllmTurboBusIntegration(rt)
integration.install()

# After vLLM initializes KV cache tensors:
integration.allocate_cpu_backings(slots_per_layer=128)

# After vLLM allocates slots for a request:
integration.restore_request_prefix(request_id)
```

This is intentionally narrow: it moves real vLLM KV slots, but does not replace
vLLM's scheduler or cache manager.

Use the real vLLM restore check to prove TurboBus is moving vLLM-owned KV cache
bytes, not simulator buffers:

```bash
python examples/vllm_turbobus_restore.py \
  --model ~/huggingface/Qwen3-0.6B \
  --target-gpu 6 \
  --relay-gpus 5 \
  --prompt-repeat 64 \
  --restore-blocks 8 \
  --min-allocated-blocks 8 \
  --iterations 3 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --mode all \
  --enforce-eager \
  --log-output benchmarks/results/vllm_qwen3_restore.log
```

The script starts vLLM once, captures real `GPUModelRunner.kv_caches` tensors
and real `KVCacheManager.allocate_slots()` block ids, then runs direct, relay,
and pool save/zero/restore/verify on the same vLLM GPU slots. The vLLM engine
must run in-process so the hook can see the real Python tensor objects; the
script disables vLLM V1 multiprocessing by default.
For tensors shaped like `(2, num_blocks, ...)`, it transfers K and V lanes as
separate byte ranges for each logical KV block.
Use `--prompt-repeat`, `--restore-blocks`, and `--min-allocated-blocks` to make
vLLM allocate enough real KV blocks for bandwidth comparisons; a one-block run
is mainly a correctness smoke test and is dominated by many small ranges. Long
prompts may trigger multiple `allocate_slots()` calls for one request; TurboBus
merges those block ids before choosing the restore block list.
By default, the script treats `--target-gpu` and `--relay-gpus` as physical GPU
ids and sets `CUDA_VISIBLE_DEVICES` before importing PyTorch/vLLM, so vLLM's
`cuda:0` maps to the requested target GPU. Use `--no-map-physical-gpus` only
when the CUDA visible-device mapping is already set outside the script. vLLM
logs and the `COPY_SUMMARY` block are written to `--log-output`.

To run TurboBus from the real vLLM allocation path, use the connector entry
point. It runs one real request to save KV blocks, then a second real request
where TurboBus restore is invoked inside `KVCacheManager.allocate_slots()`:

```bash
python examples/vllm_turbobus_connector.py \
  --model ~/huggingface/Qwen3-0.6B \
  --target-gpu 6 \
  --relay-gpus 5 \
  --prompt-repeat 64 \
  --restore-blocks 8 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --mode pool \
  --enforce-eager \
  --log-output benchmarks/results/vllm_qwen3_connector_pool.log
```

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

For future connector-style code, `OffloadManager` and `KVBlockStore` are aliases
of the same store. Blocks also track `block_id`, optional CPU/GPU slots, state,
and last transfer stats. Use `prefetch_many(names)` and `evict_many(names)` when
benchmarking or simulating batched block movement.

When blocks share packed CPU/GPU backing tensors, register byte offsets and a
byte count. `prefetch_many` and `evict_many` will use the runtime range-batched
APIs automatically:

```python
cpu_backing = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
gpu_backing = torch.empty_like(cpu_backing, device=f"cuda:{target_gpu}")

store.add(
    "kv0",
    cpu_backing,
    gpu_backing,
    cpu_offset=0,
    gpu_offset=0,
    byte_count=16 * 1024 * 1024,
)
store.add(
    "kv1",
    cpu_backing,
    gpu_backing,
    cpu_offset=16 * 1024 * 1024,
    gpu_offset=16 * 1024 * 1024,
    byte_count=16 * 1024 * 1024,
)

handle = store.prefetch_many(["kv0", "kv1"])[0]
handle.wait()
```

This is intentionally thin: it does not implement a full KV-cache state
machine, but it gives the next benchmarks a named block API instead of raw
tensor copies only.

## Real Framework Connector Boundary

`turbobus.inference` is the connector API for real framework KV slots. A
framework owns the KV cache allocation and passes TurboBus a CPU backing tensor,
a GPU KV backing tensor, and per-block byte offsets. TurboBus only restores or
saves those registered slots:

```python
from turbobus.inference import InferenceKVSlotAdapter

adapter = InferenceKVSlotAdapter(rt, cpu_backing, gpu_kv_backing)
adapter.register_slots(slots)
adapter.restore_prefix(["prefix0", "prefix1"])
```

This is the intended next boundary for vLLM/SGLang-style experiments before
attempting scheduler or full KV-cache changes.

`turbobus.vllm` narrows that boundary for vLLM. It does not import vLLM
directly; a vLLM patch should extract KV cache tensors and block ids from the
local vLLM version, then pass them into `VllmKVSlotAdapter`.
Use `examples/vllm_introspect.py` to print the installed vLLM version's
KV-cache-related modules and methods before writing a version-specific patch.
Use `examples/vllm_probe.py` with a real model to observe vLLM KV cache tensor
shapes and allocated block ids without changing vLLM behavior.
