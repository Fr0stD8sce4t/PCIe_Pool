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
The narrow real inference POC plan is recorded in
`docs/real_inference_poc.md`.
The real framework connector boundary is recorded in
`docs/real_framework_connector.md`.
The first real framework target is vLLM; its POC plan is recorded in
`docs/vllm_poc.md`.

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

`benchmarks/kv_offload.py` uses `OffloadStore` to simulate named KV-cache
blocks moving between pinned CPU memory and the target GPU:

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

## Inference Offload Simulator

`benchmarks/inference_offload_sim.py` is a lightweight simulator for future
LLM connector behavior. It does not patch vLLM or SGLang. It simulates request
KV blocks, decode steps, limited GPU block residency, prefetch, eviction, and
transfer stall using the same `OffloadManager` API as the KV benchmark:

```bash
python benchmarks/inference_offload_sim.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --requests 4 \
  --blocks-per-request 8 \
  --blocks-per-step 4 \
  --gpu-block-capacity 4 \
  --access-pattern round_robin \
  --working-set-blocks 8 \
  --seed 1 \
  --storage-layout packed \
  --block-bytes 16777216 \
  --decode-steps 32 \
  --compute-ms 0 \
  --compute-impl cuda \
  --cuda-compute-elements 16777216 \
  --cuda-compute-iterations 64 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --mode all \
  --dynamic-weights \
  --json-output benchmarks/results/infer_sim_gpu6_relay5.json \
  --summary-output benchmarks/results/infer_sim_gpu6_relay5_summary.txt
```

By default each step waits for eviction and prefetch before optional dummy
compute. Add `--overlap-compute` to run dummy compute concurrently with
transfer. `--compute-impl sleep --compute-ms N` uses a Python sleep scheduling
model. `--compute-impl cuda` runs a native CUDA dummy kernel on a preallocated
target-GPU tensor; tune its runtime with `--cuda-compute-elements` and
`--cuda-compute-iterations`. The default access pattern and GPU block capacity
are chosen to create capacity pressure, so the run should exercise both
prefetch and eviction. Use `tokens_s`, `step_p50_ms`, and `transfer_p50_ms` in
the copy summary as the main metrics. The `sim_scenario` line describes what
the run is modeling so the saved summary can be read without the full command.
Use `--storage-layout packed` to store all simulated KV blocks in shared
CPU/GPU backing tensors and exercise the range-batched manager path; use
`separate` to keep one tensor per block.

`benchmarks/inference_workload_sim.py` adds a request-level workload model on
top of the same manager API. It keeps the benchmark self-contained while adding
request arrival, prefill, decode steps, scheduling, TTFT, request latency, and
GPU KV cache hit-rate metrics:

```bash
python benchmarks/inference_workload_sim.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --preset pressure \
  --scheduler round_robin \
  --storage-layout packed \
  --prefill-mode restore_from_cpu \
  --compute-impl cuda \
  --cuda-compute-iterations 2048 \
  --overlap-compute \
  --mode all \
  --dynamic-weights
```

Use `--preset light`, `--preset pressure`, or `--preset long_context` for
repeatable workload shapes. `--prefill-mode produce_kv_on_gpu` models prompt KV
being produced on the target GPU; `--prefill-mode restore_from_cpu` models
prompt/prefix KV blocks being loaded from pinned CPU backing memory before
decode.

`benchmarks/prefix_restore_poc.py` is the first narrow real-inference POC
boundary. It focuses only on prefix/session KV restore from pinned CPU backing
memory into target-GPU KV slots, with optional CUDA dummy compute beside the
restore. It does not patch vLLM or SGLang:

```bash
python benchmarks/prefix_restore_poc.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --sessions 4 \
  --blocks-per-session 8 \
  --restore-blocks 8 \
  --iterations 5 \
  --storage-layout packed \
  --block-bytes 16777216 \
  --compute-impl cuda \
  --cuda-compute-iterations 2048 \
  --overlap-compute \
  --mode all \
  --dynamic-weights \
  --summary-output benchmarks/results/prefix_restore_poc_summary.txt
```

Use `poc_mode.restore_gib_s`, `restore_p50_ms`, `step_p50_ms`, and the
direct/relay chunk counts as the main POC metrics.

`benchmarks/real_model_sidecar_restore.py` is the next step after the POC. It
runs a real PyTorch `TransformerEncoderLayer` on the target GPU while TurboBus
restores prefix/session KV-shaped blocks beside it. This still does not patch a
real inference framework, but it replaces the dummy compute with real PyTorch
model kernels:

```bash
python benchmarks/real_model_sidecar_restore.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --sessions 4 \
  --blocks-per-session 8 \
  --restore-blocks 8 \
  --iterations 5 \
  --storage-layout packed \
  --block-bytes 16777216 \
  --model-layers 1 \
  --model-batch-size 1 \
  --model-seq-len 128 \
  --model-hidden-size 4096 \
  --model-heads 32 \
  --model-ff-size 11008 \
  --model-dtype float16 \
  --overlap-compute \
  --mode all \
  --dynamic-weights
```

Use `sidecar_mode.restore_gib_s`, `step_p50_ms`, `model_p50_ms`, and
direct/relay chunks to compare direct, relay, and pool modes.

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

`examples/framework_kv_slot_adapter.py` shows the first connector shape for a
real framework POC. A framework owns the KV cache allocation and passes
TurboBus a CPU backing tensor, a GPU KV backing tensor, and per-block byte
offsets. TurboBus only restores or saves those registered slots:

```python
adapter = FrameworkKVSlotAdapter(rt, cpu_backing, gpu_kv_backing)
adapter.register_slots(slots)
adapter.restore_prefix(["prefix0", "prefix1"])
```

This is the intended next boundary for vLLM/SGLang-style experiments before
attempting scheduler or full KV-cache changes.

`examples/vllm_kv_slot_adapter.py` narrows that boundary for vLLM. It does not
import vLLM directly; a vLLM patch should extract KV cache tensors and block ids
from the local vLLM version, then pass them into `VllmKVSlotAdapter`.
