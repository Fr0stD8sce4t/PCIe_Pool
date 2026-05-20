# TurboBus Project Instructions

TurboBus is a research prototype for single-node PCIe bandwidth pooling for
LLM memory offload tasks. Treat the project as an LLM offload transfer engine,
not only as a memcpy benchmark tool.

The current core idea is:

- use direct `CPU pinned memory -> target GPU` transfer;
- use relay `CPU pinned memory -> relay GPU -> target GPU` transfer;
- split large tensor/block transfers across direct and relay paths;
- use measured path performance to improve scheduling decisions;
- expose enough Python APIs and benchmark data to evaluate LLM offload use
  cases such as KV cache prefetch/evict and model weight reload.

## Scope

Keep the first implementation single-node and CUDA-focused.

In scope:

- pinned host memory;
- H2D and D2H transfers;
- direct and relay GPU paths;
- chunk/bucket scheduling;
- CUDA streams/events;
- path profiling and tuning;
- per-path stats and plan tracing;
- PyTorch tensor API;
- lightweight LLM offload benchmarks;
- a daemon used only for resource coordination.

Out of scope unless explicitly requested:

- RDMA;
- cross-node transfer;
- HMC integration;
- daemon-side CUDA IPC data movement;
- full vLLM/SGLang patching;
- full KV cache state machine.

## Reference Projects

Use `references/` for design guidance only. Do not edit files under
`references/`.

Important lessons from local references:

- LMCache: expose KV-cache style lifecycle APIs such as load, wait, save, and
  wait-for-save; design for vLLM/SGLang-style connectors later.
- Mooncake: model TurboBus as a transfer engine with batched data movement,
  topology-aware path choice, and KV-cache-oriented benchmarks.
- nvbandwidth: profiling should cover multiple copy patterns, bidirectional
  behavior, JSON output, and median-based reporting.
- checkpoint-engine: large object movement should use buckets, double buffering,
  and pipelined execution when useful.
- YALIS and MoE-Infinity: offload APIs should support prefetch modes, pinned CPU
  storage, preallocated GPU buffers, and overlap with computation.

## Development Roadmap

Prefer small, verifiable steps. The project should be connector-ready for real
LLM inference later, but should not patch vLLM/SGLang or implement a full KV
cache manager before the core behavior is tested.

Current completed baseline:

- H2D and D2H direct transfers.
- H2D and D2H relay transfers through a P2P-capable relay GPU.
- Pooled direct + relay transfer.
- Chunk planning, path stats, dynamic weights, and JSON/copy summaries.
- Range batched transfer APIs.
- A minimal named-block `OffloadStore`.
- `bandwidth_pool.py` and `kv_offload.py` benchmarks.

Next steps:

1. Make the Python offload object layer reusable by future connectors.
   - Keep the C++/CUDA runtime as a transfer engine only.
   - Keep request, decode-step, and KV-cache policy out of the CUDA executor.
   - Add connector-shaped concepts at the Python layer: block id, CPU backing,
     GPU slot, block state, async handle, and per-block stats.
   - Keep `OffloadStore` backward compatible while it evolves toward an
     `OffloadManager` / `KVBlockStore` style API.

2. Add batch block operations on top of the existing transfer APIs.
   - `prefetch_many(names)` and `evict_many(names)` are the stable benchmark
     and future connector entry points.
   - The simple per-block path is available and tested.
   - Packed CPU/GPU backing buffers with per-block offsets are supported by
     `OffloadStore`, the simulator, and the KV benchmark. Shared backing buffers
     use `fetch_ranges_to_gpu` and `offload_ranges_to_cpu` for many-block
     transfers.
   - Keep the per-block path as a fallback for non-packed tensors.

3. Add an inference offload simulator before patching real frameworks.
   - Simulate request arrival, block ownership, decode steps, GPU block capacity,
     prefetch, eviction, and transfer stall.
   - Non-overlap and Python-sleep overlap simulator paths are available.
   - The simulator uses the packed range-batch manager path for
     KV-cache-style backing buffers.
   - Native CUDA dummy compute is available for overlap experiments through
     `--compute-impl cuda`.
   - CUDA dummy compute overlap has been validated with heavier native kernels.
   - A request-level workload simulator is available as
     `benchmarks/inference_workload_sim.py`; it models request arrival, prefill,
     decode steps, scheduling, TTFT, request latency, and GPU KV cache hit rate.
   - The initial burst workload result is recorded and shows enough KV pressure
     to compare direct, relay, and pool modes.
   - Next priority: add workload presets such as `light`, `pressure`, and
     `long_context`, then add a prefill restore mode that can model prompt KV
     blocks loaded from CPU instead of always produced on GPU.
   - Compare direct, relay, and pool modes using the same manager API that a
     future vLLM/SGLang connector would call.

4. Only after simulator results are stable, design real connector prototypes.
   - Start with narrow adapter designs for vLLM, SGLang, or LMCache-style
     integration.
   - Do not vendor or heavily patch external inference frameworks in this repo
     unless explicitly requested.

5. Keep daemon work narrow.
   - Use the daemon for relay quota, session tracking, profile cache sharing,
     and multi-process coordination.
   - Do not move GPU pointers or CUDA IPC through the daemon in this phase.

## Coding Rules

- Keep changes surgical and tied to the current step.
- Prefer existing C++/CUDA/Python patterns in this repository.
- Keep default transfer behavior unchanged unless the user explicitly asks to
  change it.
- Add tests or benchmark checks for planner, stats, and Python API changes.
- If C++/pybind/CUDA files change, remind the user to reinstall the package
  before server testing:

```bash
pip uninstall -y turbobus
rm -rf build build-test build-temp *.egg-info turbobus/_turbobus*.so
pip install -e .
```

- Prefer GPU pair target `6`, relay `5` for server benchmark examples unless
  the user provides another topology.
- Avoid GPU 0 in suggested benchmark commands.
