# TurboBus Project Instructions

TurboBus is a research prototype for single-node PCIe bandwidth pooling for
real large-model memory offload tasks. Treat the project as a daemon-managed
LLM systems project, not as a memcpy benchmark or inference simulator.

The current core idea is:

- use direct `CPU pinned memory -> target GPU` transfer;
- use relay `CPU pinned memory -> relay GPU -> target GPU` transfer;
- split large tensor/block transfers across direct and relay paths;
- use measured path performance to improve scheduling decisions;
- expose production-shaped Python APIs for real inference/training systems:
  on-demand model loading, vLLM KV cache offload/restore, and training offload.

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
- real framework KV slot adapters;
- vLLM integration hooks;
- a per-node daemon for relay ownership, relay quota, and cross-job bandwidth
  sharing policy.

Out of scope unless explicitly requested:

- RDMA;
- cross-node transfer;
- HMC integration;
- daemon-side CUDA IPC data movement;
- broad vLLM/SGLang scheduler rewrites;
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

Prefer small, verifiable steps. The project should move toward the TurboBus
paper system: daemon-managed PCIe bandwidth pooling through relay GPUs, with
real large-model integration points instead of simulated inference workloads.

Current completed baseline:

- H2D and D2H direct transfers.
- H2D and D2H relay transfers through a P2P-capable relay GPU.
- Pooled direct + relay transfer.
- Chunk planning, path stats, dynamic weights, and JSON/copy summaries.
- Range batched transfer APIs.
- A minimal named-block `OffloadStore`.
- `bandwidth_pool.py` and `kv_offload.py` low-level validation benchmarks.
- `turbobus.inference` and `turbobus.vllm` framework-facing APIs.

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
     `OffloadStore`, the framework adapters, and the KV benchmark. Shared
     backing buffers use `fetch_ranges_to_gpu` and `offload_ranges_to_cpu` for
     many-block transfers.
   - Keep the per-block path as a fallback for non-packed tensors.

3. Build the real vLLM KV offload integration path.
   - vLLM is the first real inference framework target.
   - `turbobus.inference` defines framework KV slot registration and
     restore/save.
   - `turbobus.vllm` defines vLLM layer KV groups and block references.
   - `turbobus.vllm_integration` installs a narrow real-vLLM hook that observes
     `GPUModelRunner.kv_caches` and `KVCacheManager.allocate_slots()` results.
   - `examples/vllm_introspect.py` and `examples/vllm_probe.py` are temporary
     discovery tools for version-specific vLLM internals, not simulator code.
   - The current server vLLM is
     `0.17.1rc1.dev171+ga3e2e250f.d20260324`.
   - The Qwen3-0.6B probe showed 28 layer KV tensors shaped
     `(2, 9944, 16, 8, 128)` in bfloat16, with 65,536 bytes per layer block.
   - Next priority: implement a real vLLM restore/save hook using vLLM-owned
     `GPUModelRunner.kv_caches` tensors and block ids from `KVCacheManager`.
     The first hook exists in `turbobus.vllm_integration`; keep evolving it
     toward correctness and measurement inside an actual vLLM generation run.
   - Compare direct, relay, and pool modes using the same manager API that a
     future vLLM/SGLang connector would call.

4. Add production-shaped offload clients for the three paper workloads.
   - On-demand model loading: restore model-weight buckets into GPU memory.
   - KV cache offloading: vLLM prefix/session save and restore.
   - Training offload: expose block/bucket transfer hooks suitable for
     ZeRO-Offload style optimizer or parameter movement.

5. Expand daemon work toward the paper architecture.
   - Use the daemon for relay quota, session tracking, profile cache sharing,
     and multi-process coordination.
   - Add transfer reservation APIs for relay bandwidth sharing across jobs.
   - Keep process isolation: applications should borrow relay PCIe bandwidth
     through daemon policy without directly controlling another job's GPU.

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
