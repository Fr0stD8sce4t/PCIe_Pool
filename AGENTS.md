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

### Active Roadmap Files

At the start of each coding turn, read these files and use them to choose the
next implementation task:

1. `docs/TURBOBUS_ROADMAP.md`
2. `docs/NEXT_STEPS.md`
3. `docs/PROGRESS.md`

The task under `## Current` in `docs/NEXT_STEPS.md` is the default task. Do not
replace this roadmap with isolated test, docs, summary, or benchmark parsing
work unless that work directly unblocks the current code task.

After each coding turn:

- update `docs/PROGRESS.md` with the work completed, verification performed,
  commit id if one was created, and any remaining risk;
- update `docs/NEXT_STEPS.md` when an item is completed or blocked;
- when `## Current` is completed, move it to `## Completed` and promote the
  first `## Upcoming` item to `## Current`.

Tests are verification, not the main deliverable.

### Project Direction

Advance TurboBus from a working research prototype into a usable KV/tensor
transfer system for real inference frameworks. Do not let benchmark scripts
become the system. Keep the project organized into three layers:

1. Native transfer engine.
   - Own C++/CUDA transfer execution, chunk planning, peer access, staging
     buffers, direct/relay/pool paths, profiling, and low-level stats.
   - Do not put vLLM request, prefix, token, or scheduler policy here.

2. Python runtime API.
   - Own `Runtime`, `RuntimeOptions`, transfer mode selection, profile cache,
     transfer stats, `last_plan_dict()`, `last_auto_decision_dict()`, and
     `batch_transfer_mode()`.
   - Keep this API stable enough for framework integrations and benchmarks to
     share the same entry points.

3. Framework integration layer.
   - Own vLLM connector logic, framework KV slot adapters, examples, and
     framework-specific compatibility code.
   - Keep integration code thin: translate framework lifecycle events into
     TurboBus runtime calls, then report clear timing and transfer stats.

Project priorities:

1. Stabilize the Runtime API.
   - Make direct, relay, pool, and auto transfer behavior clear and
     explainable.
   - Keep profile refresh, fallback behavior, and plan reporting test-covered.
   - Prefer one shared runtime path over duplicated benchmark-only logic.

2. Engineer the vLLM connector path.
   - Define supported vLLM versions and configuration keys.
   - Keep save/restore prefix lifecycle clear: request metadata, allocation,
     connector metadata, worker transfer, prefix registration, completion, and
     cleanup.
   - Reduce example-side special cases as connector lifecycle support matures.

3. Grow daemon and multi-process resource management.
   - Use the daemon for session lifecycle, relay ownership, relay quota,
     transfer reservations, shared profile cache, and cleanup after failures.
   - Keep CUDA data movement in the runtime unless a daemon-side movement
     design is explicitly requested.

4. Prove value with real workloads.
   - Benchmarks should answer when TurboBus helps, when it does not, and why.
   - Track TTFT, prefix restore latency, save overhead, throughput impact, relay
     GPU pressure, transfer bytes, chunks, path choice, and fallback reason.
   - Use microbenchmarks to debug transfer behavior, but prioritize real vLLM
     generation paths for project decisions.

Near-term code direction:

1. Align README, examples, and benchmarks around the stable Runtime API.
2. Consolidate vLLM connector configuration for mode, target GPU, relay GPUs,
   prefix/session keys, restore/save flags, and prefix capacity.
3. Keep one minimal real vLLM demo path that saves a prefix, restores it in a
   later request, and reports connector events plus timing.
4. Return to auto/chunk/profile strategy only after real workload results show
   a specific gap.

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

### Immediate vLLM Connector Goal

The next connector milestone is a real vLLM `KVConnectorBase_V1` save/restore
loop. Restore already runs through the official connector path. Save must also
move into the connector lifecycle instead of being driven by example-side
allocation hooks.

Required shape:

1. First request passes `kv_transfer_params` such as `turbobus.do_save`,
   `turbobus.prefix_key`, `turbobus.save_blocks`, and
   `turbobus.matched_tokens`.
2. `TurboBusConnector.update_state_after_alloc()` or
   `build_connector_meta(scheduler_output)` records the vLLM block ids that
   should be saved.
3. `TurboBusConnector.build_connector_meta()` sends save and restore metadata
   through vLLM's connector metadata path.
4. Worker-side connector code saves from vLLM-owned KV cache tensors into
   connector-managed pinned CPU backing.
5. The connector registers the saved prefix internally after save completes.
6. `TurboBusConnector.request_finished()` delays block release only for save
   requests that were actually queued, and `get_finished()` reports completed
   saves.
7. A later request passes `turbobus.do_restore` and restores that saved prefix
   through `get_num_new_matched_tokens()`, `update_state_after_alloc()`,
   `build_connector_meta()`, and `start_load_kv()`.

The example should not call `register_saved_prefix()` or use the old
`VllmTurboBusConnector` save path once this loop is in place. It should only
create vLLM requests with connector `kv_transfer_params` and report the
connector's emitted save/restore events.

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
