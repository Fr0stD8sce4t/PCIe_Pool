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

Prefer small, verifiable steps in this order.

1. Improve observability.
   - Print concise `path_stats` summaries in benchmark stdout.
   - Keep detailed `path_stats`, `per_relay`, and `last_plan` in JSON.
   - Make each benchmark explain which path was the bottleneck.

2. Add dynamic multipath scheduling.
   - Add an option such as `enable_dynamic_weights`, defaulting to false.
   - Maintain per-path EMA bandwidth from completed `path_stats`.
   - Let the planner use dynamic weights when enabled.
   - Reduce or disable relay paths that repeatedly underperform.
   - Preserve default transfer behavior when the option is off.

3. Complete the transfer engine direction set.
   - Add D2H direct transfer.
   - Add D2H relay transfer: `target GPU -> relay GPU -> CPU pinned`.
   - Expose Python APIs such as `offload_to_cpu`.
   - Keep H2D behavior unchanged.

4. Add batched and bucketed transfer APIs.
   - Support submitting multiple tensor/block ranges in one call.
   - Use bucket sizing and optional double buffering for large objects.
   - Keep the single contiguous tensor API as the simple baseline.

5. Add a minimal offload object layer.
   - Add an `OffloadStore` or equivalent Python API for named tensors/blocks.
   - Store data in pinned CPU backing memory.
   - Provide async `prefetch` and `evict` handles.
   - Track stats per tensor/block/path.

6. Add LLM-oriented macro benchmarks.
   - KV cache prefetch benchmark.
   - KV cache evict benchmark.
   - Model weight reload benchmark.
   - Transfer plus dummy compute overlap benchmark.
   - Report latency, effective bandwidth, path split, and bottlenecks.

7. Keep daemon work narrow.
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

