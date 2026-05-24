# TurboBus

TurboBus is being rebuilt as a paper-reproduction system for pooling idle PCIe
bandwidth in multi-GPU servers.

The target system is centered on:

- a privileged per-node daemon;
- cross-job relay GPU discovery and scheduling;
- direct, relay, and pooled chunked transfers;
- application isolation and relay leases;
- backend support for CUDA/NVIDIA and ROCm/AMD scale-up fabrics;
- framework adapters for vLLM, model loading, and training offload.

## Repository Map

- `cpp/`: native transfer engine, profiler, planner, and pybind module.
- `turbobus/`: Python client API, daemon control plane, and framework adapters.
- `docs/`: active next steps, roadmap, and progress notes.
- `benchmarks/`: workload and evaluation scripts.
- `test/`: Python and native tests.

## Current Direction

The active direction is the paper-parity plan in `AGENTS.md` and `docs/`.
Start with Phase 0, which realigns core code, tests, benchmarks, examples,
exports, and adapters around daemon-first scheduling.
