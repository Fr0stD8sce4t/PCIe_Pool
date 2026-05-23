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
- `docs/`: rewrite plan, roadmap, and implementation notes.
- `benchmarks/`: workload and evaluation scripts.
- `test/`: Python and native tests.

## Current Direction

The near-term work is to define the new daemon/client/worker protocol, then
rebuild the planner, backend, and framework integration layers on top of it.

The first complete target is a daemon-managed transfer system that can be used
by real LLM workloads without requiring the client process to own the relay
GPU directly.
