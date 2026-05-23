# TurboBus Benchmark Notes

## Benchmark Goal

Measure whether the daemon-managed relay system improves real LLM workloads
without breaking isolation or fallback behavior.

## Required Scenarios

- direct only;
- relay only;
- pooled direct plus relay;
- single job;
- multiple jobs contending for the same relay resources;
- relay idle versus busy;
- target GPU PCIe saturated versus unsaturated;
- vLLM KV prefix save/restore;
- model loading;
- training offload;
- CUDA backend;
- ROCm backend when available.

## Required Metrics

- total bytes;
- direct bytes;
- relay bytes;
- direct chunks;
- relay chunks;
- per-relay chunk counts;
- latency;
- submit-to-complete latency;
- effective bandwidth;
- queueing time;
- lease wait time;
- fallback reason;
- fairness under contention;
- framework-level impact.

## Output Shape

Each benchmark should emit structured summary data that can be compared across
backends and workloads.

Preferred fields:

- scenario;
- mode;
- target GPU;
- relay GPUs;
- bytes;
- chunks;
- latency;
- bandwidth;
- direct/relay split;
- profile source;
- lease status;
- fallback status.

## Benchmark Rules

- Keep microbenchmarks small and focused on transfer behavior.
- Use workload benchmarks for product decisions.
- Do not let benchmark scripts become the system design.
- Reuse the same public client API that future framework adapters will use.
