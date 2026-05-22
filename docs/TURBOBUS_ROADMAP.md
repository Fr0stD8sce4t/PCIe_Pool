# TurboBus Roadmap

TurboBus should reproduce the paper system, not just expose a transfer API.
The codebase already has a working transport engine; the remaining plan is to
turn it into a full single-node PCIe bandwidth pooling system for real LLM
workloads.

## Paper Target

The system must show that idle relay GPUs can lend their PCIe bandwidth to a
target GPU through local scale-up fabrics such as NVLink, NVSwitch, or
Infinity Fabric. The result should improve:

- on-demand model loading;
- vLLM KV cache save and restore;
- training offload for parameters or optimizer state.

## What The Code Must Deliver

### 1. Native Transfer Engine

Own C++/CUDA data movement only:

- pinned host memory;
- direct CPU <-> target GPU copies;
- relay CPU <-> relay GPU <-> target GPU copies;
- pooled direct plus relay transfer;
- H2D and D2H directions;
- multi-relay path planning;
- chunk scheduling, CUDA streams, events, staging slots, and path stats;
- bandwidth profiling for direct, relay PCIe, and GPU-to-GPU links.

This layer must not contain vLLM request, prefix, token, or scheduler policy.

### 2. Python Runtime API

Own the stable transfer API used by every workload:

- `Runtime` and `RuntimeOptions`;
- transfer mode selection: direct, relay, pool, auto;
- profile refresh and fallback;
- daemon reservation before relay use;
- `last_plan_dict()` and `last_auto_decision_dict()`;
- range-batched transfer for KV blocks and buckets;
- transfer stats and plan trace conversion.

Benchmarks and framework integrations should call `Runtime` instead of
duplicating transfer policy.

### 3. Daemon Resource Manager

Own shared per-node policy:

- session lifecycle;
- relay ownership and relay quota;
- transfer reservations;
- shared profile cache;
- cleanup after failures;
- cross-job bandwidth sharing policy.

CUDA data movement stays in the Runtime/native engine unless a separate
daemon-side data-plane design is explicitly requested.

### 4. Workload Integration Layer

Own framework-specific adaptation:

- vLLM KV cache connector;
- model weight bucket loading;
- training offload bucket movement;
- examples and reproduction scripts.

This layer should translate workload events into Runtime transfers and report
clear metrics. It should not implement its own PCIe pooling logic.

## Remaining Gaps To Reach The Paper

- The workload managers need to stay thin, but their batch clients should be
  exercised through one common paper-reproduction harness.
- vLLM save and restore should remain on the official connector lifecycle, not
  on example-side helper code.
- Daemon policy needs to behave like a real shared resource manager under
  contention and failure, not just a reservation stub.
- The project needs benchmark/reporting paths that measure the same outcomes
  the paper claims: TTFT, restore latency, throughput, iteration time,
  transfer bytes, path split, and fallback reason.

## Reproduction Order

1. Finish the daemon and reservation behavior until relay sharing is clearly
   controlled and explainable.
2. Close the remaining workload integration gaps so model loading, KV offload,
   and training offload all run through the same Runtime-backed client shape.
3. Build a paper-style validation harness that can run the three workloads
   end to end and report the paper metrics from one output format.
4. Keep tightening correctness and performance until the measured behavior is
   close to the paper claims on the target server.

## Non-Goals

Keep the project single-node and CUDA-focused unless a separate request asks
for more:

- no RDMA;
- no cross-node transfer;
- no HMC integration;
- no daemon-side data movement;
- no broad vLLM scheduler rewrite;
- no full KV cache state machine.
