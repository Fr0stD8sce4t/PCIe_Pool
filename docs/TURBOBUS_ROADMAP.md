# TurboBus Roadmap

This roadmap pins the active project goal in the repository so future coding
sessions continue the same main line instead of drifting into isolated test or
benchmark cleanup.

## Paper Reproduction Goal

TurboBus should reproduce the core system idea from "TurboBus: Pooling PCIe
Bandwidth for LLM Workloads via Scale-Up Fabrics": use idle relay GPUs to lend
their PCIe bandwidth to a target GPU, while GPU-to-GPU movement uses NVLink,
NVSwitch, Infinity Fabric, or another local scale-up fabric.

The system must improve CPU/GPU transfer throughput for real large-model
workloads:

- on-demand model loading;
- vLLM KV cache save and restore;
- training offload for parameter or optimizer-state buckets.

Microbenchmarks are useful for debugging, but they are not the final system.

## Target Architecture

Keep the code organized into four layers.

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

Benchmarks and framework integrations should call Runtime instead of
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

## Refactor Direction

The project needs refactoring, but only refactoring that directly supports the
paper reproduction goal.

Required refactors:

- split Runtime policy helpers out of the large `turbobus/runtime.py` module;
- keep native path planning and execution independent from workload concepts;
- make daemon reservation a first-class Runtime input before relay planning;
- move vLLM save and restore into the connector lifecycle;
- keep benchmarks thin and driven by Runtime/connector APIs.

Avoid broad renames, formatting-only churn, or test-only work that does not
unblock one of the items above.

## Reproduction Workloads

### On-Demand Model Loading

Implement model weight bucket transfers from CPU pinned memory to GPU buffers.
Measure load latency, TTFT proxy, direct/relay/pool speedups, path split, and
relay pressure.

### KV Cache Offloading

Use the real vLLM `KVConnectorBase_V1` path. Save prefixes from vLLM-owned KV
cache tensors into TurboBus CPU backing, then restore later requests through
connector metadata and Runtime range transfers.

Measure restore latency, save overhead, TTFT, throughput, transfer bytes,
direct chunks, relay chunks, and auto fallback reason.

### Training Offload

Expose PyTorch bucket APIs suitable for ZeRO-Offload style parameter or
optimizer-state movement. Measure iteration time, transfer time, path split,
and overlap with computation.

## Current Baseline

The repository already has:

- direct H2D and D2H transfers;
- relay H2D and D2H transfers;
- pooled direct plus relay transfers;
- direction-aware direct/relay profiling;
- chunk planning, path stats, and dynamic weights;
- Python Runtime transfer modes and auto selection;
- range-batched transfer APIs;
- `OffloadStore` for named block movement;
- vLLM KV connector save/restore lifecycle pieces;
- daemon session, quota, and reservation foundations;
- low-level and vLLM connector benchmarks.

## Main Rule

Tests are verification, not the main deliverable. A coding turn should not end
with only tests, summaries, docs, or benchmark parsing unless that work directly
unblocks the next roadmap code task.
