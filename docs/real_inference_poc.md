# Narrow Real Inference POC

TurboBus should reach a real inference framework through a small prefix/session
KV restore path first, not by rewriting a scheduler.

## Goal

Prove that the existing pooled PCIe transfer path can speed up the part of a
real inference request that restores previously stored KV blocks from pinned CPU
memory into target-GPU KV slots.

The first POC should answer:

- can a connector call `OffloadManager.prefetch_many()` with real KV-slot sized
  blocks;
- do direct, relay, and pool modes report the expected chunk split;
- does pool mode reduce prefix restore time and TTFT-like wait time;
- can restore overlap with target-GPU work without changing the transfer API.

## Non-Goals

Do not start by patching the full vLLM or SGLang scheduler.

Do not implement a complete KV cache state machine in TurboBus.

Do not make the daemon move CUDA pointers or own CUDA IPC in this POC.

## Boundary

TurboBus owns:

- pinned CPU backing memory;
- target-GPU KV slot backing memory;
- block registration with offsets and byte counts;
- direct, relay, and pooled H2D/D2H transfers;
- transfer stats and copy summaries.

The future inference framework owns:

- request scheduling;
- KV allocation policy;
- which prefix/session blocks should be restored;
- when decode may start.

The connector boundary is therefore:

```python
manager.add(
    block_name,
    cpu_backing,
    gpu_backing,
    cpu_offset=...,
    gpu_offset=...,
    byte_count=block_bytes,
)
handles = manager.prefetch_many(prefix_block_names)
manager.wait_many(prefix_block_names)
```

## POC Phases

Phase 0 is the current next step: a framework-adjacent restore harness.

It uses real-shaped packed KV backing tensors and restores prefix/session blocks
through `OffloadManager`. Optional CUDA dummy compute runs beside the restore to
check overlap. This still does not depend on vLLM/SGLang internals, so the
result is easy to reproduce and debug.

Phase 0 status: passed on the GPU6 target + GPU5 relay test pair. The packed
prefix restore POC verified restored data, split pooled chunks evenly across
direct and relay paths, and improved restore throughput from 7.105 GiB/s direct
to 13.910 GiB/s pool, a 1.958x speedup.

Phase 1 should replace the synthetic GPU backing with addresses that match a
real framework KV slot layout. Keep the scheduler outside TurboBus.

Phase 2 can add a narrow vLLM/SGLang or LMCache-style connector that only handles
prefix/session restore and save. Broader decode-time eviction can wait until the
restore path is correct.

## Metrics

Record these for direct, relay, and pool modes:

- restore batch p50/p95 latency;
- restore GiB/s;
- direct and relay chunk counts;
- compute p50/p95 when sidecar compute is enabled;
- step p50/p95 when transfer and compute are combined;
- pool/direct and pool/relay speedups;
- optional restored data correctness.

## Success Criteria

The first POC is useful when:

- the same block list works in direct, relay, and pool modes;
- pool mode splits chunks across direct and relay paths;
- restored data verifies when `--verify` is enabled;
- pool mode shows a meaningful restore latency reduction over direct mode;
- the benchmark emits a compact summary that includes the scenario being tested.
