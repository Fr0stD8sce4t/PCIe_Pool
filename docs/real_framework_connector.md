# Real Framework Connector Boundary

TurboBus should enter real inference frameworks through a narrow KV slot restore
adapter first. The adapter should not own request scheduling or the full KV cache
state machine.

## First Target

Start with prefix/session KV restore:

```text
pinned CPU prefix KV backing -> existing target-GPU KV cache slots
```

This is the smallest useful real inference path because it affects TTFT and
uses the same H2D pooled transfer path already validated by low-level transfer
benchmarks.

## Framework Responsibilities

A real framework connector must provide:

- the target GPU id;
- the GPU KV cache backing tensor or tensors;
- the CPU pinned backing tensor that stores saved prefix/session KV blocks;
- a stable block id for each KV block;
- CPU byte offset, GPU byte offset, and byte count for each block;
- the restore timing, usually before decode starts for a reused prefix/session.

TurboBus should not decide:

- which request runs next;
- which KV blocks belong to a prefix;
- which GPU slot is allocated;
- when the framework may evict a block.

## TurboBus Responsibilities

TurboBus registers the framework-provided slot mapping with `OffloadManager`:

```python
manager.add(
    name,
    cpu_backing,
    gpu_kv_backing,
    block_id=framework_block_id,
    cpu_offset=cpu_offset,
    gpu_offset=gpu_offset,
    byte_count=byte_count,
)
```

The restore path is:

```python
handles = manager.prefetch_many(block_names)
manager.wait_many(block_names)
```

The save path is:

```python
handles = manager.evict_many(block_names)
manager.wait_many(block_names)
```

## vLLM Target

The first real framework target is vLLM. The vLLM-specific plan is recorded in
`docs/vllm_integration.md`; the package APIs are `turbobus.vllm` and
`turbobus.vllm_integration`.

## First Real Framework Integration Shape

The first vLLM integration should patch only a narrow prefix/session restore
hook:

1. Let vLLM allocate its normal GPU KV slots.
2. Export the KV cache tensor and slot offsets to a TurboBus adapter.
3. Register prefix/session blocks with `OffloadManager`.
4. Call `prefetch_many()` before decode starts.
5. Verify that vLLM can continue decode after restore.
6. Compare direct, relay, and pool modes using TTFT and restore latency.

## Metrics

Record:

- restore p50/p95 latency;
- direct and relay chunk counts;
- TTFT p50/p95;
- decode token correctness or text equality for a fixed seed;
- pool/direct restore speedup;
- pool/direct TTFT improvement.

## Non-Goals

Do not start by replacing the framework scheduler.

Do not implement full decode-time eviction until prefix/session restore is
correct.

Do not make the daemon own CUDA pointers in this phase.
