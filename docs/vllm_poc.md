# vLLM Prefix Restore POC

The first real framework target is vLLM. Keep the first patch narrow: wire only
prefix/session KV restore to TurboBus, and leave vLLM scheduling and cache
allocation in vLLM.

## Why This Boundary

The vLLM V1 KV cache API exposes request blocks through `KVCacheBlocks`, where
the outer tuple maps to KV cache groups and each group contains block ids. The
vLLM KV cache manager also has prefix-cache methods such as
`get_computed_blocks()` and `allocate_slots()`. This matches the TurboBus POC
boundary:

```text
vLLM decides block ids and GPU slots
TurboBus restores bytes into those slots
vLLM continues decode
```

## First Hook

Target a prefix/session restore hook after vLLM knows the GPU block ids for a
request and before decode starts for the reused prefix.

The POC should:

1. Let vLLM allocate its normal GPU KV blocks.
2. Convert the vLLM KV block ids into byte offsets inside the vLLM KV cache
   backing tensor.
3. Register those slots in a TurboBus adapter.
4. Call `restore_prefix()` for only the prefix/session blocks.
5. Resume vLLM decode.
6. Compare direct, relay, and pool modes.

## Adapter Inputs

For each KV cache group, the adapter needs:

- the vLLM group id;
- the vLLM GPU KV cache backing tensor for that group;
- the CPU pinned backing tensor that stores saved prefix/session KV bytes;
- block size in bytes for that group;
- block ids selected by vLLM;
- CPU slot ids for the saved prefix blocks.

If vLLM stores key and value tensors separately, register each tensor as a
separate group or use separate slot names for key and value.

## Version Discovery

The first tested target is:

```text
vllm 0.17.1rc1.dev171+ga3e2e250f.d20260324
```

Before writing a patch for this dev build, run:

```bash
python examples/vllm_introspect.py
```

Copy only the `VLLM_INTROSPECT_BEGIN` / `VLLM_INTROSPECT_END` block. The output
lists the actual module paths, classes, and KV-cache-related methods in the
installed vLLM version, so the POC can target real source paths instead of an
older API shape.

## Success Criteria

The first vLLM POC passes when:

- a fixed prompt produces the same output with and without TurboBus restore;
- direct, relay, and pool modes restore the same block list;
- pooled restore splits chunks across direct and relay paths;
- TTFT improves for restored prefix/session requests;
- the patch touches only a narrow restore hook and adapter code.

## Non-Goals

Do not replace vLLM's scheduler.

Do not replace vLLM's KV cache block manager.

Do not implement full decode-time eviction in the first vLLM patch.

Do not assume there is only one KV cache tensor; vLLM may have multiple KV cache
groups.
