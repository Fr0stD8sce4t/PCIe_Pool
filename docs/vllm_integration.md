# vLLM Prefix Restore Integration

The first real framework target is vLLM. Keep the first patch narrow: wire only
prefix/session KV restore to TurboBus, and leave vLLM scheduling and cache
allocation in vLLM.

## Why This Boundary

The vLLM V1 KV cache API exposes request blocks through `KVCacheBlocks`, where
the outer tuple maps to KV cache groups and each group contains block ids. The
vLLM KV cache manager also has prefix-cache methods such as
`get_computed_blocks()` and `allocate_slots()`. This matches the TurboBus
integration boundary:

```text
vLLM decides block ids and GPU slots
TurboBus restores bytes into those slots
vLLM continues decode
```

## First Hook

Target a prefix/session restore hook after vLLM knows the GPU block ids for a
request and before decode starts for the reused prefix.

The integration should:

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
installed vLLM version, so the integration can target real source paths instead
of an older API shape.

For this version, introspection showed the main patch points:

- `vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_kv_cache`
- `vllm.v1.worker.gpu_model_runner.GPUModelRunner._allocate_kv_cache_tensors`
- `vllm.v1.worker.gpu_model_runner.GPUModelRunner._reshape_kv_cache_tensors`
- `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots`
- `vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks`
- `vllm.v1.core.kv_cache_manager.KVCacheManager.get_block_ids`

Before modifying vLLM behavior, run the observation-only probe:

```bash
python examples/vllm_probe.py --model <model-path-or-name> --max-tokens 8
```

This monkey patches vLLM methods only inside the probe process and prints KV
cache tensor shapes plus allocated block ids. It does not restore or modify KV
bytes.

The Qwen3-0.6B probe on the target machine showed:

```text
vllm_version=0.17.1rc1.dev171+ga3e2e250f.d20260324
model=/home/sdu/huggingface/Qwen3-0.6B
kv_cache_config num_blocks=9944 tensor_count=28 group_count=1
group0 layers=28 block_size=16 spec=FullAttentionSpec
kv_caches[0] shape=(2, 9944, 16, 8, 128) dtype=torch.bfloat16 device=cuda:0
first_request allocated block_ids=([1],)
```

For this model, one layer KV tensor block is:

```text
2 * 16 * 8 * 128 * sizeof(bfloat16) = 65536 bytes
```

`turbobus.vllm` therefore supports one TurboBus group per vLLM layer tensor.
This avoids assuming that all layers share one contiguous allocation.

`turbobus.vllm_integration` installs a narrow hook on:

- `GPUModelRunner.initialize_kv_cache`
- `KVCacheManager.allocate_slots`

The hook records real `kv_caches` tensors and allocated block ids, then exposes
`restore_request_prefix()` and `save_request_prefix()` for those real vLLM
slots.

## Real KV Slot Restore Check

`examples/vllm_turbobus_restore.py` is the first real vLLM test entry point.
It does not change vLLM scheduling. It starts vLLM, captures the real
`GPUModelRunner.kv_caches` tensors and the real block ids returned by
`KVCacheManager.allocate_slots()`, then runs this correctness loop on those
same GPU slots:

```text
save real vLLM KV block -> pinned CPU backing
zero the same vLLM GPU KV block
restore from pinned CPU backing -> same vLLM GPU block
save again and compare CPU bytes
```

Run on the GPU6/GPU5 test pair:

```bash
python examples/vllm_turbobus_restore.py \
  --model ~/huggingface/Qwen3-0.6B \
  --target-gpu 6 \
  --relay-gpus 5 \
  --prompt-repeat 64 \
  --restore-blocks 8 \
  --min-allocated-blocks 8 \
  --iterations 3 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --mode all \
  --enforce-eager \
  --log-output benchmarks/results/vllm_qwen3_restore.log
```

The script starts vLLM once and reuses the same captured KV block ids for
direct, relay, and pool modes. This keeps the comparison on the same real vLLM
KV cache slots. It disables vLLM V1 multiprocessing by default because the hook
must run in the same Python process as the vLLM engine to access the actual
tensor objects.

For vLLM tensors shaped like `(2, num_blocks, ...)`, K and V live in separate
lanes. The script maps one logical KV block into separate K/V byte ranges, so
the correctness check covers both lanes instead of assuming the block is one
contiguous byte range.

For performance runs, avoid a one-block prompt unless you only want a
correctness smoke test. One Qwen3-0.6B block maps to 28 layers x K/V lanes, so a
single logical block is many small ranges. Use `--prompt-repeat` to make vLLM
allocate more real blocks, then select several blocks with `--restore-blocks`.
For long prompts, vLLM may call `allocate_slots()` more than once for the same
request. TurboBus records and merges those allocation events before selecting
the restore block list.

By default, `--target-gpu` and `--relay-gpus` are physical GPU ids. The script
sets `CUDA_VISIBLE_DEVICES=<target>,<relays>` before importing PyTorch or vLLM,
then uses logical CUDA ids internally. This prevents vLLM from allocating
`kv_caches` on logical `cuda:0` while TurboBus expects physical `cuda:6`. Pass
`--no-map-physical-gpus` only when the visible-device mapping is already
configured outside the script. All vLLM logs, TurboBus mode lines, summary
blocks, and tracebacks are written to `--log-output`.

## Success Criteria

The first vLLM integration passes when:

- `examples/vllm_turbobus_restore.py` reports `verified=True`;
- direct, relay, and pool modes restore the same block list;
- pooled restore splits chunks across direct and relay paths;
- TTFT improves for restored prefix/session requests after the correctness
  check is stable;
- the patch touches only a narrow restore hook and adapter code.

## Non-Goals

Do not replace vLLM's scheduler.

Do not replace vLLM's KV cache block manager.

Do not implement full decode-time eviction in the first vLLM patch.

Do not assume there is only one KV cache tensor; vLLM may have multiple KV cache
groups.
