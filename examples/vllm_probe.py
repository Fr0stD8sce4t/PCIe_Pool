from __future__ import annotations

import argparse
import functools
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Observe vLLM KV cache tensors and block ids")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    install_patches()

    from vllm import LLM, SamplingParams

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    start = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    init_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    generate_ms = (time.perf_counter() - start) * 1000.0

    print("VLLM_PROBE_BEGIN")
    print("llm_init_ms", f"{init_ms:.3f}")
    print("generate_ms", f"{generate_ms:.3f}")
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        print(
            "output",
            "request_id",
            getattr(output, "request_id", "unknown"),
            "prompt_tokens",
            len(getattr(output, "prompt_token_ids", []) or []),
            "generated_text",
            repr(text[:200]),
        )
    print("VLLM_PROBE_END")


def install_patches() -> None:
    from vllm.v1.core import kv_cache_manager as manager_module
    from vllm.v1.worker import gpu_model_runner as runner_module

    patch_initialize_kv_cache(runner_module.GPUModelRunner)
    patch_allocate_slots(manager_module.KVCacheManager)
    patch_get_computed_blocks(manager_module.KVCacheManager)


def patch_initialize_kv_cache(cls) -> None:
    original = cls.initialize_kv_cache

    @functools.wraps(original)
    def wrapped(self, kv_cache_config):
        result = original(self, kv_cache_config)
        print("probe_initialize_kv_cache", describe_kv_cache_config(kv_cache_config))
        describe_runner_kv_attrs(self)
        return result

    cls.initialize_kv_cache = wrapped


def patch_allocate_slots(cls) -> None:
    original = cls.allocate_slots

    @functools.wraps(original)
    def wrapped(self, request, *args, **kwargs):
        result = original(self, request, *args, **kwargs)
        print(
            "probe_allocate_slots",
            "request_id",
            getattr(request, "request_id", "unknown"),
            "result",
            describe_kv_cache_blocks(result),
        )
        return result

    cls.allocate_slots = wrapped


def patch_get_computed_blocks(cls) -> None:
    original = cls.get_computed_blocks

    @functools.wraps(original)
    def wrapped(self, request, *args, **kwargs):
        result = original(self, request, *args, **kwargs)
        blocks = result[0] if isinstance(result, tuple) and result else result
        tokens = result[1] if isinstance(result, tuple) and len(result) > 1 else "unknown"
        print(
            "probe_get_computed_blocks",
            "request_id",
            getattr(request, "request_id", "unknown"),
            "computed_tokens",
            tokens,
            "blocks",
            describe_kv_cache_blocks(blocks),
        )
        return result

    cls.get_computed_blocks = wrapped


def describe_kv_cache_config(config) -> str:
    tensors = getattr(config, "kv_cache_tensors", []) or []
    groups = getattr(config, "kv_cache_groups", []) or []
    return (
        f"num_blocks={getattr(config, 'num_blocks', 'unknown')} "
        f"tensor_count={len(tensors)} group_count={len(groups)} "
        f"groups={describe_groups(groups)}"
    )


def describe_groups(groups) -> str:
    parts = []
    for index, group in enumerate(groups):
        layer_names = getattr(group, "layer_names", []) or []
        spec = getattr(group, "kv_cache_spec", None)
        block_size = getattr(spec, "block_size", "unknown")
        parts.append(
            f"g{index}:layers={len(layer_names)} block_size={block_size} "
            f"spec={type(spec).__name__}"
        )
    return "[" + ";".join(parts) + "]"


def describe_runner_kv_attrs(runner) -> None:
    interesting = []
    for name, value in vars(runner).items():
        if "kv" not in name.lower() and "cache" not in name.lower():
            continue
        interesting.append(f"{name}={describe_value(value)}")
    if interesting:
        print("probe_runner_attrs", "|".join(interesting[:40]))


def describe_value(value) -> str:
    if isinstance(value, dict):
        items = []
        for key, item in list(value.items())[:20]:
            items.append(f"{key}:{describe_value(item)}")
        return "dict{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        items = [describe_value(item) for item in list(value)[:10]]
        return f"{type(value).__name__}[" + ",".join(items) + "]"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None:
        return f"tensor(shape={tuple(shape)},dtype={dtype},device={device})"
    return type(value).__name__


def describe_kv_cache_blocks(blocks) -> str:
    if blocks is None:
        return "None"
    get_block_ids = getattr(blocks, "get_block_ids", None)
    if get_block_ids is None:
        return type(blocks).__name__
    try:
        block_ids = get_block_ids(allow_none=True)
    except TypeError:
        block_ids = get_block_ids()
    return repr(block_ids)


if __name__ == "__main__":
    main()
