from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = None
turbobus = None
block_bytes_from_vllm_kv_tensor = None
make_vllm_layer_range_refs_from_ids = None
VllmTurboBusIntegration = None


def load_runtime_modules() -> None:
    global torch
    global turbobus
    global block_bytes_from_vllm_kv_tensor
    global make_vllm_layer_range_refs_from_ids
    global VllmTurboBusIntegration

    if torch is not None:
        return

    import torch as torch_module
    import turbobus as turbobus_module
    from turbobus.vllm import (
        block_bytes_from_vllm_kv_tensor as block_bytes_fn,
        make_vllm_layer_range_refs_from_ids as make_refs_fn,
    )
    from turbobus.vllm_integration import VllmTurboBusIntegration as integration_cls

    torch = torch_module
    turbobus = turbobus_module
    block_bytes_from_vllm_kv_tensor = block_bytes_fn
    make_vllm_layer_range_refs_from_ids = make_refs_fn
    VllmTurboBusIntegration = integration_cls


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percent / 100.0) * len(ordered)) - 1)
    return ordered[index]


def stats_summary(handles: list) -> dict:
    unique = []
    seen = set()
    for handle in handles:
        if id(handle) in seen or handle.stats is None:
            continue
        seen.add(id(handle))
        unique.append(handle.stats)
    bandwidths = [stats.gib_per_second for stats in unique if stats.gib_per_second > 0.0]
    return {
        "bytes": sum(stats.bytes for stats in unique),
        "gib_per_second": statistics.median(bandwidths) if bandwidths else 0.0,
        "direct_chunks": sum(stats.direct_chunks for stats in unique),
        "relay_chunks": sum(stats.relay_chunks for stats in unique),
        "direct_bytes": sum(stats.direct_bytes for stats in unique),
        "relay_bytes": sum(stats.relay_bytes for stats in unique),
    }


def cpu_block_view(cpu_backing, cpu_slot: int, block_bytes: int):
    return cpu_backing.narrow(0, cpu_slot * block_bytes, block_bytes)


def gpu_block_view(kv_cache, ref, block_bytes: int):
    byte_view = kv_cache.view(torch.uint8).reshape(-1)
    offset = ref.gpu_offset if ref.gpu_offset is not None else ref.gpu_slot * block_bytes
    return byte_view.narrow(0, offset, block_bytes)


def capture_cpu_snapshots(cpu_backings, refs_by_layer: dict[int, list], block_bytes_by_layer):
    snapshots = {}
    for layer_id, refs in refs_by_layer.items():
        block_bytes = block_bytes_by_layer[layer_id]
        for ref in refs:
            snapshots[(layer_id, ref.cpu_slot)] = cpu_block_view(
                cpu_backings[layer_id],
                ref.cpu_slot,
                block_bytes,
            ).clone()
    return snapshots


def verify_cpu_snapshots(cpu_backings, refs_by_layer: dict[int, list], block_bytes_by_layer, snapshots):
    for layer_id, refs in refs_by_layer.items():
        block_bytes = block_bytes_by_layer[layer_id]
        for ref in refs:
            current = cpu_block_view(cpu_backings[layer_id], ref.cpu_slot, block_bytes)
            expected = snapshots[(layer_id, ref.cpu_slot)]
            if not torch.equal(current, expected):
                return False
    return True


def zero_gpu_blocks(kv_caches, refs_by_layer: dict[int, list], block_bytes_by_layer) -> None:
    for layer_id, refs in refs_by_layer.items():
        block_bytes = block_bytes_by_layer[layer_id]
        for ref in refs:
            gpu_block_view(kv_caches[layer_id], ref, block_bytes).zero_()
    torch.cuda.synchronize()


def refs_by_layer(refs) -> dict[int, list]:
    by_layer = {}
    for ref in refs:
        by_layer.setdefault(ref.group_id, []).append(ref)
    return by_layer


def wait_for_allocation(integration, timeout_s: float) -> str:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if integration.state.allocations:
            return next(iter(integration.state.allocations))
        time.sleep(0.01)
    raise RuntimeError("vLLM did not report an allocate_slots() event in this process")


def make_runtime(target_gpu: int, relay_gpus: list[int], args, mode: str) -> turbobus.Runtime:
    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        profile_bytes=args.profile_bytes,
        transfer_mode=mode,
        enable_dynamic_weights=args.dynamic_weights,
    )
    return turbobus.Runtime(target_gpu=target_gpu, relay_gpus=relay_gpus, options=options)


def initialize_vllm(args, first_mode: str):
    load_runtime_modules()
    torch.cuda.set_device(args.runtime_target_gpu)
    relay_gpus = args.runtime_relay_gpus
    if args.disable_multiproc_executor:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    runtime = make_runtime(args.runtime_target_gpu, relay_gpus, args, first_mode)
    integration = VllmTurboBusIntegration(runtime)
    integration.install()

    from vllm import LLM, SamplingParams

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    init_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    generate_ms = (time.perf_counter() - start) * 1000.0

    request_id = wait_for_allocation(integration, args.allocation_timeout_s)
    if not integration.state.kv_caches:
        raise RuntimeError(
            "TurboBus did not observe vLLM kv_caches. "
            "Run with --disable-multiproc-executor or a vLLM build that keeps the engine in-process."
        )
    validate_kv_cache_devices(args, integration.state.kv_caches)

    block_ids = integration.block_ids_for_request(request_id)
    selected_block_ids = block_ids[: args.restore_blocks]
    if not selected_block_ids:
        raise RuntimeError("vLLM allocation did not contain any block ids")

    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    return {
        "llm": llm,
        "request_id": request_id,
        "allocation": integration.state.allocations[request_id],
        "kv_cache_config": integration.state.kv_cache_config,
        "kv_caches": integration.state.kv_caches,
        "init_ms": init_ms,
        "generate_ms": generate_ms,
        "generated_text": text[:200],
    }


def run_mode(args, captured: dict, mode: str) -> dict:
    runtime = make_runtime(args.runtime_target_gpu, args.runtime_relay_gpus, args, mode)
    integration = VllmTurboBusIntegration(runtime)
    integration.bind_kv_caches(captured["kv_caches"], captured["kv_cache_config"])
    integration.state.allocations[captured["request_id"]] = captured["allocation"]

    request_id = captured["request_id"]
    block_ids = integration.block_ids_for_request(request_id)
    selected_block_ids = block_ids[: args.restore_blocks]
    if not selected_block_ids:
        raise RuntimeError("vLLM allocation did not contain any block ids")

    lanes_per_layer = [
        int(kv_cache.shape[0]) if len(kv_cache.shape) >= 3 else 1
        for kv_cache in captured["kv_caches"]
    ]
    slots_per_layer = max(
        (len(selected_block_ids) * lane_count for lane_count in lanes_per_layer),
        default=1,
    )
    cpu_backings = integration.allocate_cpu_backings(slots_per_layer)
    refs = make_vllm_layer_range_refs_from_ids(
        request_id,
        selected_block_ids,
        integration.state.kv_caches,
    )
    grouped_refs = refs_by_layer(refs)
    block_bytes_by_layer = [
        block_bytes_from_vllm_kv_tensor(kv_cache) for kv_cache in integration.state.kv_caches
    ]

    adapter = integration._require_adapter()
    save_samples = []
    restore_samples = []
    ok = True
    for _ in range(args.iterations):
        start = time.perf_counter()
        save_handles = adapter.save_prefix(refs)
        save_ms = (time.perf_counter() - start) * 1000.0

        snapshots = capture_cpu_snapshots(cpu_backings, grouped_refs, block_bytes_by_layer)
        zero_gpu_blocks(integration.state.kv_caches, grouped_refs, block_bytes_by_layer)

        start = time.perf_counter()
        restore_handles = adapter.restore_prefix(refs)
        restore_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        adapter.save_prefix(refs)
        verify_save_ms = (time.perf_counter() - start) * 1000.0
        verified = verify_cpu_snapshots(
            cpu_backings,
            grouped_refs,
            block_bytes_by_layer,
            snapshots,
        )
        ok = ok and verified

        save_samples.append(
            {
                "ms": save_ms,
                "verify_save_ms": verify_save_ms,
                **stats_summary(save_handles),
            }
        )
        restore_samples.append({"ms": restore_ms, **stats_summary(restore_handles)})

    restore_ms_values = [sample["ms"] for sample in restore_samples]
    save_ms_values = [sample["ms"] for sample in save_samples]
    restore_stats = stats_summary([])
    if restore_samples:
        restore_stats = {
            "gib_per_second": statistics.median(
                sample["gib_per_second"] for sample in restore_samples
            ),
            "direct_chunks": sum(sample["direct_chunks"] for sample in restore_samples),
            "relay_chunks": sum(sample["relay_chunks"] for sample in restore_samples),
            "direct_bytes": sum(sample["direct_bytes"] for sample in restore_samples),
            "relay_bytes": sum(sample["relay_bytes"] for sample in restore_samples),
        }

    return {
        "mode": mode,
        "request_id": request_id,
        "init_ms": captured["init_ms"],
        "generate_ms": captured["generate_ms"],
        "layer_count": len(integration.state.kv_caches),
        "block_bytes": block_bytes_by_layer[0] if block_bytes_by_layer else 0,
        "block_ids": list(selected_block_ids),
        "restore_blocks": len(selected_block_ids),
        "save_p50_ms": statistics.median(save_ms_values),
        "save_p95_ms": percentile(save_ms_values, 95.0),
        "restore_p50_ms": statistics.median(restore_ms_values),
        "restore_p95_ms": percentile(restore_ms_values, 95.0),
        "restore_gib_s": restore_stats["gib_per_second"],
        "direct_chunks": restore_stats["direct_chunks"],
        "relay_chunks": restore_stats["relay_chunks"],
        "direct_bytes": restore_stats["direct_bytes"],
        "relay_bytes": restore_stats["relay_bytes"],
        "verified": ok,
        "generated_text": captured["generated_text"],
    }


def print_summary(args, results: list[dict]) -> None:
    print("COPY_SUMMARY_BEGIN")
    print(
        "vllm_restore_config",
        f"target={args.target_gpu}",
        f"relays={parse_relay_gpus(args.relay_gpus)}",
        f"runtime_target={args.runtime_target_gpu}",
        f"runtime_relays={args.runtime_relay_gpus}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"model={args.model}",
        f"restore_blocks={args.restore_blocks}",
        f"iterations={args.iterations}",
        f"chunk_bytes={args.chunk_bytes}",
        f"mode={args.mode}",
        f"dynamic_weights={args.dynamic_weights}",
    )
    print(
        "vllm_restore_scenario",
        "type=real_vllm_kv_slot_save_restore",
        "boundary=GPUModelRunner.kv_caches",
        "block_ids=KVCacheManager.allocate_slots",
        "operation=save_zero_restore_verify",
        "note=real_vllm_tensor_bytes_not_simulator",
    )
    for result in results:
        print(
            "vllm_restore_mode",
            f"mode={result['mode']}",
            f"layers={result['layer_count']}",
            f"block_bytes={result['block_bytes']}",
            f"restore_blocks={result['restore_blocks']}",
            f"save_p50_ms={result['save_p50_ms']:.3f}",
            f"restore_p50_ms={result['restore_p50_ms']:.3f}",
            f"restore_p95_ms={result['restore_p95_ms']:.3f}",
            f"restore_gib_s={result['restore_gib_s']:.3f}",
            f"direct_chunks={result['direct_chunks']}",
            f"relay_chunks={result['relay_chunks']}",
            f"verified={result['verified']}",
        )
    by_mode = {result["mode"]: result for result in results}
    if "pool" in by_mode and "direct" in by_mode:
        direct = by_mode["direct"]["restore_p50_ms"]
        pool = by_mode["pool"]["restore_p50_ms"]
        if pool > 0:
            print("vllm_restore_speedup", f"direct_over_pool_restore_p50={direct / pool:.3f}")
    if "pool" in by_mode and "relay" in by_mode:
        relay = by_mode["relay"]["restore_p50_ms"]
        pool = by_mode["pool"]["restore_p50_ms"]
        if pool > 0:
            print("vllm_restore_speedup", f"relay_over_pool_restore_p50={relay / pool:.3f}")
    print("COPY_SUMMARY_END")


def default_log_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("benchmarks") / "results" / f"vllm_turbobus_restore_{stamp}.log")


def configure_cuda_devices(args) -> None:
    physical_relays = parse_relay_gpus(args.relay_gpus)
    visible = [args.target_gpu, *physical_relays]
    if args.map_physical_gpus:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            print(
                "warning existing CUDA_VISIBLE_DEVICES preserved",
                os.environ["CUDA_VISIBLE_DEVICES"],
            )
            args.runtime_target_gpu = args.target_gpu
            args.runtime_relay_gpus = physical_relays
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible)
        args.runtime_target_gpu = 0
        args.runtime_relay_gpus = list(range(1, len(visible)))
        return
    args.runtime_target_gpu = args.target_gpu
    args.runtime_relay_gpus = physical_relays


def validate_kv_cache_devices(args, kv_caches) -> None:
    bad = []
    for index, kv_cache in enumerate(kv_caches):
        device = getattr(kv_cache, "device", None)
        if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) != args.runtime_target_gpu:
            bad.append((index, str(device)))
    if bad:
        raise RuntimeError(
            "vLLM KV cache tensors are not on the TurboBus runtime target GPU: "
            f"runtime_target={args.runtime_target_gpu} mismatches={bad[:8]}. "
            "Use the default physical GPU mapping or set CUDA_VISIBLE_DEVICES so "
            "vLLM cuda:0 maps to the requested target GPU."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Save/restore real vLLM KV slots with TurboBus")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-multiproc-executor",
        dest="disable_multiproc_executor",
        action="store_false",
        help="Allow vLLM to use a separate engine process. The TurboBus hook needs in-process vLLM.",
    )
    parser.set_defaults(disable_multiproc_executor=True)
    parser.add_argument("--restore-blocks", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mode", choices=["direct", "relay", "pool", "all"], default="all")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--allocation-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--log-output",
        default=None,
        help="Write vLLM logs, TurboBus summary, and errors to this file.",
    )
    parser.add_argument(
        "--no-map-physical-gpus",
        dest="map_physical_gpus",
        action="store_false",
        help="Treat --target-gpu/--relay-gpus as already-visible logical CUDA ids.",
    )
    parser.set_defaults(map_physical_gpus=True)
    args = parser.parse_args()
    if args.log_output is None:
        args.log_output = default_log_path()
    return args


def run(args) -> None:
    configure_cuda_devices(args)
    print(
        "vllm_restore_start",
        f"target={args.target_gpu}",
        f"relays={parse_relay_gpus(args.relay_gpus)}",
        f"runtime_target={args.runtime_target_gpu}",
        f"runtime_relays={args.runtime_relay_gpus}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    )
    modes = ["direct", "relay", "pool"] if args.mode == "all" else [args.mode]
    captured = initialize_vllm(args, modes[0])
    print(
        "vllm_restore_capture",
        f"request_id={captured['request_id']}",
        f"layers={len(captured['kv_caches'])}",
        f"kv_device={captured['kv_caches'][0].device if captured['kv_caches'] else 'none'}",
        f"generated_text={captured['generated_text']!r}",
    )
    results = []
    for mode in modes:
        result = run_mode(args, captured, mode)
        results.append(result)
        print(
            "mode",
            mode,
            "restore_p50_ms",
            result["restore_p50_ms"],
            "restore_gib_s",
            result["restore_gib_s"],
            "direct_chunks",
            result["direct_chunks"],
            "relay_chunks",
            result["relay_chunks"],
            "verified",
            result["verified"],
        )
    print_summary(args, results)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                run(args)
            except Exception:
                print("VLLM_RESTORE_ERROR_BEGIN")
                traceback.print_exc()
                print("VLLM_RESTORE_ERROR_END")
            else:
                ok = True
    print("vllm_turbobus_restore log", log_path)
    if ok:
        print("vllm_turbobus_restore status ok")
    else:
        print("vllm_turbobus_restore status failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
