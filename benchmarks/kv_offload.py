from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import torch

import turbobus


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def profile_to_dict(profile) -> dict:
    return {
        "target_device": profile.target_device,
        "direct_h2d_bw_gbps": profile.direct_h2d_bw_gbps,
        "relays": [
            {
                "relay_device": relay.relay_device,
                "target_device": relay.target_device,
                "h2d_bw_gbps": relay.h2d_bw_gbps,
                "p2p_bw_gbps": relay.p2p_bw_gbps,
                "effective_bw_gbps": relay.effective_bw_gbps,
                "p2p_enabled": relay.p2p_enabled,
            }
            for relay in profile.relays
        ],
    }


def stats_to_dict(stats) -> dict:
    per_relay = []
    for relay, bytes_, chunks in zip(
        stats.relay_devices,
        stats.relay_device_bytes,
        stats.relay_device_chunks,
        strict=False,
    ):
        per_relay.append(
            {
                "relay_device": relay,
                "bytes": bytes_,
                "chunks": chunks,
            }
        )
    path_stats = [
        {
            "kind": path.kind,
            "direction": path.direction,
            "target_device": path.target_device,
            "relay_device": path.relay_device,
            "bytes": path.bytes,
            "chunks": path.chunks,
            "cuda_elapsed_ms": path.cuda_elapsed_ms,
            "gib_per_second": path.gib_per_second,
        }
        for path in stats.path_stats
    ]
    return {
        "bytes": stats.bytes,
        "direct_bytes": stats.direct_bytes,
        "relay_bytes": stats.relay_bytes,
        "cuda_elapsed_ms": stats.cuda_elapsed_ms,
        "submit_to_complete_ms": stats.submit_to_complete_ms,
        "gib_per_second": stats.gib_per_second,
        "submit_gib_per_second": stats.submit_gib_per_second,
        "direct_chunks": stats.direct_chunks,
        "relay_chunks": stats.relay_chunks,
        "per_relay": per_relay,
        "path_stats": path_stats,
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percent / 100.0) * len(ordered)) - 1)
    return ordered[index]


def summarize_operation(samples: list[dict]) -> dict:
    if not samples:
        return {
            "count": 0,
            "bytes": 0,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "effective_gib_per_second": 0.0,
            "direct_bytes": 0,
            "relay_bytes": 0,
            "direct_chunks": 0,
            "relay_chunks": 0,
            "path_stats": [],
        }

    latencies = [sample["submit_to_complete_ms"] for sample in samples]
    total_bytes = sum(sample["bytes"] for sample in samples)
    total_seconds = sum(latencies) / 1000.0
    return {
        "count": len(samples),
        "bytes": total_bytes,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": percentile(latencies, 95.0),
        "effective_gib_per_second": (
            (total_bytes / (1024**3)) / total_seconds if total_seconds > 0.0 else 0.0
        ),
        "direct_bytes": sum(sample["direct_bytes"] for sample in samples),
        "relay_bytes": sum(sample["relay_bytes"] for sample in samples),
        "direct_chunks": sum(sample["direct_chunks"] for sample in samples),
        "relay_chunks": sum(sample["relay_chunks"] for sample in samples),
        "path_stats": summarize_paths(samples),
    }


def path_key(path: dict) -> tuple[str, str, int]:
    return (path["direction"], path["kind"], path["relay_device"])


def summarize_paths(samples: list[dict]) -> list[dict]:
    by_path: dict[tuple[str, str, int], list[dict]] = {}
    for sample in samples:
        for path in sample["path_stats"]:
            by_path.setdefault(path_key(path), []).append(path)

    summaries = []
    for (direction, kind, relay_device), paths in sorted(by_path.items()):
        summaries.append(
            {
                "direction": direction,
                "kind": kind,
                "relay_device": relay_device,
                "median_gib_per_second": statistics.median(
                    path["gib_per_second"] for path in paths
                ),
                "median_cuda_ms": statistics.median(
                    path["cuda_elapsed_ms"] for path in paths
                ),
                "bytes": sum(path["bytes"] for path in paths),
                "chunks": sum(path["chunks"] for path in paths),
            }
        )
    return summaries


def make_block(block_bytes: int, block_index: int):
    tensor = torch.arange(block_bytes, dtype=torch.uint8, pin_memory=True)
    if block_index:
        tensor.add_(block_index)
    return tensor


def active_indices(iteration: int, active_blocks: int, num_blocks: int) -> list[int]:
    start = (iteration * active_blocks) % num_blocks
    return [(start + offset) % num_blocks for offset in range(active_blocks)]


def run_warmup(store: turbobus.OffloadStore, names: list[str], warmup: int) -> None:
    for _ in range(warmup):
        store.prefetch_many(names)
        store.wait_many(names)
        store.evict_many(names)
        store.wait_many(names)


def run_mode(
    runtime: turbobus.Runtime,
    store: turbobus.OffloadStore,
    names: list[str],
    references: list | None,
    mode: str,
    active_blocks: int,
    warmup: int,
    iterations: int,
    verify: bool,
) -> dict:
    runtime.set_transfer_mode(mode)
    run_warmup(store, names[:active_blocks], warmup)

    samples = {"prefetch": [], "evict": []}
    mismatches = []
    for iteration in range(iterations):
        indices = active_indices(iteration, active_blocks, len(names))
        active_names = [names[index] for index in indices]

        for name, handle in zip(
            active_names,
            store.prefetch_many(active_names),
            strict=False,
        ):
            store.wait(name)
            samples["prefetch"].append(
                {
                    "block": name,
                    "iteration": iteration,
                    **stats_to_dict(handle.stats),
                }
            )

        if verify:
            for name in active_names:
                store.block(name).cpu_tensor.zero_()

        for name, handle in zip(
            active_names,
            store.evict_many(active_names),
            strict=False,
        ):
            store.wait(name)
            samples["evict"].append(
                {
                    "block": name,
                    "iteration": iteration,
                    **stats_to_dict(handle.stats),
                }
            )

        if verify and references is not None:
            for index in indices:
                block = store.block(names[index])
                if not torch.equal(block.cpu_tensor, references[index]):
                    mismatches.append(names[index])

    prefetch = summarize_operation(samples["prefetch"])
    evict = summarize_operation(samples["evict"])
    result = {
        "mode": mode,
        "active_blocks": active_blocks,
        "iterations": iterations,
        "prefetch": prefetch,
        "evict": evict,
        "samples": samples,
        "verify": len(mismatches) == 0 if verify else None,
        "mismatches": mismatches[:8],
    }
    print_mode_summary(result)
    return result


def print_mode_summary(result: dict) -> None:
    print(
        "mode",
        result["mode"],
        "prefetch_gib_per_second",
        result["prefetch"]["effective_gib_per_second"],
        "prefetch_p50_ms",
        result["prefetch"]["latency_ms_p50"],
        "prefetch_p95_ms",
        result["prefetch"]["latency_ms_p95"],
        "evict_gib_per_second",
        result["evict"]["effective_gib_per_second"],
        "evict_p50_ms",
        result["evict"]["latency_ms_p50"],
        "evict_p95_ms",
        result["evict"]["latency_ms_p95"],
        "verify",
        result["verify"],
    )


def write_json(path: str, result: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def write_text(path: str, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def compact_summary(result: dict) -> str:
    config = result["config"]
    lines = [
        "COPY_SUMMARY_BEGIN",
        (
            "kv_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"num_blocks={config['num_blocks']} active_blocks={config['active_blocks']} "
            f"block_bytes={config['block_bytes']} chunk_bytes={config['chunk_bytes']} "
            f"iterations={config['iterations']} mode={config['mode']} "
            f"dynamic_weights={config['dynamic_weights']}"
        ),
        f"profile direct_h2d_bw_gbps={result['profile']['direct_h2d_bw_gbps']:.3f}",
    ]
    for relay in result["profile"]["relays"]:
        lines.append(
            "profile_relay "
            f"relay={relay['relay_device']} h2d={relay['h2d_bw_gbps']:.3f} "
            f"p2p={relay['p2p_bw_gbps']:.3f} effective={relay['effective_bw_gbps']:.3f} "
            f"p2p_enabled={relay['p2p_enabled']}"
        )

    for mode, mode_result in result["modes"].items():
        for op in ("prefetch", "evict"):
            summary = mode_result[op]
            lines.append(
                "kv_op "
                f"mode={mode} op={op} count={summary['count']} "
                f"gib_s={summary['effective_gib_per_second']:.3f} "
                f"p50_ms={summary['latency_ms_p50']:.3f} "
                f"p95_ms={summary['latency_ms_p95']:.3f} "
                f"direct_chunks={summary['direct_chunks']} relay_chunks={summary['relay_chunks']} "
                f"direct_bytes={summary['direct_bytes']} relay_bytes={summary['relay_bytes']}"
            )
            for path in summary["path_stats"]:
                lines.append(
                    "kv_path "
                    f"mode={mode} op={op} direction={path['direction']} "
                    f"kind={path['kind']} relay={path['relay_device']} "
                    f"median_gib_s={path['median_gib_per_second']:.3f} "
                    f"median_ms={path['median_cuda_ms']:.3f} "
                    f"bytes={path['bytes']} chunks={path['chunks']}"
                )
        if mode_result["verify"] is not None:
            lines.append(f"kv_verify mode={mode} match={mode_result['verify']}")

    for key, value in result["speedups"].items():
        lines.append(f"kv_speedup {key}={value:.3f}")
    lines.append("COPY_SUMMARY_END")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus KV block offload benchmark")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--active-blocks", type=int)
    parser.add_argument("--block-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["pool", "direct", "relay", "all"], default="pool")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-weight-alpha", type=float, default=0.25)
    parser.add_argument("--json-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    relays = parse_relay_gpus(args.relay_gpus)
    active_blocks = args.active_blocks or args.num_blocks
    if active_blocks <= 0 or active_blocks > args.num_blocks:
        raise ValueError("--active-blocks must be between 1 and --num-blocks")

    torch.cuda.set_device(args.target_gpu)
    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        enable_dynamic_weights=args.dynamic_weights,
        dynamic_weight_alpha=args.dynamic_weight_alpha,
    )
    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays, options=options)
    profile = runtime.profile(args.profile_bytes, force=True)
    store = turbobus.OffloadStore(runtime)

    names = []
    references = [] if args.verify else None
    for index in range(args.num_blocks):
        name = f"kv{index}"
        cpu_tensor = make_block(args.block_bytes, index)
        gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{args.target_gpu}")
        store.add(name, cpu_tensor, gpu_tensor)
        names.append(name)
        if references is not None:
            references.append(cpu_tensor.clone())

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "num_blocks": args.num_blocks,
            "active_blocks": active_blocks,
            "block_bytes": args.block_bytes,
            "chunk_bytes": args.chunk_bytes,
            "profile_bytes": args.profile_bytes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mode": args.mode,
            "dynamic_weights": args.dynamic_weights,
            "dynamic_weight_alpha": args.dynamic_weight_alpha,
        },
        "profile": profile_to_dict(profile),
        "modes": {},
        "speedups": {},
    }

    modes = ["direct", "relay", "pool"] if args.mode == "all" else [args.mode]
    for mode in modes:
        result["modes"][mode] = run_mode(
            runtime,
            store,
            names,
            references,
            mode,
            active_blocks,
            args.warmup,
            args.iterations,
            args.verify,
        )

    if args.mode == "all":
        for op in ("prefetch", "evict"):
            direct = result["modes"]["direct"][op]["effective_gib_per_second"]
            relay = result["modes"]["relay"][op]["effective_gib_per_second"]
            pool = result["modes"]["pool"][op]["effective_gib_per_second"]
            if direct > 0.0:
                result["speedups"][f"pool_over_direct_{op}"] = pool / direct
            if relay > 0.0:
                result["speedups"][f"pool_over_relay_{op}"] = pool / relay

    if args.json_output:
        write_json(args.json_output, result)
        print("json_output", args.json_output)
    summary = compact_summary(result)
    if args.summary_output:
        write_text(args.summary_output, summary)
        print("summary_output", args.summary_output)
    if not args.no_copy_summary:
        print(summary)


if __name__ == "__main__":
    main()
