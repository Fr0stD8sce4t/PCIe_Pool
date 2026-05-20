from __future__ import annotations

import argparse
import json
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


def path_key(path: dict) -> tuple[str, int, int]:
    return (path["kind"], path["target_device"], path["relay_device"])


def print_path_stats_summary(mode: str, samples: list[dict]) -> None:
    by_path: dict[tuple[str, int, int], list[dict]] = {}
    for sample in samples:
        for path in sample["path_stats"]:
            by_path.setdefault(path_key(path), []).append(path)

    for (kind, target_device, relay_device), paths in sorted(by_path.items()):
        path_bandwidths = [path["gib_per_second"] for path in paths]
        path_milliseconds = [path["cuda_elapsed_ms"] for path in paths]
        bytes_values = [path["bytes"] for path in paths]
        chunks_values = [path["chunks"] for path in paths]
        print(
            "mode",
            mode,
            "path",
            kind,
            "target",
            target_device,
            "relay",
            relay_device,
            "median_path_gib_per_second",
            statistics.median(path_bandwidths),
            "median_path_cuda_milliseconds",
            statistics.median(path_milliseconds),
            "median_path_bytes",
            int(statistics.median(bytes_values)),
            "median_path_chunks",
            int(statistics.median(chunks_values)),
        )


def summarize_plan(plan: dict) -> dict:
    assignments = []
    for assignment in plan["assignments"]:
        chunks = assignment["chunks"]
        assignments.append(
            {
                "path": assignment["path"],
                "bytes": assignment["bytes"],
                "chunk_count": assignment["chunk_count"],
                "first_chunk": chunks[0] if chunks else None,
                "last_chunk": chunks[-1] if chunks else None,
            }
        )
    return {
        "total_bytes": plan["total_bytes"],
        "chunk_bytes": plan["chunk_bytes"],
        "assignments": assignments,
    }


def plan_result(runtime: turbobus.Runtime, include_plan: bool) -> dict:
    plan = runtime.last_plan_dict()
    if include_plan:
        return {"last_plan": plan}
    return {"last_plan_summary": summarize_plan(plan)}


def run_mode(
    runtime: turbobus.Runtime,
    cpu,
    gpu,
    mode: str,
    warmup: int,
    iterations: int,
    include_plan: bool,
):
    runtime.set_transfer_mode(mode)
    for _ in range(warmup):
        handle = runtime.fetch_to_gpu(cpu, gpu)
        handle.wait()

    samples = []
    last_stats = None
    for _ in range(iterations):
        handle = runtime.fetch_to_gpu(cpu, gpu)
        handle.wait()
        stats = handle.stats
        last_stats = stats
        sample = stats_to_dict(stats)
        samples.append(sample)
        print(
            "mode",
            mode,
            "sample_gib_per_second",
            stats.gib_per_second,
            "cuda_milliseconds",
            stats.cuda_elapsed_ms,
            "submit_milliseconds",
            stats.submit_to_complete_ms,
            "direct_chunks",
            stats.direct_chunks,
            "relay_chunks",
            stats.relay_chunks,
        )

    sample_bandwidths = [sample["gib_per_second"] for sample in samples]
    median = statistics.median(sample_bandwidths) if sample_bandwidths else 0.0
    print("mode", mode, "median_gib_per_second", median)
    print_path_stats_summary(mode, samples)
    result = {
        "mode": mode,
        "median_gib_per_second": median,
        "samples": samples,
        "last_stats": stats_to_dict(last_stats) if last_stats is not None else None,
    }
    result.update(plan_result(runtime, include_plan))
    return result


def write_json(path: str, result: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus bandwidth pool smoke benchmark")
    parser.add_argument("--target-gpu", type=int, default=0)
    parser.add_argument("--relay-gpus", default="1")
    parser.add_argument("--bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["pool", "direct", "relay", "all"], default="pool")
    parser.add_argument("--json-output")
    parser.add_argument(
        "--include-plan",
        action="store_true",
        help="include full per-chunk last_plan in JSON instead of a compact summary",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-weight-alpha", type=float, default=0.25)
    args = parser.parse_args()

    relays = parse_relay_gpus(args.relay_gpus)
    torch.cuda.set_device(args.target_gpu)

    cpu = torch.arange(args.bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(args.bytes, dtype=torch.uint8, device=f"cuda:{args.target_gpu}")

    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        enable_dynamic_weights=args.dynamic_weights,
        dynamic_weight_alpha=args.dynamic_weight_alpha,
    )
    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays, options=options)
    profile = runtime.profile(args.profile_bytes)
    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "bytes": args.bytes,
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
        "verify": None,
    }

    print("target_gpu", args.target_gpu)
    print("relay_gpus", ",".join(str(relay) for relay in relays))
    print("bytes", args.bytes)
    print("chunk_bytes", args.chunk_bytes)
    print("warmup", args.warmup)
    print("iterations", args.iterations)
    print("mode", args.mode)
    print("dynamic_weights", args.dynamic_weights)
    print("dynamic_weight_alpha", args.dynamic_weight_alpha)
    print("direct_h2d_bw_gbps", profile.direct_h2d_bw_gbps)
    for relay in profile.relays:
        print(
            "relay",
            relay.relay_device,
            "h2d",
            relay.h2d_bw_gbps,
            "p2p",
            relay.p2p_bw_gbps,
            "effective",
            relay.effective_bw_gbps,
        )

    modes = ["direct", "relay", "pool"] if args.mode == "all" else [args.mode]
    medians = {}
    for mode in modes:
        mode_result = run_mode(
            runtime,
            cpu,
            gpu,
            mode,
            args.warmup,
            args.iterations,
            args.include_plan,
        )
        median = mode_result["median_gib_per_second"]
        medians[mode] = median
        result["modes"][mode] = mode_result

    if args.mode == "all":
        direct = medians.get("direct", 0.0)
        relay = medians.get("relay", 0.0)
        pool = medians.get("pool", 0.0)
        if direct > 0.0:
            result["speedups"]["pool_over_direct_median"] = pool / direct
            print("pool_over_direct_median", result["speedups"]["pool_over_direct_median"])
        if relay > 0.0:
            result["speedups"]["pool_over_relay_median"] = pool / relay
            print("pool_over_relay_median", result["speedups"]["pool_over_relay_median"])

    if args.verify:
        result["verify"] = bool(torch.equal(cpu, gpu.cpu()))
        print("match", result["verify"])

    if args.json_output:
        write_json(args.json_output, result)
        print("json_output", args.json_output)


if __name__ == "__main__":
    main()
