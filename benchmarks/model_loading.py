from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

import turbobus
from daemon_support import (
    add_daemon_options,
    collect_daemon_reservation_info,
    daemon_profile_line,
    daemon_profile_summary,
    daemon_reservation_line,
    runtime_options_kwargs,
)


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def profile_to_dict(profile) -> dict:
    return {
        "target_device": profile.target_device,
        "direct_h2d_bw_gbps": profile.direct_h2d_bw_gbps,
        "direct_d2h_bw_gbps": profile.direct_d2h_bw_gbps,
        "relays": [
            {
                "relay_device": relay.relay_device,
                "target_device": relay.target_device,
                "h2d_bw_gbps": relay.h2d_bw_gbps,
                "d2h_bw_gbps": relay.d2h_bw_gbps,
                "p2p_bw_gbps": relay.p2p_bw_gbps,
                "effective_bw_gbps": relay.effective_bw_gbps,
                "effective_d2h_bw_gbps": relay.effective_d2h_bw_gbps,
                "p2p_enabled": relay.p2p_enabled,
            }
            for relay in profile.relays
        ],
    }


def stats_to_dict(stats) -> dict:
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
        "path_stats": path_stats,
    }


def unique_handles(handles: list) -> list:
    seen = set()
    unique = []
    for handle in handles:
        key = id(handle)
        if key in seen:
            continue
        seen.add(key)
        unique.append(handle)
    return unique


def path_key(path: dict) -> tuple[str, int]:
    return (path["kind"], path["relay_device"])


def summarize_paths(samples: list[dict]) -> list[dict]:
    by_path: dict[tuple[str, int], list[dict]] = {}
    for sample in samples:
        for path in sample["path_stats"]:
            by_path.setdefault(path_key(path), []).append(path)

    summaries = []
    for (kind, relay_device), paths in sorted(by_path.items()):
        summaries.append(
            {
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


def summarize_load(samples: list[dict]) -> dict:
    if not samples:
        return {
            "iterations": 0,
            "median_load_ms": 0.0,
            "median_gib_per_second": 0.0,
            "direct_bytes": 0,
            "relay_bytes": 0,
            "direct_chunks": 0,
            "relay_chunks": 0,
            "path_stats": [],
        }
    load_milliseconds = [sample["load_ms"] for sample in samples]
    bandwidths = [sample["load_gib_per_second"] for sample in samples]
    return {
        "iterations": len(samples),
        "median_load_ms": statistics.median(load_milliseconds),
        "median_gib_per_second": statistics.median(bandwidths),
        "direct_bytes": int(statistics.median(sample["direct_bytes"] for sample in samples)),
        "relay_bytes": int(statistics.median(sample["relay_bytes"] for sample in samples)),
        "direct_chunks": int(statistics.median(sample["direct_chunks"] for sample in samples)),
        "relay_chunks": int(statistics.median(sample["relay_chunks"] for sample in samples)),
        "path_stats": summarize_paths(samples),
    }


def fill_packed_weights(cpu_tensor, bucket_bytes: int, bucket_count: int) -> None:
    for index in range(bucket_count):
        offset = index * bucket_bytes
        view = cpu_tensor.narrow(0, offset, bucket_bytes)
        view.copy_(torch.arange(bucket_bytes, dtype=torch.uint8, pin_memory=True))
        if index:
            view.add_(index)


def fill_separate_weight(byte_count: int, index: int):
    tensor = torch.arange(byte_count, dtype=torch.uint8, pin_memory=True)
    if index:
        tensor.add_(index)
    return tensor


def make_loader(args, runtime: turbobus.Runtime) -> turbobus.ModelWeightLoader:
    loader = turbobus.ModelWeightLoader(runtime)
    if args.storage_layout == "packed":
        total_bytes = args.bucket_count * args.bucket_bytes
        cpu_backing = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        gpu_backing = torch.empty_like(cpu_backing, device=f"cuda:{args.target_gpu}")
        fill_packed_weights(cpu_backing, args.bucket_bytes, args.bucket_count)
        loader.add_packed_buckets(
            "weight",
            cpu_backing,
            gpu_backing,
            bucket_bytes=args.bucket_bytes,
            bucket_count=args.bucket_count,
        )
        return loader

    for index in range(args.bucket_count):
        cpu_tensor = fill_separate_weight(args.bucket_bytes, index)
        gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{args.target_gpu}")
        loader.add_bucket(f"weight{index}", cpu_tensor, gpu_tensor, bucket_id=index)
    return loader


def run_warmup(loader: turbobus.ModelWeightLoader, names: list[str], warmup: int) -> None:
    for _ in range(warmup):
        loader.mark_unloaded(names)
        loader.load_buckets(names)
        loader.wait_many(names)


def run_mode(
    runtime: turbobus.Runtime,
    loader: turbobus.ModelWeightLoader,
    names: list[str],
    mode: str,
    warmup: int,
    iterations: int,
    verify: bool,
) -> dict:
    runtime.set_transfer_mode(mode)
    run_warmup(loader, names, warmup)

    samples = []
    mismatches = []
    total_bytes = sum(loader.bucket(name).bytes for name in names)
    for iteration in range(iterations):
        loader.mark_unloaded(names)
        start = time.perf_counter()
        handles = loader.load_buckets(names)
        loader.wait_many(names)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        handle_samples = [
            stats_to_dict(handle.stats)
            for handle in unique_handles(handles)
            if handle.stats is not None
        ]
        sample = {
            "iteration": iteration,
            "load_ms": elapsed_ms,
            "load_gib_per_second": (
                (total_bytes / (1024**3)) / (elapsed_ms / 1000.0)
                if elapsed_ms > 0.0
                else 0.0
            ),
            "bytes": total_bytes,
            "direct_bytes": sum(item["direct_bytes"] for item in handle_samples),
            "relay_bytes": sum(item["relay_bytes"] for item in handle_samples),
            "direct_chunks": sum(item["direct_chunks"] for item in handle_samples),
            "relay_chunks": sum(item["relay_chunks"] for item in handle_samples),
            "path_stats": [
                path for item in handle_samples for path in item["path_stats"]
            ],
            "auto_decision": runtime.last_auto_decision_dict(),
            "daemon_reservation": collect_daemon_reservation_info(handles),
        }
        samples.append(sample)
        print(
            "mode",
            mode,
            "sample_load_ms",
            elapsed_ms,
            "sample_gib_per_second",
            sample["load_gib_per_second"],
            "direct_chunks",
            sample["direct_chunks"],
            "relay_chunks",
            sample["relay_chunks"],
        )

        if verify:
            for name in names:
                bucket = loader.bucket(name)
                cpu_view = bucket.cpu_tensor.narrow(
                    0,
                    bucket.cpu_offset,
                    bucket.bytes,
                )
                gpu_view = bucket.gpu_tensor.narrow(
                    0,
                    bucket.gpu_offset,
                    bucket.bytes,
                )
                if not torch.equal(cpu_view, gpu_view.cpu()):
                    mismatches.append(name)

    summary = summarize_load(samples)
    result = {
        "mode": mode,
        "samples": samples,
        "summary": summary,
        "verify": len(mismatches) == 0 if verify else None,
        "mismatches": mismatches[:8],
        "last_plan": runtime.last_plan_dict(),
        "last_auto_decision": runtime.last_auto_decision_dict(),
        "daemon_reservation": samples[-1]["daemon_reservation"] if samples else {},
    }
    print(
        "mode",
        mode,
        "median_load_ms",
        summary["median_load_ms"],
        "median_gib_per_second",
        summary["median_gib_per_second"],
        "direct_chunks",
        summary["direct_chunks"],
        "relay_chunks",
        summary["relay_chunks"],
        "verify",
        result["verify"],
    )
    return result


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
        "MODEL_LOAD_SUMMARY_BEGIN",
        (
            "model_load_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"bucket_count={config['bucket_count']} bucket_bytes={config['bucket_bytes']} "
            f"storage_layout={config['storage_layout']} chunk_bytes={config['chunk_bytes']} "
            f"iterations={config['iterations']} mode={config['mode']} "
            f"dynamic_weights={config['dynamic_weights']} "
            f"daemon_socket_path={config['daemon_socket_path']} "
            f"daemon_max_inflight_chunks={config['daemon_max_inflight_chunks']} "
            f"daemon_profile_max_age_seconds={config['daemon_profile_max_age_seconds']}"
        ),
        (
            "profile "
            f"direct_h2d_bw_gbps={result['profile']['direct_h2d_bw_gbps']:.3f} "
            f"direct_d2h_bw_gbps={result['profile']['direct_d2h_bw_gbps']:.3f}"
        ),
    ]
    for relay in result["profile"]["relays"]:
        lines.append(
            "profile_relay "
            f"relay={relay['relay_device']} h2d={relay['h2d_bw_gbps']:.3f} "
            f"d2h={relay['d2h_bw_gbps']:.3f} p2p={relay['p2p_bw_gbps']:.3f} "
            f"effective={relay['effective_bw_gbps']:.3f} "
            f"effective_d2h={relay['effective_d2h_bw_gbps']:.3f} "
            f"p2p_enabled={relay['p2p_enabled']}"
        )

    for mode, mode_result in result["modes"].items():
        summary = mode_result["summary"]
        lines.append(
            "model_load_mode "
            f"mode={mode} median_load_ms={summary['median_load_ms']:.3f} "
            f"median_gib_s={summary['median_gib_per_second']:.3f} "
            f"direct_chunks={summary['direct_chunks']} relay_chunks={summary['relay_chunks']} "
            f"direct_bytes={summary['direct_bytes']} relay_bytes={summary['relay_bytes']}"
        )
        for path in summary["path_stats"]:
            lines.append(
                "model_load_path "
                f"mode={mode} kind={path['kind']} relay={path['relay_device']} "
                f"median_gib_s={path['median_gib_per_second']:.3f} "
                f"median_ms={path['median_cuda_ms']:.3f} "
                f"bytes={path['bytes']} chunks={path['chunks']}"
            )
        if mode_result["verify"] is not None:
            lines.append(f"model_load_verify mode={mode} match={mode_result['verify']}")
        daemon_reservation = daemon_reservation_line(
            mode_result.get("daemon_reservation", {})
        )
        if daemon_reservation:
            lines.append(f"{daemon_reservation} mode={mode}")

    for key, value in result["speedups"].items():
        lines.append(f"model_load_speedup {key}={value:.3f}")
    daemon_profile_initial = daemon_profile_line(result.get("daemon_profile_initial", {}))
    if daemon_profile_initial:
        lines.append(f"{daemon_profile_initial} phase=initial")
    daemon_profile = daemon_profile_line(result.get("daemon_profile", {}))
    if daemon_profile:
        lines.append(f"{daemon_profile} phase=after_profile")
    lines.append("MODEL_LOAD_SUMMARY_END")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus model weight loading benchmark")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--min-pool-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["auto", "pool", "direct", "relay", "all"], default="all")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-weight-alpha", type=float, default=0.25)
    parser.add_argument("--json-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    parser.add_argument("--summary-output")
    add_daemon_options(parser)
    args = parser.parse_args()

    if args.bucket_count <= 0:
        raise ValueError("--bucket-count must be positive")
    if args.bucket_bytes <= 0:
        raise ValueError("--bucket-bytes must be positive")

    relays = parse_relay_gpus(args.relay_gpus)
    torch.cuda.set_device(args.target_gpu)
    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        min_pool_bytes=args.min_pool_bytes,
        enable_dynamic_weights=args.dynamic_weights,
        dynamic_weight_alpha=args.dynamic_weight_alpha,
        **runtime_options_kwargs(args),
    )
    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays, options=options)
    daemon_profile_initial = daemon_profile_summary(runtime)
    profile = runtime.profile(args.profile_bytes, force=args.force_profile)
    loader = make_loader(args, runtime)
    names = loader.names()

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "bucket_count": args.bucket_count,
            "bucket_bytes": args.bucket_bytes,
            "storage_layout": args.storage_layout,
            "chunk_bytes": args.chunk_bytes,
            "min_pool_bytes": args.min_pool_bytes,
            "profile_bytes": args.profile_bytes,
            "force_profile": args.force_profile,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mode": args.mode,
            "dynamic_weights": args.dynamic_weights,
            "dynamic_weight_alpha": args.dynamic_weight_alpha,
            "daemon_socket_path": args.daemon_socket_path,
            "daemon_max_inflight_chunks": args.daemon_max_inflight_chunks,
            "daemon_profile_max_age_seconds": args.daemon_profile_max_age_seconds,
        },
        "profile": profile_to_dict(profile),
        "daemon_profile_initial": daemon_profile_initial,
        "daemon_profile": daemon_profile_summary(runtime),
        "modes": {},
        "speedups": {},
    }

    modes = ["direct", "relay", "pool", "auto"] if args.mode == "all" else [args.mode]
    for mode in modes:
        result["modes"][mode] = run_mode(
            runtime,
            loader,
            names,
            mode,
            args.warmup,
            args.iterations,
            args.verify,
        )

    if args.mode == "all":
        direct = result["modes"]["direct"]["summary"]["median_gib_per_second"]
        relay = result["modes"]["relay"]["summary"]["median_gib_per_second"]
        pool = result["modes"]["pool"]["summary"]["median_gib_per_second"]
        auto = result["modes"]["auto"]["summary"]["median_gib_per_second"]
        if direct > 0.0:
            result["speedups"]["pool_over_direct"] = pool / direct
            result["speedups"]["auto_over_direct"] = auto / direct
        if relay > 0.0:
            result["speedups"]["pool_over_relay"] = pool / relay
            result["speedups"]["auto_over_relay"] = auto / relay

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
