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


def summarize_handle_stats(handles: list) -> dict:
    samples = [
        stats_to_dict(handle.stats)
        for handle in unique_handles(handles)
        if handle.stats is not None
    ]
    return {
        "bytes": sum(sample["bytes"] for sample in samples),
        "direct_bytes": sum(sample["direct_bytes"] for sample in samples),
        "relay_bytes": sum(sample["relay_bytes"] for sample in samples),
        "direct_chunks": sum(sample["direct_chunks"] for sample in samples),
        "relay_chunks": sum(sample["relay_chunks"] for sample in samples),
        "transfer_ms": sum(sample["submit_to_complete_ms"] for sample in samples),
        "path_stats": [path for sample in samples for path in sample["path_stats"]],
    }


def path_key(path: dict) -> tuple[str, str, int]:
    return (path["direction"], path["kind"], path["relay_device"])


def summarize_paths(samples: list[dict], op: str) -> list[dict]:
    by_path: dict[tuple[str, str, int], list[dict]] = {}
    for sample in samples:
        for path in sample[op]["path_stats"]:
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


def summarize_mode(samples: list[dict]) -> dict:
    if not samples:
        return {
            "iterations": 0,
            "median_iteration_ms": 0.0,
            "median_transfer_ms": 0.0,
            "median_compute_ms": 0.0,
            "median_gib_per_second": 0.0,
            "prefetch": {},
            "offload": {},
        }
    total_bytes = [sample["prefetch"]["bytes"] + sample["offload"]["bytes"] for sample in samples]
    iteration_ms = [sample["iteration_ms"] for sample in samples]
    transfer_ms = [sample["transfer_ms"] for sample in samples]
    return {
        "iterations": len(samples),
        "median_iteration_ms": statistics.median(iteration_ms),
        "median_transfer_ms": statistics.median(transfer_ms),
        "median_compute_ms": statistics.median(sample["compute_ms"] for sample in samples),
        "median_gib_per_second": statistics.median(
            (bytes_ / (1024**3)) / (ms / 1000.0) if ms > 0.0 else 0.0
            for bytes_, ms in zip(total_bytes, transfer_ms, strict=False)
        ),
        "prefetch": summarize_transfer_side(samples, "prefetch"),
        "offload": summarize_transfer_side(samples, "offload"),
    }


def summarize_transfer_side(samples: list[dict], op: str) -> dict:
    return {
        "median_transfer_ms": statistics.median(sample[op]["transfer_ms"] for sample in samples),
        "direct_bytes": int(statistics.median(sample[op]["direct_bytes"] for sample in samples)),
        "relay_bytes": int(statistics.median(sample[op]["relay_bytes"] for sample in samples)),
        "direct_chunks": int(statistics.median(sample[op]["direct_chunks"] for sample in samples)),
        "relay_chunks": int(statistics.median(sample[op]["relay_chunks"] for sample in samples)),
        "path_stats": summarize_paths(samples, op),
    }


def fill_packed_buckets(cpu_tensor, bucket_bytes: int, bucket_count: int) -> None:
    for index in range(bucket_count):
        offset = index * bucket_bytes
        view = cpu_tensor.narrow(0, offset, bucket_bytes)
        view.copy_(torch.arange(bucket_bytes, dtype=torch.uint8, pin_memory=True))
        if index:
            view.add_(index)


def fill_separate_bucket(byte_count: int, index: int):
    tensor = torch.arange(byte_count, dtype=torch.uint8, pin_memory=True)
    if index:
        tensor.add_(index)
    return tensor


def make_manager(args, runtime: turbobus.Runtime) -> turbobus.TrainingOffloadManager:
    manager = turbobus.TrainingOffloadManager(runtime)
    if args.storage_layout == "packed":
        total_bytes = args.bucket_count * args.bucket_bytes
        cpu_backing = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        gpu_backing = torch.empty_like(cpu_backing, device=f"cuda:{args.target_gpu}")
        fill_packed_buckets(cpu_backing, args.bucket_bytes, args.bucket_count)
        manager.add_packed_buckets(
            "bucket",
            cpu_backing,
            gpu_backing,
            bucket_bytes=args.bucket_bytes,
            bucket_count=args.bucket_count,
        )
        return manager

    for index in range(args.bucket_count):
        cpu_tensor = fill_separate_bucket(args.bucket_bytes, index)
        gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{args.target_gpu}")
        manager.add_bucket(f"bucket{index}", cpu_tensor, gpu_tensor, bucket_id=index)
    return manager


def active_names(names: list[str], iteration: int, active_buckets: int) -> list[str]:
    start = (iteration * active_buckets) % len(names)
    return [names[(start + offset) % len(names)] for offset in range(active_buckets)]


def run_compute_proxy(runtime: turbobus.Runtime, tensor, iterations: int) -> float:
    if iterations <= 0:
        return 0.0
    start = time.perf_counter()
    runtime.run_dummy_compute(tensor, iterations)
    return (time.perf_counter() - start) * 1000.0


def run_warmup(
    runtime: turbobus.Runtime,
    manager: turbobus.TrainingOffloadManager,
    names: list[str],
    compute_tensor,
    compute_iterations: int,
    warmup: int,
) -> None:
    for iteration in range(warmup):
        current = active_names(names, iteration, len(names))
        manager.mark_on_cpu(current)
        manager.prefetch_buckets(current)
        manager.wait_many(current)
        run_compute_proxy(runtime, compute_tensor, compute_iterations)
        manager.offload_buckets(current)
        manager.wait_many(current)


def run_mode(
    runtime: turbobus.Runtime,
    manager: turbobus.TrainingOffloadManager,
    names: list[str],
    mode: str,
    active_buckets: int,
    warmup: int,
    iterations: int,
    compute_tensor,
    compute_iterations: int,
    verify: bool,
) -> dict:
    runtime.set_transfer_mode(mode)
    run_warmup(runtime, manager, names[:active_buckets], compute_tensor, compute_iterations, warmup)

    samples = []
    mismatches = []
    for iteration in range(iterations):
        current = active_names(names, iteration, active_buckets)
        manager.mark_on_cpu(current)

        iteration_start = time.perf_counter()
        prefetch_handles = manager.prefetch_buckets(current)
        manager.wait_many(current)
        prefetch = summarize_handle_stats(prefetch_handles)

        compute_ms = run_compute_proxy(runtime, compute_tensor, compute_iterations)

        offload_handles = manager.offload_buckets(current)
        manager.wait_many(current)
        offload = summarize_handle_stats(offload_handles)
        iteration_ms = (time.perf_counter() - iteration_start) * 1000.0
        transfer_ms = prefetch["transfer_ms"] + offload["transfer_ms"]

        sample = {
            "iteration": iteration,
            "bucket_count": len(current),
            "iteration_ms": iteration_ms,
            "transfer_ms": transfer_ms,
            "compute_ms": compute_ms,
            "prefetch": prefetch,
            "offload": offload,
            "prefetch_daemon_reservation": collect_daemon_reservation_info(
                prefetch_handles
            ),
            "offload_daemon_reservation": collect_daemon_reservation_info(
                offload_handles
            ),
            "auto_decision": runtime.last_auto_decision_dict(),
        }
        samples.append(sample)
        print(
            "mode",
            mode,
            "sample_iteration_ms",
            iteration_ms,
            "sample_transfer_ms",
            transfer_ms,
            "sample_compute_ms",
            compute_ms,
            "prefetch_direct_chunks",
            prefetch["direct_chunks"],
            "prefetch_relay_chunks",
            prefetch["relay_chunks"],
            "offload_direct_chunks",
            offload["direct_chunks"],
            "offload_relay_chunks",
            offload["relay_chunks"],
        )

        if verify:
            for name in current:
                bucket = manager.bucket(name)
                cpu_view = bucket.cpu_tensor.narrow(0, bucket.cpu_offset, bucket.bytes)
                gpu_view = bucket.gpu_tensor.narrow(0, bucket.gpu_offset, bucket.bytes)
                if not torch.equal(cpu_view, gpu_view.cpu()):
                    mismatches.append(name)

    summary = summarize_mode(samples)
    result = {
        "mode": mode,
        "active_buckets": active_buckets,
        "samples": samples,
        "summary": summary,
        "verify": len(mismatches) == 0 if verify else None,
        "mismatches": mismatches[:8],
        "last_plan": runtime.last_plan_dict(),
        "last_auto_decision": runtime.last_auto_decision_dict(),
        "prefetch_daemon_reservation": (
            samples[-1]["prefetch_daemon_reservation"] if samples else {}
        ),
        "offload_daemon_reservation": (
            samples[-1]["offload_daemon_reservation"] if samples else {}
        ),
    }
    print(
        "mode",
        mode,
        "median_iteration_ms",
        summary["median_iteration_ms"],
        "median_transfer_ms",
        summary["median_transfer_ms"],
        "median_compute_ms",
        summary["median_compute_ms"],
        "median_gib_per_second",
        summary["median_gib_per_second"],
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
        "TRAINING_OFFLOAD_SUMMARY_BEGIN",
        (
            "training_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"bucket_count={config['bucket_count']} active_buckets={config['active_buckets']} "
            f"bucket_bytes={config['bucket_bytes']} storage_layout={config['storage_layout']} "
            f"chunk_bytes={config['chunk_bytes']} iterations={config['iterations']} "
            f"compute_iterations={config['compute_iterations']} mode={config['mode']} "
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
            "training_mode "
            f"mode={mode} median_iteration_ms={summary['median_iteration_ms']:.3f} "
            f"median_transfer_ms={summary['median_transfer_ms']:.3f} "
            f"median_compute_ms={summary['median_compute_ms']:.3f} "
            f"median_gib_s={summary['median_gib_per_second']:.3f}"
        )
        for op in ("prefetch", "offload"):
            side = summary[op]
            lines.append(
                "training_transfer "
                f"mode={mode} op={op} median_ms={side['median_transfer_ms']:.3f} "
                f"direct_chunks={side['direct_chunks']} relay_chunks={side['relay_chunks']} "
                f"direct_bytes={side['direct_bytes']} relay_bytes={side['relay_bytes']}"
            )
            for path in side["path_stats"]:
                lines.append(
                    "training_path "
                    f"mode={mode} op={op} direction={path['direction']} "
                    f"kind={path['kind']} relay={path['relay_device']} "
                    f"median_gib_s={path['median_gib_per_second']:.3f} "
                    f"median_ms={path['median_cuda_ms']:.3f} "
                    f"bytes={path['bytes']} chunks={path['chunks']}"
                )
        if mode_result["verify"] is not None:
            lines.append(f"training_verify mode={mode} match={mode_result['verify']}")
        prefetch_daemon = daemon_reservation_line(
            mode_result.get("prefetch_daemon_reservation", {})
        )
        if prefetch_daemon:
            lines.append(f"{prefetch_daemon} mode={mode} op=prefetch")
        offload_daemon = daemon_reservation_line(
            mode_result.get("offload_daemon_reservation", {})
        )
        if offload_daemon:
            lines.append(f"{offload_daemon} mode={mode} op=offload")

    for key, value in result["speedups"].items():
        lines.append(f"training_speedup {key}={value:.3f}")
    daemon_profile_initial = daemon_profile_line(result.get("daemon_profile_initial", {}))
    if daemon_profile_initial:
        lines.append(f"{daemon_profile_initial} phase=initial")
    daemon_profile = daemon_profile_line(result.get("daemon_profile", {}))
    if daemon_profile:
        lines.append(f"{daemon_profile} phase=after_profile")
    lines.append("TRAINING_OFFLOAD_SUMMARY_END")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus training offload bucket benchmark")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--active-buckets", type=int)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--min-pool-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mode", choices=["auto", "pool", "direct", "relay", "all"], default="all")
    parser.add_argument("--compute-elements", type=int, default=1_048_576)
    parser.add_argument("--compute-iterations", type=int, default=20)
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
    active_buckets = args.active_buckets or args.bucket_count
    if active_buckets <= 0 or active_buckets > args.bucket_count:
        raise ValueError("--active-buckets must be between 1 and --bucket-count")

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
    manager = make_manager(args, runtime)
    names = manager.names()
    compute_tensor = torch.ones(
        max(1, args.compute_elements),
        dtype=torch.float32,
        device=f"cuda:{args.target_gpu}",
    )

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "bucket_count": args.bucket_count,
            "active_buckets": active_buckets,
            "bucket_bytes": args.bucket_bytes,
            "storage_layout": args.storage_layout,
            "chunk_bytes": args.chunk_bytes,
            "min_pool_bytes": args.min_pool_bytes,
            "profile_bytes": args.profile_bytes,
            "force_profile": args.force_profile,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mode": args.mode,
            "compute_elements": args.compute_elements,
            "compute_iterations": args.compute_iterations,
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
            manager,
            names,
            mode,
            active_buckets,
            args.warmup,
            args.iterations,
            compute_tensor,
            args.compute_iterations,
            args.verify,
        )

    if args.mode == "all":
        direct = result["modes"]["direct"]["summary"]["median_iteration_ms"]
        relay = result["modes"]["relay"]["summary"]["median_iteration_ms"]
        pool = result["modes"]["pool"]["summary"]["median_iteration_ms"]
        auto = result["modes"]["auto"]["summary"]["median_iteration_ms"]
        if pool > 0.0:
            result["speedups"]["direct_over_pool_iteration"] = direct / pool
            result["speedups"]["relay_over_pool_iteration"] = relay / pool
        if auto > 0.0:
            result["speedups"]["direct_over_auto_iteration"] = direct / auto
            result["speedups"]["relay_over_auto_iteration"] = relay / auto

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
