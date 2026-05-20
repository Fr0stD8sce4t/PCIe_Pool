from __future__ import annotations

import argparse
import statistics
import threading
import time

try:
    import torch
except ImportError as exc:  # pragma: no cover - import-time convenience only
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

import turbobus
if torch is not None:
    from inference_offload_sim import (
        DummyCompute,
        fill_block,
        make_block,
        parse_relay_gpus,
        percentile,
        profile_to_dict,
        stats_to_dict,
        write_json,
        write_text,
    )


def block_names(session_count: int, blocks_per_session: int) -> list[str]:
    names = []
    for session_id in range(session_count):
        for block_index in range(blocks_per_session):
            names.append(f"session{session_id}_prefix{block_index}")
    return names


def create_store(args, runtime) -> tuple[turbobus.OffloadManager, list[str]]:
    store = turbobus.OffloadManager(runtime)
    names = block_names(args.sessions, args.blocks_per_session)
    total_blocks = len(names)

    if args.storage_layout == "packed":
        total_bytes = total_blocks * args.block_bytes
        cpu_backing = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        gpu_backing = torch.empty_like(cpu_backing, device=f"cuda:{args.target_gpu}")
    else:
        cpu_backing = None
        gpu_backing = None

    for block_index, name in enumerate(names):
        if args.storage_layout == "packed":
            offset = block_index * args.block_bytes
            fill_block(cpu_backing, offset, args.block_bytes, block_index)
            store.add(
                name,
                cpu_backing,
                gpu_backing,
                block_id=block_index,
                cpu_slot=block_index,
                gpu_slot=block_index,
                cpu_offset=offset,
                gpu_offset=offset,
                byte_count=args.block_bytes,
            )
        else:
            cpu_tensor = make_block(args.block_bytes, block_index)
            gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{args.target_gpu}")
            store.add(
                name,
                cpu_tensor,
                gpu_tensor,
                block_id=block_index,
                cpu_slot=block_index,
                gpu_slot=block_index,
            )
    return store, names


def restore_window(names: list[str], start: int, count: int) -> list[str]:
    return [names[(start + index) % len(names)] for index in range(count)]


def restore_batch(store: turbobus.OffloadManager, names: list[str]) -> dict:
    start = time.perf_counter()
    handles = store.prefetch_many(names)
    unique_stats = []
    seen_handles = set()
    for name, handle in zip(names, handles, strict=False):
        store.wait(name)
        handle_key = id(handle)
        if handle_key not in seen_handles:
            unique_stats.append(stats_to_dict(handle.stats))
            seen_handles.add(handle_key)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    bytes_ = sum(sample["bytes"] for sample in unique_stats)
    return {
        "blocks": len(names),
        "bytes": bytes_,
        "elapsed_ms": elapsed_ms,
        "gib_per_second": (
            (bytes_ / (1024**3)) / (elapsed_ms / 1000.0) if elapsed_ms > 0.0 else 0.0
        ),
        "direct_bytes": sum(sample["direct_bytes"] for sample in unique_stats),
        "relay_bytes": sum(sample["relay_bytes"] for sample in unique_stats),
        "direct_chunks": sum(sample["direct_chunks"] for sample in unique_stats),
        "relay_chunks": sum(sample["relay_chunks"] for sample in unique_stats),
    }


def run_restore_step(
    store: turbobus.OffloadManager,
    names: list[str],
    compute: DummyCompute,
    overlap_compute: bool,
) -> dict:
    start = time.perf_counter()
    compute_ms = 0.0
    if overlap_compute and compute.enabled:
        result: dict[str, dict] = {}
        worker = threading.Thread(
            target=lambda: result.update({"restore": restore_batch(store, names)})
        )
        worker.start()
        compute_ms = compute.run()
        worker.join()
        restore = result["restore"]
    else:
        restore = restore_batch(store, names)
        compute_ms = compute.run()
    step_ms = (time.perf_counter() - start) * 1000.0
    return {
        "names": names,
        "restore": restore,
        "compute_ms": compute_ms,
        "step_ms": step_ms,
    }


def run_mode(runtime, args, mode: str) -> dict:
    runtime.set_transfer_mode(mode)
    store, names = create_store(args, runtime)
    compute = DummyCompute(
        runtime,
        args.compute_impl,
        args.compute_ms,
        args.cuda_compute_elements,
        args.cuda_compute_iterations,
    )
    steps = []
    verified = None

    for iteration in range(args.iterations):
        selected = restore_window(names, iteration * args.restore_blocks, args.restore_blocks)
        steps.append(run_restore_step(store, selected, compute, args.overlap_compute))

    if args.verify:
        verified = verify_blocks(store, sorted({name for step in steps for name in step["names"]}))

    summary = summarize_steps(steps)
    print(
        "mode",
        mode,
        "restore_gib_per_second",
        summary["restore_gib_per_second"],
        "restore_ms_p50",
        summary["restore_ms_p50"],
        "step_ms_p50",
        summary["step_ms_p50"],
        "compute_ms_p50",
        summary["compute_ms_p50"],
        "restored_blocks",
        summary["restored_blocks"],
    )
    return {
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "verified": verified,
    }


def summarize_steps(steps: list[dict]) -> dict:
    restore_ms = [step["restore"]["elapsed_ms"] for step in steps]
    step_ms = [step["step_ms"] for step in steps]
    compute_ms = [step["compute_ms"] for step in steps]
    bytes_ = sum(step["restore"]["bytes"] for step in steps)
    restore_seconds = sum(restore_ms) / 1000.0
    return {
        "iterations": len(steps),
        "restored_blocks": sum(step["restore"]["blocks"] for step in steps),
        "restore_bytes": bytes_,
        "restore_gib_per_second": (
            (bytes_ / (1024**3)) / restore_seconds if restore_seconds > 0.0 else 0.0
        ),
        "restore_ms_p50": statistics.median(restore_ms) if restore_ms else 0.0,
        "restore_ms_p95": percentile(restore_ms, 95.0),
        "step_ms_p50": statistics.median(step_ms) if step_ms else 0.0,
        "step_ms_p95": percentile(step_ms, 95.0),
        "compute_ms_p50": statistics.median(compute_ms) if compute_ms else 0.0,
        "compute_ms_p95": percentile(compute_ms, 95.0),
        "direct_bytes": sum(step["restore"]["direct_bytes"] for step in steps),
        "relay_bytes": sum(step["restore"]["relay_bytes"] for step in steps),
        "direct_chunks": sum(step["restore"]["direct_chunks"] for step in steps),
        "relay_chunks": sum(step["restore"]["relay_chunks"] for step in steps),
    }


def verify_blocks(store: turbobus.OffloadManager, names: list[str]) -> bool:
    for name in names:
        block = store.block(name)
        cpu_view = block.cpu_tensor.narrow(0, block.cpu_offset, block.bytes)
        gpu_view = block.gpu_tensor.narrow(0, block.gpu_offset, block.bytes).cpu()
        if not torch.equal(cpu_view, gpu_view):
            return False
    return True


def compact_summary(result: dict) -> str:
    config = result["config"]
    lines = [
        "COPY_SUMMARY_BEGIN",
        (
            "poc_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"sessions={config['sessions']} blocks_per_session={config['blocks_per_session']} "
            f"restore_blocks={config['restore_blocks']} iterations={config['iterations']} "
            f"storage_layout={config['storage_layout']} block_bytes={config['block_bytes']} "
            f"compute_impl={config['compute_impl']} overlap_compute={config['overlap_compute']} "
            f"cuda_compute_elements={config['cuda_compute_elements']} "
            f"cuda_compute_iterations={config['cuda_compute_iterations']} "
            f"mode={config['mode']} dynamic_weights={config['dynamic_weights']}"
        ),
        (
            "poc_scenario "
            "type=prefix_session_restore "
            "boundary=framework_adjacent "
            "transfer=cpu_pinned_prefix_kv_to_gpu_slots "
            "policy=no_scheduler_rewrite "
            f"verify={config['verify']} "
            "note=first_real_inference_poc_boundary"
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
        summary = mode_result["summary"]
        line = (
            "poc_mode "
            f"mode={mode} restore_gib_s={summary['restore_gib_per_second']:.3f} "
            f"restore_p50_ms={summary['restore_ms_p50']:.3f} "
            f"restore_p95_ms={summary['restore_ms_p95']:.3f} "
            f"step_p50_ms={summary['step_ms_p50']:.3f} "
            f"step_p95_ms={summary['step_ms_p95']:.3f} "
            f"compute_p50_ms={summary['compute_ms_p50']:.3f} "
            f"compute_p95_ms={summary['compute_ms_p95']:.3f} "
            f"restored_blocks={summary['restored_blocks']} "
            f"direct_chunks={summary['direct_chunks']} relay_chunks={summary['relay_chunks']}"
        )
        if mode_result["verified"] is not None:
            line += f" verified={mode_result['verified']}"
        lines.append(line)
    for key, value in result["speedups"].items():
        lines.append(f"poc_speedup {key}={value:.3f}")
    lines.append("COPY_SUMMARY_END")
    return "\n".join(lines)


def main() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to run the prefix restore POC benchmark"
        ) from _TORCH_IMPORT_ERROR

    parser = argparse.ArgumentParser(description="TurboBus prefix/session restore POC")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--blocks-per-session", type=int, default=8)
    parser.add_argument("--restore-blocks", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--block-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--compute-ms", type=float, default=0.0)
    parser.add_argument("--compute-impl", choices=["sleep", "cuda"], default="cuda")
    parser.add_argument("--cuda-compute-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--cuda-compute-iterations", type=int, default=2048)
    parser.add_argument("--overlap-compute", action="store_true")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mode", choices=["pool", "direct", "relay", "all"], default="all")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-weight-alpha", type=float, default=0.25)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    args = parser.parse_args()

    validate_args(args)
    relays = parse_relay_gpus(args.relay_gpus)
    torch.cuda.set_device(args.target_gpu)
    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        enable_dynamic_weights=args.dynamic_weights,
        dynamic_weight_alpha=args.dynamic_weight_alpha,
    )
    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays, options=options)
    profile = runtime.profile(args.profile_bytes, force=True)

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "sessions": args.sessions,
            "blocks_per_session": args.blocks_per_session,
            "restore_blocks": args.restore_blocks,
            "iterations": args.iterations,
            "storage_layout": args.storage_layout,
            "block_bytes": args.block_bytes,
            "compute_ms": args.compute_ms,
            "compute_impl": args.compute_impl,
            "cuda_compute_elements": args.cuda_compute_elements,
            "cuda_compute_iterations": args.cuda_compute_iterations,
            "overlap_compute": args.overlap_compute,
            "chunk_bytes": args.chunk_bytes,
            "profile_bytes": args.profile_bytes,
            "mode": args.mode,
            "dynamic_weights": args.dynamic_weights,
            "dynamic_weight_alpha": args.dynamic_weight_alpha,
            "verify": args.verify,
        },
        "profile": profile_to_dict(profile),
        "modes": {},
        "speedups": {},
    }

    modes = ["direct", "relay", "pool"] if args.mode == "all" else [args.mode]
    for mode in modes:
        result["modes"][mode] = run_mode(runtime, args, mode)

    if args.mode == "all":
        direct = result["modes"]["direct"]["summary"]["restore_gib_per_second"]
        relay = result["modes"]["relay"]["summary"]["restore_gib_per_second"]
        pool = result["modes"]["pool"]["summary"]["restore_gib_per_second"]
        if direct > 0.0:
            result["speedups"]["pool_over_direct_restore"] = pool / direct
        if relay > 0.0:
            result["speedups"]["pool_over_relay_restore"] = pool / relay

    if args.json_output:
        write_json(args.json_output, result)
        print("json_output", args.json_output)
    summary = compact_summary(result)
    if args.summary_output:
        write_text(args.summary_output, summary)
        print("summary_output", args.summary_output)
    if not args.no_copy_summary:
        print(summary)


def validate_args(args) -> None:
    if args.sessions <= 0:
        raise ValueError("--sessions must be positive")
    if args.blocks_per_session <= 0:
        raise ValueError("--blocks-per-session must be positive")
    if args.restore_blocks <= 0:
        raise ValueError("--restore-blocks must be positive")
    if args.restore_blocks > args.sessions * args.blocks_per_session:
        raise ValueError("--restore-blocks cannot exceed the total block count")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.block_bytes <= 0:
        raise ValueError("--block-bytes must be positive")
    if args.cuda_compute_elements <= 0:
        raise ValueError("--cuda-compute-elements must be positive")
    if args.cuda_compute_iterations <= 0:
        raise ValueError("--cuda-compute-iterations must be positive")


if __name__ == "__main__":
    main()
