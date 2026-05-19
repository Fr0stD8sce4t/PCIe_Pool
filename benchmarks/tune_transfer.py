from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

import turbobus


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_size_list_mib(value: str) -> list[int]:
    return [int(item) * 1024 * 1024 for item in value.split(",") if item.strip()]


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
    return {
        "bytes": stats.bytes,
        "cuda_elapsed_ms": stats.cuda_elapsed_ms,
        "submit_to_complete_ms": stats.submit_to_complete_ms,
        "gib_per_second": stats.gib_per_second,
        "submit_gib_per_second": stats.submit_gib_per_second,
        "direct_chunks": stats.direct_chunks,
        "relay_chunks": stats.relay_chunks,
    }


def run_candidate(args, cpu, gpu, chunk_bytes: int, staging_slots: int) -> dict:
    options = turbobus.RuntimeOptions(
        chunk_bytes=chunk_bytes,
        staging_slots=staging_slots,
        transfer_mode=turbobus.TransferMode.POOL,
    )
    runtime = turbobus.Runtime(
        target_gpu=args.target_gpu,
        relay_gpus=parse_int_list(args.relay_gpus),
        options=options,
    )
    profile = runtime.profile(args.profile_bytes)

    for _ in range(args.warmup):
        handle = runtime.fetch_to_gpu(cpu, gpu)
        handle.wait()

    samples = []
    for _ in range(args.iterations):
        handle = runtime.fetch_to_gpu(cpu, gpu)
        handle.wait()
        samples.append(stats_to_dict(handle.stats))

    bandwidths = [sample["gib_per_second"] for sample in samples]
    median = statistics.median(bandwidths) if bandwidths else 0.0
    last_stats = samples[-1] if samples else None
    print(
        "candidate",
        "chunk_bytes",
        chunk_bytes,
        "staging_slots",
        staging_slots,
        "median_gib_per_second",
        median,
        "direct_chunks",
        last_stats["direct_chunks"] if last_stats else 0,
        "relay_chunks",
        last_stats["relay_chunks"] if last_stats else 0,
    )
    return {
        "chunk_bytes": chunk_bytes,
        "staging_slots": staging_slots,
        "median_gib_per_second": median,
        "profile": profile_to_dict(profile),
        "samples": samples,
        "last_stats": last_stats,
    }


def write_json(path: str, result: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune TurboBus chunk size and staging slots")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--chunk-mib", default="4,8,16,32,64")
    parser.add_argument("--staging-slots", default="2,3,4")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    torch.cuda.set_device(args.target_gpu)
    cpu = torch.arange(args.bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(args.bytes, dtype=torch.uint8, device=f"cuda:{args.target_gpu}")

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": parse_int_list(args.relay_gpus),
            "bytes": args.bytes,
            "profile_bytes": args.profile_bytes,
            "chunk_mib": args.chunk_mib,
            "staging_slots": args.staging_slots,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "candidates": [],
        "best": None,
    }

    for chunk_bytes in parse_size_list_mib(args.chunk_mib):
        for staging_slots in parse_int_list(args.staging_slots):
            candidate = run_candidate(args, cpu, gpu, chunk_bytes, staging_slots)
            result["candidates"].append(candidate)
            if (
                result["best"] is None
                or candidate["median_gib_per_second"]
                > result["best"]["median_gib_per_second"]
            ):
                result["best"] = {
                    "chunk_bytes": candidate["chunk_bytes"],
                    "staging_slots": candidate["staging_slots"],
                    "median_gib_per_second": candidate["median_gib_per_second"],
                }

    if result["best"] is not None:
        print(
            "best",
            "chunk_bytes",
            result["best"]["chunk_bytes"],
            "staging_slots",
            result["best"]["staging_slots"],
            "median_gib_per_second",
            result["best"]["median_gib_per_second"],
        )

    if args.json_output:
        write_json(args.json_output, result)
        print("json_output", args.json_output)


if __name__ == "__main__":
    main()
