from __future__ import annotations

import argparse
import statistics

import torch

import turbobus


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run_mode(runtime: turbobus.Runtime, cpu, gpu, mode: str, warmup: int, iterations: int):
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
        samples.append(stats.gib_per_second)
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

    median = statistics.median(samples) if samples else 0.0
    print("mode", mode, "median_gib_per_second", median)
    return median, last_stats


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
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    relays = parse_relay_gpus(args.relay_gpus)
    torch.cuda.set_device(args.target_gpu)

    cpu = torch.arange(args.bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(args.bytes, dtype=torch.uint8, device=f"cuda:{args.target_gpu}")

    options = turbobus.RuntimeOptions(chunk_bytes=args.chunk_bytes)
    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays, options=options)
    profile = runtime.profile(args.profile_bytes)

    print("target_gpu", args.target_gpu)
    print("relay_gpus", ",".join(str(relay) for relay in relays))
    print("bytes", args.bytes)
    print("chunk_bytes", args.chunk_bytes)
    print("warmup", args.warmup)
    print("iterations", args.iterations)
    print("mode", args.mode)
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
        median, _ = run_mode(runtime, cpu, gpu, mode, args.warmup, args.iterations)
        medians[mode] = median

    if args.mode == "all":
        direct = medians.get("direct", 0.0)
        relay = medians.get("relay", 0.0)
        pool = medians.get("pool", 0.0)
        if direct > 0.0:
            print("pool_over_direct_median", pool / direct)
        if relay > 0.0:
            print("pool_over_relay_median", pool / relay)

    if args.verify:
        print("match", torch.equal(cpu, gpu.cpu()))


if __name__ == "__main__":
    main()
