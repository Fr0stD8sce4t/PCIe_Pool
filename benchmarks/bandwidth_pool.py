from __future__ import annotations

import argparse
import statistics

import torch

import turbobus


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus bandwidth pool smoke benchmark")
    parser.add_argument("--target-gpu", type=int, default=0)
    parser.add_argument("--relay-gpus", default="1")
    parser.add_argument("--bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
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

    for _ in range(args.warmup):
        handle = runtime.fetch_to_gpu(cpu, gpu)
        handle.wait()

    samples = []
    for _ in range(args.iterations):
        handle = runtime.fetch_to_gpu(cpu, gpu)
        handle.wait()
        stats = handle.stats
        samples.append(stats.gib_per_second)
        print(
            "sample_gib_per_second",
            stats.gib_per_second,
            "milliseconds",
            stats.submit_to_complete_ms,
            "direct_chunks",
            stats.direct_chunks,
            "relay_chunks",
            stats.relay_chunks,
        )

    if samples:
        print("median_gib_per_second", statistics.median(samples))

    if args.verify:
        print("match", torch.equal(cpu, gpu.cpu()))


if __name__ == "__main__":
    main()
