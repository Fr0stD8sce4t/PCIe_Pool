from __future__ import annotations

import argparse

import torch

import turbobus


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus bandwidth pool smoke benchmark")
    parser.add_argument("--target-gpu", type=int, default=0)
    parser.add_argument("--relay-gpus", default="1")
    parser.add_argument("--bytes", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()

    relays = [int(item) for item in args.relay_gpus.split(",") if item.strip()]
    cpu = torch.empty(args.bytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(args.bytes, dtype=torch.uint8, device=f"cuda:{args.target_gpu}")

    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays)
    profile = runtime.profile()
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

    handle = runtime.fetch_to_gpu(cpu, gpu)
    handle.wait()


if __name__ == "__main__":
    main()
