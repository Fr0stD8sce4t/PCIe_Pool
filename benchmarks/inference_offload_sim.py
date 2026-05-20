from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import random
import statistics
import threading
import time

import torch

import turbobus


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percent / 100.0) * len(ordered)) - 1)
    return ordered[index]


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
        "direct_bytes": stats.direct_bytes,
        "relay_bytes": stats.relay_bytes,
        "submit_to_complete_ms": stats.submit_to_complete_ms,
        "gib_per_second": stats.gib_per_second,
        "direct_chunks": stats.direct_chunks,
        "relay_chunks": stats.relay_chunks,
    }


def make_block(block_bytes: int, block_index: int):
    tensor = torch.arange(block_bytes, dtype=torch.uint8, pin_memory=True)
    if block_index:
        tensor.add_(block_index)
    return tensor


def fill_block(tensor, offset: int, block_bytes: int, block_index: int) -> None:
    view = tensor.narrow(0, offset, block_bytes)
    view.copy_(torch.arange(block_bytes, dtype=torch.uint8, pin_memory=True))
    if block_index:
        view.add_(block_index)


def request_blocks(request_id: int, blocks_per_request: int) -> list[str]:
    return [f"req{request_id}_kv{index}" for index in range(blocks_per_request)]


def create_store(args, runtime) -> tuple[turbobus.OffloadManager, list[list[str]]]:
    store = turbobus.OffloadManager(runtime)
    request_block_names = []
    total_blocks = args.requests * args.blocks_per_request

    if args.storage_layout == "packed":
        total_bytes = total_blocks * args.block_bytes
        cpu_backing = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        gpu_backing = torch.empty_like(cpu_backing, device=f"cuda:{args.target_gpu}")
    else:
        cpu_backing = None
        gpu_backing = None

    for request_id in range(args.requests):
        names = request_blocks(request_id, args.blocks_per_request)
        request_block_names.append(names)
        for block_index, name in enumerate(names):
            global_index = request_id * args.blocks_per_request + block_index
            if args.storage_layout == "packed":
                offset = global_index * args.block_bytes
                fill_block(cpu_backing, offset, args.block_bytes, global_index)
                store.add(
                    name,
                    cpu_backing,
                    gpu_backing,
                    block_id=(request_id, block_index),
                    cpu_slot=global_index,
                    gpu_slot=global_index,
                    cpu_offset=offset,
                    gpu_offset=offset,
                    byte_count=args.block_bytes,
                )
            else:
                cpu_tensor = make_block(args.block_bytes, global_index)
                gpu_tensor = torch.empty_like(cpu_tensor, device=f"cuda:{args.target_gpu}")
                store.add(
                    name,
                    cpu_tensor,
                    gpu_tensor,
                    block_id=(request_id, block_index),
                    cpu_slot=global_index,
                    gpu_slot=None,
                )
    return store, request_block_names


def select_blocks(
    blocks: list[str],
    step: int,
    blocks_per_step: int,
    access_pattern: str,
    rng: random.Random,
) -> list[str]:
    if access_pattern == "round_robin":
        start = (step * blocks_per_step) % len(blocks)
        return block_window(blocks, start, blocks_per_step)
    if access_pattern == "sliding":
        start = step % len(blocks)
        return block_window(blocks, start, blocks_per_step)
    if access_pattern == "random":
        return rng.sample(blocks, blocks_per_step)
    raise ValueError(f"unknown access pattern: {access_pattern}")


def block_window(blocks: list[str], start: int, blocks_per_step: int) -> list[str]:
    return [blocks[(start + offset) % len(blocks)] for offset in range(blocks_per_step)]


class ResidentSet:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._entries: OrderedDict[str, None] = OrderedDict()

    def contains(self, name: str) -> bool:
        return name in self._entries

    def touch(self, name: str) -> None:
        if name in self._entries:
            self._entries.move_to_end(name)
            return
        self._entries[name] = None

    def victims_for(self, incoming: list[str]) -> list[str]:
        victims = []
        missing = [name for name in incoming if name not in self._entries]
        protected = set(incoming)
        while len(self._entries) + len(missing) - len(victims) > self.capacity:
            victim = next((name for name in self._entries if name not in protected), None)
            if victim is None:
                break
            victims.append(victim)
            del self._entries[victim]
        return victims

    def add_many(self, names: list[str]) -> None:
        for name in names:
            self.touch(name)


class DummyCompute:
    def __init__(
        self,
        runtime: turbobus.Runtime,
        impl: str,
        compute_ms: float,
        cuda_elements: int,
        cuda_iterations: int,
    ) -> None:
        self.runtime = runtime
        self.impl = impl
        self.compute_ms = compute_ms
        self.cuda_elements = cuda_elements
        self.cuda_iterations = cuda_iterations
        self.cuda_tensor = None
        if self.impl == "cuda":
            self.cuda_tensor = torch.zeros(
                self.cuda_elements,
                dtype=torch.float32,
                device=f"cuda:{self.runtime.target_gpu}",
            )

    @property
    def enabled(self) -> bool:
        if self.impl == "cuda":
            return True
        return self.compute_ms > 0.0

    def run(self) -> float:
        if not self.enabled:
            return 0.0
        start = time.perf_counter()
        if self.impl == "sleep":
            time.sleep(self.compute_ms / 1000.0)
        elif self.impl == "cuda":
            self.runtime.run_dummy_compute(self.cuda_tensor, self.cuda_iterations)
        else:
            raise ValueError(f"unknown compute impl: {self.impl}")
        return (time.perf_counter() - start) * 1000.0


def transfer_batch(store: turbobus.OffloadManager, names: list[str], op: str) -> dict:
    if not names:
        return empty_transfer_batch(op)

    start = time.perf_counter()
    if op == "prefetch":
        handles = store.prefetch_many(names)
    elif op == "evict":
        handles = store.evict_many(names)
    else:
        raise ValueError(f"unknown transfer op: {op}")

    samples = []
    unique_stats = []
    seen_handles = set()
    for name, handle in zip(names, handles, strict=False):
        store.wait(name)
        stats = stats_to_dict(handle.stats)
        samples.append({"block": name, **stats})
        handle_key = id(handle)
        if handle_key not in seen_handles:
            unique_stats.append(stats)
            seen_handles.add(handle_key)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    bytes_ = sum(sample["bytes"] for sample in unique_stats)
    return {
        "op": op,
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
        "samples": samples,
    }


def run_transfers(
    store: turbobus.OffloadManager,
    victims: list[str],
    missing: list[str],
) -> tuple[dict, dict]:
    evict = transfer_batch(store, victims, "evict")
    prefetch = transfer_batch(store, missing, "prefetch")
    return evict, prefetch


def run_step_work(
    store: turbobus.OffloadManager,
    victims: list[str],
    missing: list[str],
    compute: DummyCompute,
    overlap_compute: bool,
) -> tuple[dict, dict, float, float, float]:
    transfer_start = time.perf_counter()
    compute_elapsed_ms = 0.0
    if overlap_compute and compute.enabled:
        result: dict[str, tuple[dict, dict]] = {}
        worker = threading.Thread(
            target=lambda: result.update(
                {"transfers": run_transfers(store, victims, missing)}
            )
        )
        worker.start()
        compute_elapsed_ms = compute.run()
        worker.join()
        evict, prefetch = result["transfers"]
    else:
        evict, prefetch = run_transfers(store, victims, missing)
        compute_elapsed_ms = compute.run()
    elapsed_ms = (time.perf_counter() - transfer_start) * 1000.0
    transfer_ms = evict["elapsed_ms"] + prefetch["elapsed_ms"]
    return evict, prefetch, transfer_ms, compute_elapsed_ms, elapsed_ms


def empty_transfer_batch(op: str) -> dict:
    return {
        "op": op,
        "blocks": 0,
        "bytes": 0,
        "elapsed_ms": 0.0,
        "gib_per_second": 0.0,
        "direct_bytes": 0,
        "relay_bytes": 0,
        "direct_chunks": 0,
        "relay_chunks": 0,
        "samples": [],
    }


def summarize_steps(steps: list[dict]) -> dict:
    step_latencies = [step["step_ms"] for step in steps]
    transfer_stalls = [step["transfer_ms"] for step in steps]
    compute_latencies = [step["compute_elapsed_ms"] for step in steps]
    prefetch_batches = [step["prefetch"] for step in steps if step["prefetch"]["blocks"]]
    evict_batches = [step["evict"] for step in steps if step["evict"]["blocks"]]
    total_tokens = len(steps)
    total_ms = sum(step_latencies)
    return {
        "steps": total_tokens,
        "tokens_per_second": total_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
        "step_ms_p50": statistics.median(step_latencies) if step_latencies else 0.0,
        "step_ms_p95": percentile(step_latencies, 95.0),
        "transfer_ms_p50": statistics.median(transfer_stalls) if transfer_stalls else 0.0,
        "transfer_ms_p95": percentile(transfer_stalls, 95.0),
        "compute_ms_p50": statistics.median(compute_latencies) if compute_latencies else 0.0,
        "compute_ms_p95": percentile(compute_latencies, 95.0),
        "prefetch_gib_per_second": summarize_transfer_gib(prefetch_batches),
        "evict_gib_per_second": summarize_transfer_gib(evict_batches),
        "prefetch_batches": len(prefetch_batches),
        "evict_batches": len(evict_batches),
        "prefetch_blocks": sum(batch["blocks"] for batch in prefetch_batches),
        "evict_blocks": sum(batch["blocks"] for batch in evict_batches),
        "direct_chunks": sum(
            step["prefetch"]["direct_chunks"] + step["evict"]["direct_chunks"]
            for step in steps
        ),
        "relay_chunks": sum(
            step["prefetch"]["relay_chunks"] + step["evict"]["relay_chunks"]
            for step in steps
        ),
        "direct_bytes": sum(
            step["prefetch"]["direct_bytes"] + step["evict"]["direct_bytes"]
            for step in steps
        ),
        "relay_bytes": sum(
            step["prefetch"]["relay_bytes"] + step["evict"]["relay_bytes"]
            for step in steps
        ),
    }


def summarize_transfer_gib(batches: list[dict]) -> float:
    bytes_ = sum(batch["bytes"] for batch in batches)
    seconds = sum(batch["elapsed_ms"] for batch in batches) / 1000.0
    return (bytes_ / (1024**3)) / seconds if seconds > 0.0 else 0.0


def run_mode(
    runtime: turbobus.Runtime,
    store: turbobus.OffloadManager,
    request_block_names: list[list[str]],
    mode: str,
    gpu_block_capacity: int,
    blocks_per_step: int,
    access_pattern: str,
    working_set_blocks: int,
    seed: int,
    decode_steps: int,
    compute: DummyCompute,
    overlap_compute: bool,
) -> dict:
    runtime.set_transfer_mode(mode)
    resident = ResidentSet(gpu_block_capacity)
    rng = random.Random(seed)
    steps = []

    for step_index in range(decode_steps):
        request_id = step_index % len(request_block_names)
        working_set = request_block_names[request_id][:working_set_blocks]
        needed = select_blocks(
            working_set,
            step_index,
            blocks_per_step,
            access_pattern,
            rng,
        )
        missing = [name for name in needed if not resident.contains(name)]
        victims = resident.victims_for(missing)

        step_start = time.perf_counter()
        evict, prefetch, transfer_ms, compute_elapsed_ms, overlapped_ms = run_step_work(
            store,
            victims,
            missing,
            compute,
            overlap_compute,
        )
        resident.add_many(needed)
        step_ms = (time.perf_counter() - step_start) * 1000.0

        steps.append(
            {
                "step": step_index,
                "request_id": request_id,
                "needed": needed,
                "missing": missing,
                "victims": victims,
                "evict": evict,
                "prefetch": prefetch,
                "transfer_ms": transfer_ms,
                "compute_ms": compute.compute_ms,
                "compute_elapsed_ms": compute_elapsed_ms,
                "overlapped_ms": overlapped_ms,
                "step_ms": step_ms,
            }
        )

    summary = summarize_steps(steps)
    print(
        "mode",
        mode,
        "tokens_per_second",
        summary["tokens_per_second"],
        "step_ms_p50",
        summary["step_ms_p50"],
        "transfer_ms_p50",
        summary["transfer_ms_p50"],
        "compute_ms_p50",
        summary["compute_ms_p50"],
        "prefetch_blocks",
        summary["prefetch_blocks"],
        "evict_blocks",
        summary["evict_blocks"],
    )
    return {"mode": mode, "summary": summary, "steps": steps}


def compact_summary(result: dict) -> str:
    config = result["config"]
    lines = [
        "COPY_SUMMARY_BEGIN",
        (
            "sim_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"requests={config['requests']} blocks_per_request={config['blocks_per_request']} "
            f"blocks_per_step={config['blocks_per_step']} gpu_block_capacity={config['gpu_block_capacity']} "
            f"access_pattern={config['access_pattern']} working_set_blocks={config['working_set_blocks']} "
            f"seed={config['seed']} storage_layout={config['storage_layout']} "
            f"block_bytes={config['block_bytes']} decode_steps={config['decode_steps']} "
            f"compute_ms={config['compute_ms']} overlap_compute={config['overlap_compute']} "
            f"compute_impl={config['compute_impl']} "
            f"cuda_compute_elements={config['cuda_compute_elements']} "
            f"cuda_compute_iterations={config['cuda_compute_iterations']} "
            f"mode={config['mode']} "
            f"dynamic_weights={config['dynamic_weights']}"
        ),
        (
            "sim_scenario "
            "type=capacity_pressure "
            "unit=decode_step "
            "policy=lru_eviction "
            "transfer=evict_then_prefetch "
            f"access_pattern={config['access_pattern']} "
            f"working_set_blocks={config['working_set_blocks']} "
            f"gpu_block_capacity={config['gpu_block_capacity']} "
            f"blocks_per_step={config['blocks_per_step']} "
            f"dummy_compute_ms={config['compute_ms']} "
            f"overlap_compute={config['overlap_compute']} "
            f"compute_impl={config['compute_impl']} "
            f"cuda_compute_elements={config['cuda_compute_elements']} "
            f"cuda_compute_iterations={config['cuda_compute_iterations']} "
            f"note={config['compute_note']}"
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
        lines.append(
            "sim_mode "
            f"mode={mode} tokens_s={summary['tokens_per_second']:.3f} "
            f"step_p50_ms={summary['step_ms_p50']:.3f} "
            f"step_p95_ms={summary['step_ms_p95']:.3f} "
            f"transfer_p50_ms={summary['transfer_ms_p50']:.3f} "
            f"transfer_p95_ms={summary['transfer_ms_p95']:.3f} "
            f"compute_p50_ms={summary['compute_ms_p50']:.3f} "
            f"compute_p95_ms={summary['compute_ms_p95']:.3f} "
            f"prefetch_gib_s={summary['prefetch_gib_per_second']:.3f} "
            f"evict_gib_s={summary['evict_gib_per_second']:.3f} "
            f"prefetch_blocks={summary['prefetch_blocks']} "
            f"evict_blocks={summary['evict_blocks']} "
            f"direct_chunks={summary['direct_chunks']} relay_chunks={summary['relay_chunks']}"
        )

    for key, value in result["speedups"].items():
        lines.append(f"sim_speedup {key}={value:.3f}")
    lines.append("COPY_SUMMARY_END")
    return "\n".join(lines)


def write_json(path: str, result: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def write_text(path: str, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus inference offload simulator")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--blocks-per-request", type=int, default=8)
    parser.add_argument("--blocks-per-step", type=int, default=4)
    parser.add_argument("--gpu-block-capacity", type=int, default=4)
    parser.add_argument(
        "--access-pattern",
        choices=["round_robin", "sliding", "random"],
        default="round_robin",
    )
    parser.add_argument("--working-set-blocks", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--storage-layout",
        choices=["separate", "packed"],
        default="separate",
    )
    parser.add_argument("--block-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--compute-ms", type=float, default=0.0)
    parser.add_argument("--compute-impl", choices=["sleep", "cuda"], default="sleep")
    parser.add_argument("--cuda-compute-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--cuda-compute-iterations", type=int, default=64)
    parser.add_argument(
        "--overlap-compute",
        action="store_true",
        help="run dummy compute concurrently with prefetch/evict in each step",
    )
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mode", choices=["pool", "direct", "relay", "all"], default="pool")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-weight-alpha", type=float, default=0.25)
    parser.add_argument("--json-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    args = parser.parse_args()

    if args.blocks_per_step <= 0 or args.blocks_per_step > args.blocks_per_request:
        raise ValueError("--blocks-per-step must be between 1 and --blocks-per-request")
    if args.gpu_block_capacity < args.blocks_per_step:
        raise ValueError("--gpu-block-capacity must be at least --blocks-per-step")
    working_set_blocks = args.working_set_blocks or args.blocks_per_request
    if working_set_blocks < args.blocks_per_step:
        raise ValueError("--working-set-blocks must be at least --blocks-per-step")
    if working_set_blocks > args.blocks_per_request:
        raise ValueError("--working-set-blocks must be at most --blocks-per-request")
    if args.cuda_compute_elements <= 0:
        raise ValueError("--cuda-compute-elements must be positive")
    if args.cuda_compute_iterations <= 0:
        raise ValueError("--cuda-compute-iterations must be positive")

    relays = parse_relay_gpus(args.relay_gpus)
    torch.cuda.set_device(args.target_gpu)
    options = turbobus.RuntimeOptions(
        chunk_bytes=args.chunk_bytes,
        enable_dynamic_weights=args.dynamic_weights,
        dynamic_weight_alpha=args.dynamic_weight_alpha,
    )
    runtime = turbobus.Runtime(target_gpu=args.target_gpu, relay_gpus=relays, options=options)
    profile = runtime.profile(args.profile_bytes, force=True)
    store, request_block_names = create_store(args, runtime)
    compute = DummyCompute(
        runtime,
        args.compute_impl,
        args.compute_ms,
        args.cuda_compute_elements,
        args.cuda_compute_iterations,
    )
    compute_note = (
        "cuda_kernel_overlap_model"
        if args.compute_impl == "cuda"
        else "python_sleep_not_cuda_kernel_overlap"
    )

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "requests": args.requests,
            "blocks_per_request": args.blocks_per_request,
            "blocks_per_step": args.blocks_per_step,
            "gpu_block_capacity": args.gpu_block_capacity,
            "access_pattern": args.access_pattern,
            "working_set_blocks": working_set_blocks,
            "seed": args.seed,
            "storage_layout": args.storage_layout,
            "block_bytes": args.block_bytes,
            "decode_steps": args.decode_steps,
            "compute_ms": args.compute_ms,
            "compute_impl": args.compute_impl,
            "cuda_compute_elements": args.cuda_compute_elements,
            "cuda_compute_iterations": args.cuda_compute_iterations,
            "compute_note": compute_note,
            "overlap_compute": args.overlap_compute,
            "chunk_bytes": args.chunk_bytes,
            "profile_bytes": args.profile_bytes,
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
            request_block_names,
            mode,
            args.gpu_block_capacity,
            args.blocks_per_step,
            args.access_pattern,
            working_set_blocks,
            args.seed,
            args.decode_steps,
            compute,
            args.overlap_compute,
        )

    if args.mode == "all":
        direct_tokens = result["modes"]["direct"]["summary"]["tokens_per_second"]
        relay_tokens = result["modes"]["relay"]["summary"]["tokens_per_second"]
        pool_tokens = result["modes"]["pool"]["summary"]["tokens_per_second"]
        if direct_tokens > 0.0:
            result["speedups"]["pool_over_direct_tokens_per_second"] = (
                pool_tokens / direct_tokens
            )
        if relay_tokens > 0.0:
            result["speedups"]["pool_over_relay_tokens_per_second"] = (
                pool_tokens / relay_tokens
            )

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
