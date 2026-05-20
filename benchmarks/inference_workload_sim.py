from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import random
import statistics
import time

import torch

import turbobus
from inference_offload_sim import (
    DummyCompute,
    ResidentSet,
    fill_block,
    make_block,
    parse_relay_gpus,
    percentile,
    profile_to_dict,
    run_step_work,
    select_blocks,
    write_json,
    write_text,
)


PRESETS = {
    "light": {
        "request_count": 4,
        "prompt_blocks_min": 4,
        "prompt_blocks_max": 6,
        "decode_steps_min": 8,
        "decode_steps_max": 12,
        "gpu_block_capacity": 16,
        "blocks_per_step": 2,
        "access_pattern": "sliding",
    },
    "pressure": {
        "request_count": 8,
        "prompt_blocks_min": 6,
        "prompt_blocks_max": 10,
        "decode_steps_min": 12,
        "decode_steps_max": 20,
        "gpu_block_capacity": 12,
        "blocks_per_step": 4,
        "access_pattern": "sliding",
    },
    "long_context": {
        "request_count": 8,
        "prompt_blocks_min": 12,
        "prompt_blocks_max": 18,
        "decode_steps_min": 16,
        "decode_steps_max": 28,
        "gpu_block_capacity": 16,
        "blocks_per_step": 6,
        "access_pattern": "sliding",
    },
}


@dataclass
class RequestSpec:
    request_id: int
    arrival_ms: float
    prompt_blocks: int
    decode_steps: int


@dataclass
class RequestState:
    spec: RequestSpec
    blocks: list[str]
    decoded: int = 0
    first_token_ms: float | None = None
    completed_ms: float | None = None


def generate_requests(args) -> list[RequestSpec]:
    rng = random.Random(args.seed)
    arrival_ms = 0.0
    requests = []
    for request_id in range(args.request_count):
        if args.arrival_pattern == "burst":
            arrival_ms = 0.0
        elif args.arrival_pattern == "fixed":
            arrival_ms = request_id * args.arrival_interval_ms
        elif args.arrival_pattern == "poisson":
            if request_id == 0:
                arrival_ms = 0.0
            else:
                rate = 1.0 / max(args.arrival_interval_ms, 0.001)
                arrival_ms += rng.expovariate(rate)
        else:
            raise ValueError(f"unknown arrival pattern: {args.arrival_pattern}")

        requests.append(
            RequestSpec(
                request_id=request_id,
                arrival_ms=arrival_ms,
                prompt_blocks=rng.randint(args.prompt_blocks_min, args.prompt_blocks_max),
                decode_steps=rng.randint(args.decode_steps_min, args.decode_steps_max),
            )
        )
    return requests


def block_names(request: RequestSpec) -> list[str]:
    return [f"req{request.request_id}_kv{index}" for index in range(request.prompt_blocks)]


def create_store(
    args,
    runtime,
    requests: list[RequestSpec],
) -> tuple[turbobus.OffloadManager, dict[int, list[str]]]:
    store = turbobus.OffloadManager(runtime)
    request_blocks = {request.request_id: block_names(request) for request in requests}
    total_blocks = sum(request.prompt_blocks for request in requests)

    if args.storage_layout == "packed":
        total_bytes = total_blocks * args.block_bytes
        cpu_backing = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        gpu_backing = torch.empty_like(cpu_backing, device=f"cuda:{args.target_gpu}")
    else:
        cpu_backing = None
        gpu_backing = None

    global_index = 0
    for request in requests:
        for block_index, name in enumerate(request_blocks[request.request_id]):
            if args.storage_layout == "packed":
                offset = global_index * args.block_bytes
                fill_block(cpu_backing, offset, args.block_bytes, global_index)
                store.add(
                    name,
                    cpu_backing,
                    gpu_backing,
                    block_id=(request.request_id, block_index),
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
                    block_id=(request.request_id, block_index),
                    cpu_slot=global_index,
                    gpu_slot=global_index,
                )
            global_index += 1
    return store, request_blocks


def select_decode_request(
    ready: deque[RequestState],
    scheduler: str,
    rng: random.Random,
) -> RequestState:
    if scheduler == "oldest_first":
        return ready[0]
    if scheduler == "round_robin":
        request = ready.popleft()
        ready.append(request)
        return request
    if scheduler == "random_ready":
        return ready[rng.randrange(len(ready))]
    raise ValueError(f"unknown scheduler: {scheduler}")


def resident_prefill_blocks(blocks: list[str], capacity: int) -> list[str]:
    return blocks[-min(len(blocks), capacity) :]


def run_prefill(
    resident: ResidentSet,
    store: turbobus.OffloadManager,
    blocks: list[str],
    prefill_compute_ms: float,
    prefill_mode: str,
) -> tuple[float, dict]:
    incoming = resident_prefill_blocks(blocks, resident.capacity)
    victims = resident.victims_for(incoming)
    if prefill_mode == "produce_kv_on_gpu":
        evict, prefetch, transfer_ms, _compute_elapsed_ms, elapsed_ms = run_step_work(
            store,
            victims,
            [],
            NullCompute(prefill_compute_ms),
            overlap_compute=False,
        )
    elif prefill_mode == "restore_from_cpu":
        evict, prefetch, transfer_ms, _compute_elapsed_ms, elapsed_ms = run_step_work(
            store,
            victims,
            incoming,
            NullCompute(prefill_compute_ms),
            overlap_compute=False,
        )
    else:
        raise ValueError(f"unknown prefill mode: {prefill_mode}")
    resident.add_many(incoming)
    return elapsed_ms, {
        "evict": evict,
        "prefetch": prefetch,
        "transfer_ms": transfer_ms,
        "blocks": len(blocks),
        "restored_blocks": len(incoming) if prefill_mode == "restore_from_cpu" else 0,
    }


class NullCompute:
    def __init__(self, compute_ms: float) -> None:
        self.compute_ms = compute_ms

    @property
    def enabled(self) -> bool:
        return self.compute_ms > 0.0

    def run(self) -> float:
        if self.compute_ms <= 0.0:
            return 0.0
        start = time.perf_counter()
        time.sleep(self.compute_ms / 1000.0)
        return (time.perf_counter() - start) * 1000.0


def run_mode(
    runtime: turbobus.Runtime,
    store: turbobus.OffloadManager,
    requests: list[RequestSpec],
    request_blocks: dict[int, list[str]],
    mode: str,
    args,
) -> dict:
    runtime.set_transfer_mode(mode)
    rng = random.Random(args.seed)
    resident = ResidentSet(args.gpu_block_capacity)
    compute = DummyCompute(
        runtime,
        args.compute_impl,
        args.compute_ms,
        args.cuda_compute_elements,
        args.cuda_compute_iterations,
    )

    pending = deque(requests)
    ready: deque[RequestState] = deque()
    completed: list[RequestState] = []
    steps = []
    prefill_events = []
    now_ms = 0.0

    while pending or ready:
        if not ready and pending and now_ms < pending[0].arrival_ms:
            now_ms = pending[0].arrival_ms

        while pending and pending[0].arrival_ms <= now_ms:
            spec = pending.popleft()
            state = RequestState(spec=spec, blocks=request_blocks[spec.request_id])
            prefill_ms, event = run_prefill(
                resident,
                store,
                state.blocks,
                args.prefill_compute_ms,
                args.prefill_mode,
            )
            now_ms += prefill_ms
            event.update(
                {
                    "request_id": spec.request_id,
                    "arrival_ms": spec.arrival_ms,
                    "prefill_ms": prefill_ms,
                }
            )
            prefill_events.append(event)
            ready.append(state)

        if not ready:
            continue

        state = select_decode_request(ready, args.scheduler, rng)
        needed = select_blocks(
            state.blocks,
            state.decoded,
            args.blocks_per_step,
            args.access_pattern,
            rng,
        )
        hits = sum(1 for name in needed if resident.contains(name))
        missing = [name for name in needed if not resident.contains(name)]
        victims = resident.victims_for(missing)
        evict, prefetch, transfer_ms, compute_ms, step_ms = run_step_work(
            store,
            victims,
            missing,
            compute,
            args.overlap_compute,
        )
        now_ms += step_ms
        resident.add_many(needed)
        state.decoded += 1
        if state.first_token_ms is None:
            state.first_token_ms = now_ms
        if state.decoded >= state.spec.decode_steps:
            state.completed_ms = now_ms
            try:
                ready.remove(state)
            except ValueError:
                pass
            completed.append(state)

        steps.append(
            {
                "request_id": state.spec.request_id,
                "decode_step": state.decoded,
                "needed": needed,
                "hits": hits,
                "missing": missing,
                "victims": victims,
                "evict": evict,
                "prefetch": prefetch,
                "transfer_ms": transfer_ms,
                "compute_ms": compute_ms,
                "step_ms": step_ms,
            }
        )

    summary = summarize_workload(requests, completed, steps, prefill_events, now_ms)
    print(
        "mode",
        mode,
        "requests_per_second",
        summary["requests_per_second"],
        "tokens_per_second",
        summary["tokens_per_second"],
        "ttft_p50_ms",
        summary["ttft_ms_p50"],
        "decode_step_p50_ms",
        summary["decode_step_ms_p50"],
        "hit_rate",
        summary["gpu_cache_hit_rate"],
    )
    return {"mode": mode, "summary": summary, "steps": steps, "prefill": prefill_events}


def summarize_workload(
    requests: list[RequestSpec],
    completed: list[RequestState],
    steps: list[dict],
    prefill_events: list[dict],
    now_ms: float,
) -> dict:
    ttft = [
        state.first_token_ms - state.spec.arrival_ms
        for state in completed
        if state.first_token_ms is not None
    ]
    request_latency = [
        state.completed_ms - state.spec.arrival_ms
        for state in completed
        if state.completed_ms is not None
    ]
    decode_steps = [step["step_ms"] for step in steps]
    transfer = [step["transfer_ms"] for step in steps]
    compute = [step["compute_ms"] for step in steps]
    prefetch_batches = [step["prefetch"] for step in steps if step["prefetch"]["blocks"]]
    evict_batches = [step["evict"] for step in steps if step["evict"]["blocks"]]
    prefill_prefetch_batches = [
        event["prefetch"] for event in prefill_events if event["prefetch"]["blocks"]
    ]
    prefill_evict_batches = [
        event["evict"] for event in prefill_events if event["evict"]["blocks"]
    ]
    total_needed = sum(len(step["needed"]) for step in steps)
    total_hits = sum(step["hits"] for step in steps)
    total_tokens = len(steps)
    seconds = now_ms / 1000.0 if now_ms > 0.0 else 0.0

    return {
        "requests": len(requests),
        "completed_requests": len(completed),
        "tokens": total_tokens,
        "duration_ms": now_ms,
        "requests_per_second": len(completed) / seconds if seconds > 0.0 else 0.0,
        "tokens_per_second": total_tokens / seconds if seconds > 0.0 else 0.0,
        "ttft_ms_p50": statistics.median(ttft) if ttft else 0.0,
        "ttft_ms_p95": percentile(ttft, 95.0),
        "request_latency_ms_p50": statistics.median(request_latency) if request_latency else 0.0,
        "request_latency_ms_p95": percentile(request_latency, 95.0),
        "decode_step_ms_p50": statistics.median(decode_steps) if decode_steps else 0.0,
        "decode_step_ms_p95": percentile(decode_steps, 95.0),
        "transfer_ms_p50": statistics.median(transfer) if transfer else 0.0,
        "transfer_ms_p95": percentile(transfer, 95.0),
        "compute_ms_p50": statistics.median(compute) if compute else 0.0,
        "compute_ms_p95": percentile(compute, 95.0),
        "prefetch_blocks": sum(batch["blocks"] for batch in prefetch_batches),
        "evict_blocks": sum(batch["blocks"] for batch in evict_batches),
        "prefill_restore_blocks": sum(event["restored_blocks"] for event in prefill_events),
        "prefill_prefetch_blocks": sum(
            batch["blocks"] for batch in prefill_prefetch_batches
        ),
        "prefill_evict_blocks": sum(batch["blocks"] for batch in prefill_evict_batches),
        "prefetch_gib_per_second": summarize_transfer_gib(prefetch_batches),
        "evict_gib_per_second": summarize_transfer_gib(evict_batches),
        "direct_chunks": sum(
            step["prefetch"]["direct_chunks"] + step["evict"]["direct_chunks"]
            for step in steps
        )
        + sum(
            event["prefetch"]["direct_chunks"] + event["evict"]["direct_chunks"]
            for event in prefill_events
        ),
        "relay_chunks": sum(
            step["prefetch"]["relay_chunks"] + step["evict"]["relay_chunks"]
            for step in steps
        )
        + sum(
            event["prefetch"]["relay_chunks"] + event["evict"]["relay_chunks"]
            for event in prefill_events
        ),
        "gpu_cache_hit_rate": total_hits / total_needed if total_needed else 0.0,
        "prefill_count": len(prefill_events),
        "prefill_ms_p50": statistics.median(
            event["prefill_ms"] for event in prefill_events
        )
        if prefill_events
        else 0.0,
    }


def summarize_transfer_gib(batches: list[dict]) -> float:
    bytes_ = sum(batch["bytes"] for batch in batches)
    seconds = sum(batch["elapsed_ms"] for batch in batches) / 1000.0
    return (bytes_ / (1024**3)) / seconds if seconds > 0.0 else 0.0


def compact_summary(result: dict) -> str:
    config = result["config"]
    lines = [
        "COPY_SUMMARY_BEGIN",
        (
            "workload_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"preset={config['preset']} "
            f"arrival_pattern={config['arrival_pattern']} request_count={config['request_count']} "
            f"prompt_blocks={config['prompt_blocks_min']}..{config['prompt_blocks_max']} "
            f"decode_steps={config['decode_steps_min']}..{config['decode_steps_max']} "
            f"scheduler={config['scheduler']} access_pattern={config['access_pattern']} "
            f"blocks_per_step={config['blocks_per_step']} gpu_block_capacity={config['gpu_block_capacity']} "
            f"storage_layout={config['storage_layout']} block_bytes={config['block_bytes']} "
            f"compute_impl={config['compute_impl']} overlap_compute={config['overlap_compute']} "
            f"prefill_mode={config['prefill_mode']} "
            f"mode={config['mode']} dynamic_weights={config['dynamic_weights']}"
        ),
        (
            "workload_scenario "
            "type=prefill_decode "
            "policy=lru_eviction "
            "transfer=evict_then_prefetch_on_decode "
            f"prefill={config['prefill_mode']} "
            f"arrival_interval_ms={config['arrival_interval_ms']} "
            f"prefill_compute_ms={config['prefill_compute_ms']} "
            f"decode_compute_ms={config['compute_ms']} "
            f"cuda_compute_iterations={config['cuda_compute_iterations']}"
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
            "workload_mode "
            f"mode={mode} requests_s={summary['requests_per_second']:.3f} "
            f"tokens_s={summary['tokens_per_second']:.3f} "
            f"ttft_p50_ms={summary['ttft_ms_p50']:.3f} "
            f"ttft_p95_ms={summary['ttft_ms_p95']:.3f} "
            f"decode_p50_ms={summary['decode_step_ms_p50']:.3f} "
            f"decode_p95_ms={summary['decode_step_ms_p95']:.3f} "
            f"transfer_p50_ms={summary['transfer_ms_p50']:.3f} "
            f"compute_p50_ms={summary['compute_ms_p50']:.3f} "
            f"hit_rate={summary['gpu_cache_hit_rate']:.3f} "
            f"prefetch_blocks={summary['prefetch_blocks']} "
            f"evict_blocks={summary['evict_blocks']} "
            f"prefill_restore_blocks={summary['prefill_restore_blocks']} "
            f"prefill_evict_blocks={summary['prefill_evict_blocks']} "
            f"direct_chunks={summary['direct_chunks']} relay_chunks={summary['relay_chunks']}"
        )
    for key, value in result["speedups"].items():
        lines.append(f"workload_speedup {key}={value:.3f}")
    lines.append("COPY_SUMMARY_END")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus prefill/decode workload simulator")
    parser.add_argument("--preset", choices=["light", "pressure", "long_context"])
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument(
        "--arrival-pattern",
        choices=["burst", "fixed", "poisson"],
        default="burst",
    )
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0)
    parser.add_argument("--prompt-blocks-min", type=int, default=6)
    parser.add_argument("--prompt-blocks-max", type=int, default=10)
    parser.add_argument("--decode-steps-min", type=int, default=12)
    parser.add_argument("--decode-steps-max", type=int, default=20)
    parser.add_argument("--scheduler", choices=["round_robin", "oldest_first", "random_ready"], default="round_robin")
    parser.add_argument(
        "--access-pattern",
        choices=["round_robin", "sliding", "random"],
        default="sliding",
    )
    parser.add_argument("--blocks-per-step", type=int, default=4)
    parser.add_argument("--gpu-block-capacity", type=int, default=12)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--block-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--prefill-mode",
        choices=["produce_kv_on_gpu", "restore_from_cpu"],
        default="produce_kv_on_gpu",
    )
    parser.add_argument("--prefill-compute-ms", type=float, default=0.0)
    parser.add_argument("--compute-ms", type=float, default=0.0)
    parser.add_argument("--compute-impl", choices=["sleep", "cuda"], default="cuda")
    parser.add_argument("--cuda-compute-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--cuda-compute-iterations", type=int, default=2048)
    parser.add_argument("--overlap-compute", action="store_true")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mode", choices=["pool", "direct", "relay", "all"], default="pool")
    parser.add_argument("--dynamic-weights", action="store_true")
    parser.add_argument("--dynamic-weight-alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    args = parser.parse_args()

    apply_preset(args, parser)
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
    requests = generate_requests(args)

    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": relays,
            "preset": args.preset or "custom",
            "request_count": args.request_count,
            "arrival_pattern": args.arrival_pattern,
            "arrival_interval_ms": args.arrival_interval_ms,
            "prompt_blocks_min": args.prompt_blocks_min,
            "prompt_blocks_max": args.prompt_blocks_max,
            "decode_steps_min": args.decode_steps_min,
            "decode_steps_max": args.decode_steps_max,
            "scheduler": args.scheduler,
            "access_pattern": args.access_pattern,
            "blocks_per_step": args.blocks_per_step,
            "gpu_block_capacity": args.gpu_block_capacity,
            "storage_layout": args.storage_layout,
            "block_bytes": args.block_bytes,
            "prefill_mode": args.prefill_mode,
            "prefill_compute_ms": args.prefill_compute_ms,
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
            "seed": args.seed,
        },
        "profile": profile_to_dict(profile),
        "requests": [request.__dict__ for request in requests],
        "modes": {},
        "speedups": {},
    }

    modes = ["direct", "relay", "pool"] if args.mode == "all" else [args.mode]
    for mode in modes:
        store, request_blocks = create_store(args, runtime, requests)
        result["modes"][mode] = run_mode(
            runtime,
            store,
            requests,
            request_blocks,
            mode,
            args,
        )

    if args.mode == "all":
        direct = result["modes"]["direct"]["summary"]
        relay = result["modes"]["relay"]["summary"]
        pool = result["modes"]["pool"]["summary"]
        if direct["tokens_per_second"] > 0.0:
            result["speedups"]["pool_over_direct_tokens_per_second"] = (
                pool["tokens_per_second"] / direct["tokens_per_second"]
            )
        if relay["tokens_per_second"] > 0.0:
            result["speedups"]["pool_over_relay_tokens_per_second"] = (
                pool["tokens_per_second"] / relay["tokens_per_second"]
            )
        if direct["ttft_ms_p50"] > 0.0:
            result["speedups"]["direct_over_pool_ttft_p50"] = (
                direct["ttft_ms_p50"] / pool["ttft_ms_p50"]
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


def validate_args(args) -> None:
    if args.request_count <= 0:
        raise ValueError("--request-count must be positive")
    if args.prompt_blocks_min <= 0 or args.prompt_blocks_max < args.prompt_blocks_min:
        raise ValueError("--prompt-blocks-min/max are invalid")
    if args.decode_steps_min <= 0 or args.decode_steps_max < args.decode_steps_min:
        raise ValueError("--decode-steps-min/max are invalid")
    if args.blocks_per_step <= 0:
        raise ValueError("--blocks-per-step must be positive")
    if args.blocks_per_step > args.prompt_blocks_min:
        raise ValueError("--blocks-per-step must be at most --prompt-blocks-min")
    if args.gpu_block_capacity < args.blocks_per_step:
        raise ValueError("--gpu-block-capacity must be at least --blocks-per-step")
    if args.arrival_interval_ms < 0.0:
        raise ValueError("--arrival-interval-ms must be non-negative")
    if args.cuda_compute_elements <= 0:
        raise ValueError("--cuda-compute-elements must be positive")
    if args.cuda_compute_iterations <= 0:
        raise ValueError("--cuda-compute-iterations must be positive")


def apply_preset(args, parser: argparse.ArgumentParser) -> None:
    if args.preset is None:
        return
    preset = PRESETS[args.preset]
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest is not argparse.SUPPRESS
    }
    for name, value in preset.items():
        if getattr(args, name) == defaults[name]:
            setattr(args, name, value)


if __name__ == "__main__":
    main()
