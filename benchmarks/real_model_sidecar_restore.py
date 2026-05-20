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
        parse_relay_gpus,
        percentile,
        profile_to_dict,
        write_json,
        write_text,
    )
    from prefix_restore_poc import (
        create_store,
        restore_batch,
        restore_window,
        verify_blocks,
    )


class TorchModelCompute:
    def __init__(
        self,
        target_gpu: int,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        heads: int,
        ff_size: int,
        layers: int,
        iterations: int,
        dtype: str,
    ) -> None:
        self.device = torch.device(f"cuda:{target_gpu}")
        self.iterations = iterations
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=heads,
                    dim_feedforward=ff_size,
                    batch_first=True,
                    dropout=0.0,
                    activation="gelu",
                    device=self.device,
                    dtype=model_dtype(dtype),
                ).eval()
                for _ in range(layers)
            ]
        )
        self.input = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            device=self.device,
            dtype=model_dtype(dtype),
        )
        self.stream = torch.cuda.Stream(device=self.device)

    @property
    def enabled(self) -> bool:
        return self.iterations > 0 and len(self.layers) > 0

    def run(self) -> float:
        if not self.enabled:
            return 0.0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            start.record(self.stream)
            output = self.input
            for _ in range(self.iterations):
                for layer in self.layers:
                    output = layer(output)
            self.input = output
            end.record(self.stream)
        end.synchronize()
        return float(start.elapsed_time(end))


def model_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown model dtype: {name}")


def run_sidecar_step(store, names: list[str], compute: TorchModelCompute, overlap: bool) -> dict:
    start = time.perf_counter()
    compute_ms = 0.0
    if overlap and compute.enabled:
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
    compute = TorchModelCompute(
        args.target_gpu,
        args.model_batch_size,
        args.model_seq_len,
        args.model_hidden_size,
        args.model_heads,
        args.model_ff_size,
        args.model_layers,
        args.model_iterations,
        args.model_dtype,
    )
    warmup_model(compute, args.model_warmup)

    steps = []
    verified = None
    for iteration in range(args.iterations):
        selected = restore_window(names, iteration * args.restore_blocks, args.restore_blocks)
        steps.append(run_sidecar_step(store, selected, compute, args.overlap_compute))

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
        "model_ms_p50",
        summary["model_ms_p50"],
        "restored_blocks",
        summary["restored_blocks"],
    )
    return {
        "mode": mode,
        "summary": summary,
        "steps": steps,
        "verified": verified,
    }


def warmup_model(compute: TorchModelCompute, warmup: int) -> None:
    for _ in range(warmup):
        compute.run()


def summarize_steps(steps: list[dict]) -> dict:
    restore_ms = [step["restore"]["elapsed_ms"] for step in steps]
    step_ms = [step["step_ms"] for step in steps]
    model_ms = [step["compute_ms"] for step in steps]
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
        "model_ms_p50": statistics.median(model_ms) if model_ms else 0.0,
        "model_ms_p95": percentile(model_ms, 95.0),
        "direct_bytes": sum(step["restore"]["direct_bytes"] for step in steps),
        "relay_bytes": sum(step["restore"]["relay_bytes"] for step in steps),
        "direct_chunks": sum(step["restore"]["direct_chunks"] for step in steps),
        "relay_chunks": sum(step["restore"]["relay_chunks"] for step in steps),
    }


def compact_summary(result: dict) -> str:
    config = result["config"]
    lines = [
        "COPY_SUMMARY_BEGIN",
        (
            "sidecar_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"sessions={config['sessions']} blocks_per_session={config['blocks_per_session']} "
            f"restore_blocks={config['restore_blocks']} iterations={config['iterations']} "
            f"storage_layout={config['storage_layout']} block_bytes={config['block_bytes']} "
            f"model_layers={config['model_layers']} model_batch_size={config['model_batch_size']} "
            f"model_seq_len={config['model_seq_len']} model_hidden_size={config['model_hidden_size']} "
            f"model_heads={config['model_heads']} model_ff_size={config['model_ff_size']} "
            f"model_iterations={config['model_iterations']} model_dtype={config['model_dtype']} "
            f"overlap_compute={config['overlap_compute']} "
            f"mode={config['mode']} dynamic_weights={config['dynamic_weights']}"
        ),
        (
            "sidecar_scenario "
            "type=real_torch_model_sidecar_restore "
            "boundary=framework_adjacent "
            "transfer=cpu_pinned_prefix_kv_to_gpu_slots "
            "model=torch_transformer_encoder_layer "
            "policy=no_scheduler_rewrite "
            f"verify={config['verify']} "
            "note=real_model_compute_not_full_inference_framework"
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
            "sidecar_mode "
            f"mode={mode} restore_gib_s={summary['restore_gib_per_second']:.3f} "
            f"restore_p50_ms={summary['restore_ms_p50']:.3f} "
            f"restore_p95_ms={summary['restore_ms_p95']:.3f} "
            f"step_p50_ms={summary['step_ms_p50']:.3f} "
            f"step_p95_ms={summary['step_ms_p95']:.3f} "
            f"model_p50_ms={summary['model_ms_p50']:.3f} "
            f"model_p95_ms={summary['model_ms_p95']:.3f} "
            f"restored_blocks={summary['restored_blocks']} "
            f"direct_chunks={summary['direct_chunks']} relay_chunks={summary['relay_chunks']}"
        )
        if mode_result["verified"] is not None:
            line += f" verified={mode_result['verified']}"
        lines.append(line)
    for key, value in result["speedups"].items():
        lines.append(f"sidecar_speedup {key}={value:.3f}")
    lines.append("COPY_SUMMARY_END")
    return "\n".join(lines)


def main() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to run the real model sidecar restore benchmark"
        ) from _TORCH_IMPORT_ERROR

    parser = argparse.ArgumentParser(description="TurboBus real model sidecar restore benchmark")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--blocks-per-session", type=int, default=8)
    parser.add_argument("--restore-blocks", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--block-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--model-layers", type=int, default=1)
    parser.add_argument("--model-batch-size", type=int, default=1)
    parser.add_argument("--model-seq-len", type=int, default=128)
    parser.add_argument("--model-hidden-size", type=int, default=4096)
    parser.add_argument("--model-heads", type=int, default=32)
    parser.add_argument("--model-ff-size", type=int, default=11008)
    parser.add_argument("--model-iterations", type=int, default=1)
    parser.add_argument("--model-warmup", type=int, default=1)
    parser.add_argument(
        "--model-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
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
            "model_layers": args.model_layers,
            "model_batch_size": args.model_batch_size,
            "model_seq_len": args.model_seq_len,
            "model_hidden_size": args.model_hidden_size,
            "model_heads": args.model_heads,
            "model_ff_size": args.model_ff_size,
            "model_iterations": args.model_iterations,
            "model_warmup": args.model_warmup,
            "model_dtype": args.model_dtype,
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
        direct = result["modes"]["direct"]["summary"]
        relay = result["modes"]["relay"]["summary"]
        pool = result["modes"]["pool"]["summary"]
        if direct["restore_gib_per_second"] > 0.0:
            result["speedups"]["pool_over_direct_restore"] = (
                pool["restore_gib_per_second"] / direct["restore_gib_per_second"]
            )
        if relay["restore_gib_per_second"] > 0.0:
            result["speedups"]["pool_over_relay_restore"] = (
                pool["restore_gib_per_second"] / relay["restore_gib_per_second"]
            )
        if pool["step_ms_p50"] > 0.0:
            result["speedups"]["direct_over_pool_step_p50"] = (
                direct["step_ms_p50"] / pool["step_ms_p50"]
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
    if args.model_layers <= 0:
        raise ValueError("--model-layers must be positive")
    if args.model_batch_size <= 0:
        raise ValueError("--model-batch-size must be positive")
    if args.model_seq_len <= 0:
        raise ValueError("--model-seq-len must be positive")
    if args.model_hidden_size <= 0:
        raise ValueError("--model-hidden-size must be positive")
    if args.model_heads <= 0 or args.model_hidden_size % args.model_heads != 0:
        raise ValueError("--model-heads must divide --model-hidden-size")
    if args.model_ff_size <= 0:
        raise ValueError("--model-ff-size must be positive")
    if args.model_iterations <= 0:
        raise ValueError("--model-iterations must be positive")
    if args.model_warmup < 0:
        raise ValueError("--model-warmup must be non-negative")


if __name__ == "__main__":
    main()
