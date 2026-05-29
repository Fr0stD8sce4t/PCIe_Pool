from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turbobus import TurboBusClient, WorkloadKind
from daemon_support import (
    add_daemon_options,
    benchmark_job_id,
    make_benchmark_transfer_intent,
    receipt_to_trace,
    receipt_trace_line,
)


def bucket_indices(bucket_count: int, active_buckets: int, iteration: int) -> tuple[int, ...]:
    start = (int(iteration) * int(active_buckets)) % int(bucket_count)
    return tuple(
        (start + offset) % int(bucket_count)
        for offset in range(int(active_buckets))
    )


def bucket_ranges(
    bucket_count: int,
    active_buckets: int,
    bucket_bytes: int,
    iteration: int,
) -> tuple[dict[str, int], ...]:
    ranges = []
    for index in bucket_indices(bucket_count, active_buckets, iteration):
        offset = int(index) * int(bucket_bytes)
        ranges.append(
            {
                "src_offset": offset,
                "dst_offset": offset,
                "bytes": int(bucket_bytes),
            }
        )
    return tuple(ranges)


def workload_kind(value: str) -> WorkloadKind:
    return WorkloadKind(str(value))


def active_bucket_count(args) -> int:
    return int(args.bucket_count if args.active_buckets is None else args.active_buckets)


def build_training_intent(args, *, iteration: int, phase: str, operation: str):
    active_buckets = active_bucket_count(args)
    ranges = bucket_ranges(
        args.bucket_count,
        active_buckets,
        args.bucket_bytes,
        iteration,
    )
    if operation == "prefetch":
        direction = "h2d"
        source_buffer_id = args.cpu_buffer_id
        destination_buffer_id = args.gpu_buffer_id
    elif operation == "offload":
        direction = "d2h"
        source_buffer_id = args.gpu_buffer_id
        destination_buffer_id = args.cpu_buffer_id
    else:
        raise ValueError("operation must be prefetch or offload")
    return make_benchmark_transfer_intent(
        intent_id=f"{args.intent_prefix}-{args.run_id}-{phase}-{iteration}-{operation}",
        workload_kind=workload_kind(args.workload_kind),
        job_id=args.job_id,
        session_id=args.session_id,
        source_buffer_id=source_buffer_id,
        destination_buffer_id=destination_buffer_id,
        direction=direction,
        total_bytes=sum(item["bytes"] for item in ranges),
        ranges=ranges,
        policy_hints={},
        metadata={
            "benchmark": "training-offload",
            "phase": phase,
            "iteration": int(iteration),
            "operation": operation,
            "policy": args.policy,
            "storage_layout": args.storage_layout,
            "bucket_count": int(args.bucket_count),
            "active_buckets": active_buckets,
            "bucket_bytes": int(args.bucket_bytes),
            "chunk_bytes": int(args.chunk_bytes),
            "bucket_indices": list(bucket_indices(args.bucket_count, active_buckets, iteration)),
        },
    )


def submit_training_intent(
    client: TurboBusClient,
    args,
    *,
    iteration: int,
    phase: str,
    operation: str,
) -> dict:
    intent = build_training_intent(
        args,
        iteration=iteration,
        phase=phase,
        operation=operation,
    )
    start = time.perf_counter()
    receipt = client.submit_transfer_intent(intent)
    if args.wait_timeout_seconds is not None:
        receipt = client.wait_transfer_receipt(
            intent.intent_id,
            timeout_seconds=args.wait_timeout_seconds,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    trace = receipt_to_trace(receipt)
    return {
        "operation": operation,
        "transfer_ms": elapsed_ms,
        "intent": {
            "intent_id": intent.intent_id,
            "job_id": intent.job_id,
            "session_id": intent.session_id,
            "source_buffer_id": intent.source_buffer_id,
            "destination_buffer_id": intent.destination_buffer_id,
            "direction": intent.direction,
            "workload_kind": intent.workload_kind.value,
            "total_bytes": intent.total_bytes,
            "ranges": list(intent.ranges),
            "policy_hints": dict(intent.policy_hints),
            "metadata": dict(intent.metadata),
        },
        "receipt": trace,
        "bytes": int(trace["bytes_total"]),
        "bytes_completed": int(trace["bytes_completed"]),
        "direct_bytes": int(trace["direct_bytes"]),
        "relay_bytes": int(trace["relay_bytes"]),
        "direct_chunks": int(trace["direct_chunks"]),
        "relay_chunks": int(trace["relay_chunks"]),
        "decision_id": str(trace["decision_id"]),
        "topology_snapshot_id": str(trace["topology_snapshot_id"]),
        "ticket_id": str(trace["ticket_id"]),
        "fallback_reason": str(trace.get("fallback_reason", "") or ""),
    }


def run_compute_delay(delay_ms: float) -> float:
    delay_ms = float(delay_ms)
    if delay_ms <= 0.0:
        return 0.0
    start = time.perf_counter()
    time.sleep(delay_ms / 1000.0)
    return (time.perf_counter() - start) * 1000.0


def run_iteration(
    client: TurboBusClient,
    args,
    *,
    iteration: int,
    phase: str,
) -> dict:
    iteration_start = time.perf_counter()
    prefetch = submit_training_intent(
        client,
        args,
        iteration=iteration,
        phase=phase,
        operation="prefetch",
    )
    compute_ms = run_compute_delay(args.compute_delay_ms)
    offload = submit_training_intent(
        client,
        args,
        iteration=iteration,
        phase=phase,
        operation="offload",
    )
    iteration_ms = (time.perf_counter() - iteration_start) * 1000.0
    transfer_ms = prefetch["transfer_ms"] + offload["transfer_ms"]
    return {
        "iteration": int(iteration),
        "phase": phase,
        "active_buckets": active_bucket_count(args),
        "iteration_ms": iteration_ms,
        "transfer_ms": transfer_ms,
        "compute_ms": compute_ms,
        "prefetch": prefetch,
        "offload": offload,
    }


def run_warmup(client: TurboBusClient, args) -> list[dict]:
    return [
        run_iteration(client, args, iteration=iteration, phase="warmup")
        for iteration in range(int(args.warmup))
    ]


def run_benchmark(args, *, client: TurboBusClient | None = None) -> dict:
    if client is None:
        client = TurboBusClient(socket_path=args.daemon_socket_path)
    warmup_samples = run_warmup(client, args)
    samples = [
        run_iteration(client, args, iteration=iteration, phase="measure")
        for iteration in range(int(args.iterations))
    ]
    return {
        "config": config_dict(args),
        "warmup_samples": warmup_samples,
        "samples": samples,
        "summary": summarize_training(samples),
    }


def summarize_training(samples: list[dict]) -> dict:
    if not samples:
        return {
            "iterations": 0,
            "median_iteration_ms": 0.0,
            "median_transfer_ms": 0.0,
            "median_compute_ms": 0.0,
            "median_gib_per_second": 0.0,
            "prefetch": empty_transfer_summary(),
            "offload": empty_transfer_summary(),
        }
    total_bytes = [
        sample["prefetch"]["bytes"] + sample["offload"]["bytes"]
        for sample in samples
    ]
    transfer_ms = [sample["transfer_ms"] for sample in samples]
    return {
        "iterations": len(samples),
        "median_iteration_ms": statistics.median(
            sample["iteration_ms"] for sample in samples
        ),
        "median_transfer_ms": statistics.median(transfer_ms),
        "median_compute_ms": statistics.median(
            sample["compute_ms"] for sample in samples
        ),
        "median_gib_per_second": statistics.median(
            (bytes_ / (1024**3)) / (ms / 1000.0) if ms > 0.0 else 0.0
            for bytes_, ms in zip(total_bytes, transfer_ms, strict=False)
        ),
        "prefetch": summarize_transfer_side(samples, "prefetch"),
        "offload": summarize_transfer_side(samples, "offload"),
    }


def empty_transfer_summary() -> dict:
    return {
        "median_transfer_ms": 0.0,
        "bytes": 0,
        "bytes_completed": 0,
        "direct_bytes": 0,
        "relay_bytes": 0,
        "direct_chunks": 0,
        "relay_chunks": 0,
        "decision_ids": [],
        "topology_snapshot_ids": [],
        "ticket_ids": [],
        "receipt_ids": [],
        "fallback_reasons": [],
    }


def summarize_transfer_side(samples: list[dict], operation: str) -> dict:
    return {
        "median_transfer_ms": statistics.median(
            sample[operation]["transfer_ms"] for sample in samples
        ),
        "bytes": int(statistics.median(sample[operation]["bytes"] for sample in samples)),
        "bytes_completed": int(
            statistics.median(sample[operation]["bytes_completed"] for sample in samples)
        ),
        "direct_bytes": int(
            statistics.median(sample[operation]["direct_bytes"] for sample in samples)
        ),
        "relay_bytes": int(
            statistics.median(sample[operation]["relay_bytes"] for sample in samples)
        ),
        "direct_chunks": int(
            statistics.median(sample[operation]["direct_chunks"] for sample in samples)
        ),
        "relay_chunks": int(
            statistics.median(sample[operation]["relay_chunks"] for sample in samples)
        ),
        "decision_ids": sorted({sample[operation]["decision_id"] for sample in samples}),
        "topology_snapshot_ids": sorted(
            {sample[operation]["topology_snapshot_id"] for sample in samples}
        ),
        "ticket_ids": sorted({sample[operation]["ticket_id"] for sample in samples}),
        "receipt_ids": sorted(
            {
                str(sample[operation]["receipt"].get("receipt_id", ""))
                for sample in samples
                if sample[operation]["receipt"].get("receipt_id")
            }
        ),
        "fallback_reasons": sorted(
            {
                sample[operation]["fallback_reason"]
                for sample in samples
                if sample[operation]["fallback_reason"]
            }
        ),
    }


def config_dict(args) -> dict[str, object]:
    return {
        "session_id": args.session_id,
        "job_id": args.job_id,
        "cpu_buffer_id": args.cpu_buffer_id,
        "gpu_buffer_id": args.gpu_buffer_id,
        "workload_kind": args.workload_kind,
        "bucket_count": int(args.bucket_count),
        "active_buckets": active_bucket_count(args),
        "bucket_bytes": int(args.bucket_bytes),
        "storage_layout": args.storage_layout,
        "chunk_bytes": int(args.chunk_bytes),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "compute_delay_ms": float(args.compute_delay_ms),
        "policy": args.policy,
        "run_id": args.run_id,
        "daemon_socket_path": args.daemon_socket_path,
        "wait_timeout_seconds": args.wait_timeout_seconds,
    }


def write_json(path: str, result: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_text(path: str, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def compact_summary(result: dict) -> str:
    config = result["config"]
    summary = result["summary"]
    lines = [
        "TRAINING_OFFLOAD_SUMMARY_BEGIN",
        (
            "training_config "
            f"session_id={config['session_id']} job_id={config['job_id']} "
            f"cpu_buffer_id={config['cpu_buffer_id']} "
            f"gpu_buffer_id={config['gpu_buffer_id']} "
            f"workload_kind={config['workload_kind']} "
            f"bucket_count={config['bucket_count']} "
            f"active_buckets={config['active_buckets']} "
            f"bucket_bytes={config['bucket_bytes']} "
            f"storage_layout={config['storage_layout']} "
            f"chunk_bytes={config['chunk_bytes']} "
            f"iterations={config['iterations']} policy={config['policy']} "
            f"daemon_socket_path={config['daemon_socket_path']}"
        ),
        (
            "training_summary "
            f"median_iteration_ms={summary['median_iteration_ms']:.3f} "
            f"median_transfer_ms={summary['median_transfer_ms']:.3f} "
            f"median_compute_ms={summary['median_compute_ms']:.3f} "
            f"median_gib_s={summary['median_gib_per_second']:.3f}"
        ),
    ]
    for operation in ("prefetch", "offload"):
        side = summary[operation]
        lines.append(
            "training_transfer "
            f"op={operation} median_ms={side['median_transfer_ms']:.3f} "
            f"bytes={side['bytes']} bytes_completed={side['bytes_completed']} "
            f"direct_chunks={side['direct_chunks']} "
            f"relay_chunks={side['relay_chunks']} "
            f"direct_bytes={side['direct_bytes']} relay_bytes={side['relay_bytes']}"
        )
    for sample in result["samples"]:
        lines.append(
            "training_sample "
            f"iteration={sample['iteration']} "
            f"iteration_ms={sample['iteration_ms']:.3f} "
            f"transfer_ms={sample['transfer_ms']:.3f} "
            f"compute_ms={sample['compute_ms']:.3f} "
            f"prefetch_decision_id={sample['prefetch']['decision_id']} "
            f"prefetch_topology_snapshot_id={sample['prefetch']['topology_snapshot_id']} "
            f"prefetch_ticket_id={sample['prefetch']['ticket_id']} "
            f"offload_decision_id={sample['offload']['decision_id']} "
            f"offload_topology_snapshot_id={sample['offload']['topology_snapshot_id']} "
            f"offload_ticket_id={sample['offload']['ticket_id']}"
        )
        lines.append(
            receipt_trace_line(
                _receipt_from_trace(sample["prefetch"]["receipt"]),
                prefix="training_prefetch_receipt",
            )
        )
        lines.append(
            receipt_trace_line(
                _receipt_from_trace(sample["offload"]["receipt"]),
                prefix="training_offload_receipt",
            )
        )
    lines.append("TRAINING_OFFLOAD_SUMMARY_END")
    return "\n".join(lines)


def _receipt_from_trace(trace: dict):
    from turbobus import TransferReceipt

    return TransferReceipt(
        receipt_id=trace["receipt_id"],
        ticket_id=trace["ticket_id"],
        intent_id=trace["intent_id"],
        decision_id=trace["decision_id"],
        topology_snapshot_id=trace["topology_snapshot_id"],
        job_id=trace["job_id"],
        session_id=trace["session_id"],
        state=trace["state"],
        bytes_total=trace["bytes_total"],
        bytes_completed=trace["bytes_completed"],
        started_at=trace.get("started_at", 0.0),
        completed_at=trace.get("completed_at"),
        path_stats=tuple(trace.get("path_stats", ())),
        error=trace.get("error"),
        metadata=trace.get("metadata", {}),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit training state offload intent through the public TurboBus client"
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--job-id", default=benchmark_job_id("training-offload"))
    parser.add_argument("--cpu-buffer-id", required=True)
    parser.add_argument("--gpu-buffer-id", required=True)
    parser.add_argument(
        "--workload-kind",
        choices=[WorkloadKind.TRAINING_STATE.value, WorkloadKind.OPTIMIZER_STATE.value],
        default=WorkloadKind.TRAINING_STATE.value,
    )
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--active-buckets", type=int)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["packed", "separate"], default="packed")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compute-delay-ms", type=float, default=0.0)
    parser.add_argument("--policy", default="daemon-default")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--intent-prefix", default="training-offload")
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--json-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    add_daemon_options(parser)
    return parser


def validate_args(args) -> None:
    if not args.daemon_socket_path:
        raise ValueError("--daemon-socket-path is required")
    if args.bucket_count <= 0:
        raise ValueError("--bucket-count must be positive")
    if args.bucket_bytes <= 0:
        raise ValueError("--bucket-bytes must be positive")
    if args.chunk_bytes <= 0:
        raise ValueError("--chunk-bytes must be positive")
    active_buckets = active_bucket_count(args)
    if active_buckets <= 0 or active_buckets > args.bucket_count:
        raise ValueError("--active-buckets must be between 1 and --bucket-count")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.compute_delay_ms < 0.0:
        raise ValueError("--compute-delay-ms must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    result = run_benchmark(args)
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
