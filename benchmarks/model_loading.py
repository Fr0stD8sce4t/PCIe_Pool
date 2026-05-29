from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import uuid

from turbobus import TurboBusClient, WorkloadKind
from daemon_support import (
    add_daemon_options,
    benchmark_job_id,
    make_benchmark_transfer_intent,
    receipt_to_trace,
    receipt_trace_line,
)


def bucket_ranges(bucket_count: int, bucket_bytes: int) -> tuple[dict[str, int], ...]:
    ranges = []
    for index in range(int(bucket_count)):
        offset = index * int(bucket_bytes)
        ranges.append(
            {
                "src_offset": offset,
                "dst_offset": offset,
                "bytes": int(bucket_bytes),
            }
        )
    return tuple(ranges)


def build_model_loading_intent(args, *, iteration: int, phase: str):
    total_bytes = int(args.bucket_count) * int(args.bucket_bytes)
    return make_benchmark_transfer_intent(
        intent_id=f"{args.intent_prefix}-{args.run_id}-{phase}-{iteration}",
        workload_kind=WorkloadKind.MODEL_WEIGHTS,
        job_id=args.job_id,
        session_id=args.session_id,
        source_buffer_id=args.source_buffer_id,
        destination_buffer_id=args.destination_buffer_id,
        direction="h2d",
        total_bytes=total_bytes,
        ranges=bucket_ranges(args.bucket_count, args.bucket_bytes),
        policy_hints={},
        metadata={
            "benchmark": "model-loading",
            "phase": phase,
            "iteration": int(iteration),
            "policy": args.policy,
            "storage_layout": args.storage_layout,
            "bucket_count": int(args.bucket_count),
            "bucket_bytes": int(args.bucket_bytes),
            "chunk_bytes": int(args.chunk_bytes),
        },
    )


def submit_load_intent(client: TurboBusClient, args, *, iteration: int, phase: str) -> dict:
    intent = build_model_loading_intent(args, iteration=iteration, phase=phase)
    start = time.perf_counter()
    receipt = client.submit_transfer_intent(intent)
    if args.wait_timeout_seconds is not None:
        receipt = client.wait_transfer_receipt(
            intent.intent_id,
            timeout_seconds=args.wait_timeout_seconds,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    trace = receipt_to_trace(receipt)
    gib_per_second = (
        (int(trace["bytes_total"]) / (1024**3)) / (elapsed_ms / 1000.0)
        if elapsed_ms > 0.0
        else 0.0
    )
    return {
        "iteration": int(iteration),
        "phase": phase,
        "load_ms": elapsed_ms,
        "load_gib_per_second": gib_per_second,
        "intent": {
            "intent_id": intent.intent_id,
            "job_id": intent.job_id,
            "session_id": intent.session_id,
            "source_buffer_id": intent.source_buffer_id,
            "destination_buffer_id": intent.destination_buffer_id,
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


def run_warmup(client: TurboBusClient, args) -> list[dict]:
    samples = []
    for iteration in range(int(args.warmup)):
        samples.append(
            submit_load_intent(
                client,
                args,
                iteration=iteration,
                phase="warmup",
            )
        )
    return samples


def run_benchmark(args, *, client: TurboBusClient | None = None) -> dict:
    if client is None:
        client = TurboBusClient(socket_path=args.daemon_socket_path)
    warmup_samples = run_warmup(client, args)
    samples = [
        submit_load_intent(client, args, iteration=iteration, phase="measure")
        for iteration in range(int(args.iterations))
    ]
    return {
        "config": config_dict(args),
        "warmup_samples": warmup_samples,
        "samples": samples,
        "summary": summarize_load(samples),
    }


def summarize_load(samples: list[dict]) -> dict:
    if not samples:
        return {
            "iterations": 0,
            "median_load_ms": 0.0,
            "median_gib_per_second": 0.0,
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
    return {
        "iterations": len(samples),
        "median_load_ms": statistics.median(sample["load_ms"] for sample in samples),
        "median_gib_per_second": statistics.median(
            sample["load_gib_per_second"] for sample in samples
        ),
        "bytes": int(statistics.median(sample["bytes"] for sample in samples)),
        "bytes_completed": int(
            statistics.median(sample["bytes_completed"] for sample in samples)
        ),
        "direct_bytes": int(
            statistics.median(sample["direct_bytes"] for sample in samples)
        ),
        "relay_bytes": int(statistics.median(sample["relay_bytes"] for sample in samples)),
        "direct_chunks": int(
            statistics.median(sample["direct_chunks"] for sample in samples)
        ),
        "relay_chunks": int(
            statistics.median(sample["relay_chunks"] for sample in samples)
        ),
        "decision_ids": sorted({sample["decision_id"] for sample in samples}),
        "topology_snapshot_ids": sorted(
            {sample["topology_snapshot_id"] for sample in samples}
        ),
        "ticket_ids": sorted({sample["ticket_id"] for sample in samples}),
        "receipt_ids": sorted(
            {
                str(sample["receipt"].get("receipt_id", ""))
                for sample in samples
                if sample["receipt"].get("receipt_id")
            }
        ),
        "fallback_reasons": sorted(
            {sample["fallback_reason"] for sample in samples if sample["fallback_reason"]}
        ),
    }


def config_dict(args) -> dict[str, object]:
    return {
        "session_id": args.session_id,
        "job_id": args.job_id,
        "source_buffer_id": args.source_buffer_id,
        "destination_buffer_id": args.destination_buffer_id,
        "workload_kind": WorkloadKind.MODEL_WEIGHTS.value,
        "bucket_count": int(args.bucket_count),
        "bucket_bytes": int(args.bucket_bytes),
        "storage_layout": args.storage_layout,
        "chunk_bytes": int(args.chunk_bytes),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
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
        "MODEL_LOAD_SUMMARY_BEGIN",
        (
            "model_load_config "
            f"session_id={config['session_id']} job_id={config['job_id']} "
            f"source_buffer_id={config['source_buffer_id']} "
            f"destination_buffer_id={config['destination_buffer_id']} "
            f"bucket_count={config['bucket_count']} "
            f"bucket_bytes={config['bucket_bytes']} "
            f"storage_layout={config['storage_layout']} "
            f"chunk_bytes={config['chunk_bytes']} "
            f"iterations={config['iterations']} policy={config['policy']} "
            f"daemon_socket_path={config['daemon_socket_path']}"
        ),
        (
            "model_load_summary "
            f"median_load_ms={summary['median_load_ms']:.3f} "
            f"median_gib_s={summary['median_gib_per_second']:.3f} "
            f"bytes={summary['bytes']} "
            f"bytes_completed={summary['bytes_completed']} "
            f"direct_bytes={summary['direct_bytes']} "
            f"relay_bytes={summary['relay_bytes']} "
            f"direct_chunks={summary['direct_chunks']} "
            f"relay_chunks={summary['relay_chunks']}"
        ),
    ]
    for sample in result["samples"]:
        lines.append(
            "model_load_sample "
            f"iteration={sample['iteration']} "
            f"load_ms={sample['load_ms']:.3f} "
            f"gib_s={sample['load_gib_per_second']:.3f} "
            f"decision_id={sample['decision_id']} "
            f"topology_snapshot_id={sample['topology_snapshot_id']} "
            f"ticket_id={sample['ticket_id']}"
        )
        lines.append(
            receipt_trace_line(
                _receipt_from_trace(sample["receipt"]),
                prefix="model_load_receipt",
            )
        )
    lines.append("MODEL_LOAD_SUMMARY_END")
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
        description="Submit model weight loading intent through the public TurboBus client"
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--job-id", default=benchmark_job_id("model-loading"))
    parser.add_argument("--source-buffer-id", required=True)
    parser.add_argument("--destination-buffer-id", required=True)
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["packed", "separate"], default="packed")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--policy", default="daemon-default")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--intent-prefix", default="model-load")
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
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")


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
