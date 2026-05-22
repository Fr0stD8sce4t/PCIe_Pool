from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from daemon_support import add_daemon_options


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"
EXAMPLES = REPO_ROOT / "examples"

WORKLOADS = ("model-loading", "vllm-kv", "training-offload")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def selected_workloads(value: str) -> list[str]:
    items = parse_csv(value)
    if not items or items == ["all"]:
        return list(WORKLOADS)
    unknown = [item for item in items if item not in WORKLOADS]
    if unknown:
        raise ValueError(f"unknown workloads: {unknown}")
    return items


def mode_list(mode: str) -> str:
    if mode == "all":
        return "auto,direct,relay,pool"
    return mode


def output_paths(output_dir: Path, workload: str) -> dict[str, Path]:
    safe = workload.replace("-", "_")
    return {
        "json": output_dir / f"{safe}.json",
        "summary": output_dir / f"{safe}_summary.txt",
        "cases_json": output_dir / f"{safe}_cases.json",
        "cases_csv": output_dir / f"{safe}_cases.csv",
        "log_dir": output_dir / f"{safe}_logs",
    }


def daemon_command_args(args) -> list[str]:
    command = [
        "--daemon-max-inflight-chunks",
        str(args.daemon_max_inflight_chunks),
        "--daemon-profile-max-age-seconds",
        str(args.daemon_profile_max_age_seconds),
    ]
    if args.daemon_socket_path:
        command.extend(["--daemon-socket-path", args.daemon_socket_path])
    return command


def build_model_loading_command(args, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARKS / "model_loading.py"),
        "--target-gpu",
        str(args.target_gpu),
        "--relay-gpus",
        args.relay_gpus,
        "--bucket-count",
        str(args.bucket_count),
        "--bucket-bytes",
        str(args.bucket_bytes),
        "--storage-layout",
        args.storage_layout,
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--profile-bytes",
        str(args.profile_bytes),
        "--min-pool-bytes",
        str(args.min_pool_bytes),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--mode",
        args.mode,
        "--json-output",
        str(paths["json"]),
        "--summary-output",
        str(paths["summary"]),
        "--no-copy-summary",
    ]
    if args.verify:
        command.append("--verify")
    if args.force_profile:
        command.append("--force-profile")
    command.extend(daemon_command_args(args))
    return command


def build_training_offload_command(args, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARKS / "training_offload.py"),
        "--target-gpu",
        str(args.target_gpu),
        "--relay-gpus",
        args.relay_gpus,
        "--bucket-count",
        str(args.bucket_count),
        "--bucket-bytes",
        str(args.bucket_bytes),
        "--storage-layout",
        args.storage_layout,
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--profile-bytes",
        str(args.profile_bytes),
        "--min-pool-bytes",
        str(args.min_pool_bytes),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--mode",
        args.mode,
        "--compute-elements",
        str(args.compute_elements),
        "--compute-iterations",
        str(args.compute_iterations),
        "--json-output",
        str(paths["json"]),
        "--summary-output",
        str(paths["summary"]),
        "--no-copy-summary",
    ]
    if args.active_buckets is not None:
        command.extend(["--active-buckets", str(args.active_buckets)])
    if args.verify:
        command.append("--verify")
    if args.force_profile:
        command.append("--force-profile")
    command.extend(daemon_command_args(args))
    return command


def build_vllm_kv_command(args, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES / "vllm_turbobus_kv_connector_sweep.py"),
        "--model",
        args.vllm_model,
        "--target-gpu",
        str(args.target_gpu),
        "--relay-gpus",
        args.relay_gpus,
        "--prompt-repeat",
        str(args.vllm_prompt_repeat),
        "--restore-blocks-list",
        args.vllm_restore_blocks_list,
        "--tokens-per-block",
        str(args.vllm_tokens_per_block),
        "--modes",
        mode_list(args.mode),
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--profile-bytes",
        str(args.profile_bytes),
        "--min-pool-bytes",
        str(args.min_pool_bytes),
        "--log-dir",
        str(paths["log_dir"]),
        "--summary-output",
        str(paths["summary"]),
        "--cases-json-output",
        str(paths["cases_json"]),
        "--cases-csv-output",
        str(paths["cases_csv"]),
    ]
    if args.vllm_enforce_eager:
        command.append("--enforce-eager")
    if args.vllm_enable_multiproc_executor:
        command.append("--enable-multiproc-executor")
    if args.vllm_no_map_physical_gpus:
        command.append("--no-map-physical-gpus")
    command.extend(daemon_command_args(args))
    return command


def build_workload_command(args, workload: str, paths: dict[str, Path]) -> list[str]:
    if workload == "model-loading":
        return build_model_loading_command(args, paths)
    if workload == "training-offload":
        return build_training_offload_command(args, paths)
    if workload == "vllm-kv":
        return build_vllm_kv_command(args, paths)
    raise ValueError(f"unsupported workload: {workload}")


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NA"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    try:
        if value in (None, "", "NA"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def first_status(*items: dict) -> str:
    for item in items:
        status = item.get("daemon_reservation_status") if item else None
        if status:
            return str(status)
    return ""


def first_fallback_reason(mode_result: dict) -> str:
    decision = mode_result.get("last_auto_decision", {}) or {}
    reason = decision.get("auto_reason", "")
    return str(reason or "")


def collect_model_metrics(result: dict) -> list[dict[str, object]]:
    metrics = []
    for mode, mode_result in sorted((result.get("modes") or {}).items()):
        summary = mode_result.get("summary", {}) or {}
        direct_bytes = as_int(summary.get("direct_bytes"))
        relay_bytes = as_int(summary.get("relay_bytes"))
        metrics.append(
            {
                "workload": "model-loading",
                "mode": mode,
                "ttft_proxy_ms": as_float(summary.get("median_load_ms")),
                "throughput_gib_s": as_float(summary.get("median_gib_per_second")),
                "transfer_bytes": direct_bytes + relay_bytes,
                "direct_bytes": direct_bytes,
                "relay_bytes": relay_bytes,
                "direct_chunks": as_int(summary.get("direct_chunks")),
                "relay_chunks": as_int(summary.get("relay_chunks")),
                "daemon_reservation_status": first_status(
                    mode_result.get("daemon_reservation", {}) or {}
                ),
                "fallback_reason": first_fallback_reason(mode_result),
            }
        )
    return metrics


def transfer_side_bytes(summary: dict, side: str) -> tuple[int, int, int, int]:
    data = summary.get(side, {}) or {}
    return (
        as_int(data.get("direct_bytes")),
        as_int(data.get("relay_bytes")),
        as_int(data.get("direct_chunks")),
        as_int(data.get("relay_chunks")),
    )


def collect_training_metrics(result: dict) -> list[dict[str, object]]:
    metrics = []
    for mode, mode_result in sorted((result.get("modes") or {}).items()):
        summary = mode_result.get("summary", {}) or {}
        prefetch = transfer_side_bytes(summary, "prefetch")
        offload = transfer_side_bytes(summary, "offload")
        direct_bytes = prefetch[0] + offload[0]
        relay_bytes = prefetch[1] + offload[1]
        direct_chunks = prefetch[2] + offload[2]
        relay_chunks = prefetch[3] + offload[3]
        metrics.append(
            {
                "workload": "training-offload",
                "mode": mode,
                "iteration_ms": as_float(summary.get("median_iteration_ms")),
                "transfer_ms": as_float(summary.get("median_transfer_ms")),
                "compute_ms": as_float(summary.get("median_compute_ms")),
                "throughput_gib_s": as_float(summary.get("median_gib_per_second")),
                "transfer_bytes": direct_bytes + relay_bytes,
                "direct_bytes": direct_bytes,
                "relay_bytes": relay_bytes,
                "direct_chunks": direct_chunks,
                "relay_chunks": relay_chunks,
                "daemon_reservation_status": first_status(
                    mode_result.get("prefetch_daemon_reservation", {}) or {},
                    mode_result.get("offload_daemon_reservation", {}) or {},
                ),
                "fallback_reason": first_fallback_reason(mode_result),
            }
        )
    return metrics


def collect_vllm_metrics(case_rows: list[dict]) -> list[dict[str, object]]:
    metrics = []
    for row in case_rows:
        transfer_bytes = as_int(row.get("bytes"))
        metrics.append(
            {
                "workload": "vllm-kv",
                "mode": str(row.get("mode", "")),
                "restore_blocks": as_int(row.get("restore_blocks")),
                "matched_tokens": as_int(row.get("matched_tokens")),
                "ttft_ms": as_float(row.get("start_load_ms")),
                "save_ms": as_float(row.get("save_ms")),
                "restore_latency_ms": as_float(row.get("restore_ms")),
                "restore_transfer_ms": as_float(row.get("restore_transfer_ms")),
                "throughput_gib_s": as_float(row.get("restore_gib_s")),
                "transfer_bytes": transfer_bytes,
                "direct_bytes": 0,
                "relay_bytes": 0,
                "direct_chunks": as_int(row.get("direct_chunks")),
                "relay_chunks": as_int(row.get("relay_chunks")),
                "daemon_reservation_status": str(row.get("daemon_reservation_status", "")),
                "fallback_reason": str(row.get("auto_reason", "")),
            }
        )
    return metrics


def collect_workload_metrics(workload: str, paths: dict[str, Path]) -> tuple[object, list[dict]]:
    if workload == "model-loading":
        data = read_json(paths["json"], {})
        return data, collect_model_metrics(data)
    if workload == "training-offload":
        data = read_json(paths["json"], {})
        return data, collect_training_metrics(data)
    if workload == "vllm-kv":
        rows = read_json(paths["cases_json"], [])
        return rows, collect_vllm_metrics(rows)
    raise ValueError(f"unsupported workload: {workload}")


def metric_line(metric: dict) -> str:
    ordered = [
        "workload",
        "mode",
        "restore_blocks",
        "matched_tokens",
        "ttft_proxy_ms",
        "ttft_ms",
        "save_ms",
        "restore_latency_ms",
        "restore_transfer_ms",
        "iteration_ms",
        "transfer_ms",
        "compute_ms",
        "throughput_gib_s",
        "transfer_bytes",
        "direct_bytes",
        "relay_bytes",
        "direct_chunks",
        "relay_chunks",
        "daemon_reservation_status",
        "fallback_reason",
    ]
    fields = ["paper_metric"]
    for name in ordered:
        value = metric.get(name)
        if value in (None, ""):
            continue
        if isinstance(value, float):
            fields.append(f"{name}={value:.3f}")
        else:
            fields.append(f"{name}={value}")
    return " ".join(fields)


def compact_summary(result: dict) -> str:
    config = result["config"]
    lines = [
        "PAPER_VALIDATION_SUMMARY_BEGIN",
        (
            "paper_validation_config "
            f"target={config['target_gpu']} relays={config['relay_gpus']} "
            f"workloads={','.join(config['workloads'])} mode={config['mode']} "
            f"output_dir={config['output_dir']}"
        ),
    ]
    for workload in result["workloads"]:
        lines.append(
            "paper_workload "
            f"workload={workload['workload']} status={workload['status']} "
            f"returncode={workload['returncode']} summary={workload['summary_path']} "
            f"json={workload['data_path']}"
        )
        for metric in workload["metrics"]:
            lines.append(metric_line(metric))
    lines.append("PAPER_VALIDATION_SUMMARY_END")
    return "\n".join(lines)


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def run_validation(args) -> dict:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = selected_workloads(args.workloads)
    result = {
        "config": {
            "target_gpu": args.target_gpu,
            "relay_gpus": args.relay_gpus,
            "workloads": workloads,
            "mode": args.mode,
            "output_dir": str(output_dir),
            "daemon_socket_path": args.daemon_socket_path,
            "daemon_max_inflight_chunks": args.daemon_max_inflight_chunks,
            "daemon_profile_max_age_seconds": args.daemon_profile_max_age_seconds,
        },
        "workloads": [],
    }

    for workload in workloads:
        paths = output_paths(output_dir, workload)
        command = build_workload_command(args, workload, paths)
        print("paper_validation_run", f"workload={workload}", " ".join(command), flush=True)
        completed = run_command(command)
        data, metrics = collect_workload_metrics(workload, paths)
        data_path = paths["cases_json"] if workload == "vllm-kv" else paths["json"]
        workload_result = {
            "workload": workload,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "summary_path": str(paths["summary"]),
            "data_path": str(data_path),
            "data": data,
            "metrics": metrics,
        }
        result["workloads"].append(workload_result)
        if completed.returncode != 0 and not args.keep_going:
            break
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TurboBus paper-style workload validation")
    parser.add_argument("--workloads", default="all", help="Comma-separated: all, model-loading, vllm-kv, training-offload")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--mode", choices=["auto", "pool", "direct", "relay", "all"], default="all")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--min-pool-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--active-buckets", type=int)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--compute-elements", type=int, default=1_048_576)
    parser.add_argument("--compute-iterations", type=int, default=20)
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vllm-prompt-repeat", type=int, default=64)
    parser.add_argument("--vllm-restore-blocks-list", default="8")
    parser.add_argument("--vllm-tokens-per-block", type=int, default=16)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-enable-multiproc-executor", action="store_true")
    parser.add_argument("--vllm-no-map-physical-gpus", action="store_true")
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results/paper_validation")
    parser.add_argument("--json-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    add_daemon_options(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_validation(args)
    summary = compact_summary(result)
    if args.json_output:
        output = Path(args.json_output)
        if not output.is_absolute():
            output = REPO_ROOT / output
        write_json(output, result)
        print("paper_validation json_output", output)
    if args.summary_output:
        output = Path(args.summary_output)
        if not output.is_absolute():
            output = REPO_ROOT / output
        write_text(output, summary)
        print("paper_validation summary_output", output)
    if not args.no_copy_summary:
        print(summary)
    if any(item["returncode"] != 0 for item in result["workloads"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
