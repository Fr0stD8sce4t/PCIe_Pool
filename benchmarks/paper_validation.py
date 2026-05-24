from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from daemon_support import add_daemon_options


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"
EXAMPLES = REPO_ROOT / "examples"

WORKLOADS = ("model-loading", "training-offload", "vllm-kv")
OUTPUT_FILE_KEYS = ("json", "summary")


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


def output_paths(output_dir: Path, workload: str) -> dict[str, Path]:
    safe = workload.replace("-", "_")
    paths = {
        "json": output_dir / f"{safe}.json",
        "summary": output_dir / f"{safe}_summary.txt",
    }
    if workload == "vllm-kv":
        paths["log"] = output_dir / f"{safe}.log"
    return paths


def clear_workload_outputs(paths: dict[str, Path]) -> None:
    for key in OUTPUT_FILE_KEYS:
        path = paths[key]
        if path.is_file():
            path.unlink()


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
        "--session-id",
        args.session_id,
        "--job-id",
        args.job_id,
        "--source-buffer-id",
        args.cpu_buffer_id,
        "--destination-buffer-id",
        args.gpu_buffer_id,
        "--bucket-count",
        str(args.bucket_count),
        "--bucket-bytes",
        str(args.bucket_bytes),
        "--storage-layout",
        args.storage_layout,
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--policy",
        args.policy,
        "--run-id",
        args.run_id,
        "--json-output",
        str(paths["json"]),
        "--summary-output",
        str(paths["summary"]),
        "--no-copy-summary",
    ]
    command.extend(daemon_command_args(args))
    return command


def build_training_offload_command(args, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARKS / "training_offload.py"),
        "--session-id",
        args.session_id,
        "--job-id",
        args.job_id,
        "--cpu-buffer-id",
        args.cpu_buffer_id,
        "--gpu-buffer-id",
        args.gpu_buffer_id,
        "--workload-kind",
        args.training_workload_kind,
        "--bucket-count",
        str(args.bucket_count),
        "--bucket-bytes",
        str(args.bucket_bytes),
        "--storage-layout",
        args.storage_layout,
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--compute-delay-ms",
        str(args.compute_delay_ms),
        "--policy",
        args.policy,
        "--run-id",
        args.run_id,
        "--json-output",
        str(paths["json"]),
        "--summary-output",
        str(paths["summary"]),
        "--no-copy-summary",
    ]
    if args.active_buckets is not None:
        command.extend(["--active-buckets", str(args.active_buckets)])
    command.extend(daemon_command_args(args))
    return command


def build_vllm_kv_command(args, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES / "vllm_turbobus_kv_connector.py"),
        "--model",
        args.vllm_model,
        "--job-id",
        args.job_id,
        "--session-id",
        args.session_id,
        "--cpu-buffer-id",
        args.cpu_buffer_id,
        "--gpu-buffer-id",
        args.gpu_buffer_id,
        "--prompt-repeat",
        str(args.vllm_prompt_repeat),
        "--second-prompt-suffix",
        args.vllm_second_prompt_suffix,
        "--prefix-key",
        args.vllm_prefix_key,
        "--matched-tokens",
        str(args.vllm_matched_tokens),
        "--restore-blocks",
        str(args.vllm_restore_blocks),
        "--restore-enabled",
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--daemon-socket-path",
        args.daemon_socket_path,
        "--log-output",
        str(paths["log"]),
    ]
    if args.vllm_prompt:
        command.extend(["--prompt", args.vllm_prompt])
    if args.vllm_wait_timeout_seconds is not None:
        command.extend(["--wait-timeout-seconds", str(args.vllm_wait_timeout_seconds)])
    if args.vllm_enforce_eager:
        command.append("--enforce-eager")
    if args.vllm_enable_multiproc_executor:
        command.append("--enable-multiproc-executor")
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


def parse_summary_line(line: str) -> tuple[str, dict[str, str]]:
    parts = shlex.split(str(line))
    if not parts:
        return "", {}
    values = {}
    for item in parts[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key] = value
    return parts[0], values


def parse_vllm_kv_summary(log_path: Path) -> dict[str, dict[str, str]]:
    if not log_path.exists():
        return {}
    parsed = {}
    in_summary = False
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "COPY_SUMMARY_BEGIN":
            in_summary = True
            continue
        if line == "COPY_SUMMARY_END":
            break
        if not in_summary:
            continue
        name, values = parse_summary_line(line)
        if name:
            parsed[name] = values
    return parsed


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


def join_values(values) -> str:
    if values in (None, ""):
        return ""
    if isinstance(values, (list, tuple, set)):
        return ",".join(str(item) for item in values if item not in (None, ""))
    return str(values)


def _gib_per_second_value(byte_count: int, elapsed_ms: float) -> float:
    if byte_count <= 0 or elapsed_ms <= 0:
        return 0.0
    return float(byte_count) / (1024.0**3) / (float(elapsed_ms) / 1000.0)


def collect_model_metrics(result: dict) -> list[dict[str, object]]:
    summary = result.get("summary", {}) or {}
    if not summary:
        return []
    config = result.get("config", {}) or {}
    return [
        {
            "workload": "model-loading",
            "policy": str(config.get("policy", "")),
            "iterations": as_int(summary.get("iterations")),
            "ttft_proxy_ms": as_float(summary.get("median_load_ms")),
            "throughput_gib_s": as_float(summary.get("median_gib_per_second")),
            "transfer_bytes": as_int(summary.get("bytes")),
            "bytes_completed": as_int(summary.get("bytes_completed")),
            "direct_bytes": as_int(summary.get("direct_bytes")),
            "relay_bytes": as_int(summary.get("relay_bytes")),
            "direct_chunks": as_int(summary.get("direct_chunks")),
            "relay_chunks": as_int(summary.get("relay_chunks")),
            "decision_ids": join_values(summary.get("decision_ids")),
            "topology_snapshot_ids": join_values(summary.get("topology_snapshot_ids")),
            "ticket_ids": join_values(summary.get("ticket_ids")),
            "fallback_reason": join_values(summary.get("fallback_reasons")),
        }
    ]


def transfer_side_summary(summary: dict, side: str) -> dict:
    return summary.get(side, {}) or {}


def collect_training_metrics(result: dict) -> list[dict[str, object]]:
    summary = result.get("summary", {}) or {}
    if not summary:
        return []
    config = result.get("config", {}) or {}
    prefetch = transfer_side_summary(summary, "prefetch")
    offload = transfer_side_summary(summary, "offload")
    direct_bytes = as_int(prefetch.get("direct_bytes")) + as_int(offload.get("direct_bytes"))
    relay_bytes = as_int(prefetch.get("relay_bytes")) + as_int(offload.get("relay_bytes"))
    direct_chunks = as_int(prefetch.get("direct_chunks")) + as_int(offload.get("direct_chunks"))
    relay_chunks = as_int(prefetch.get("relay_chunks")) + as_int(offload.get("relay_chunks"))
    return [
        {
            "workload": "training-offload",
            "policy": str(config.get("policy", "")),
            "iterations": as_int(summary.get("iterations")),
            "iteration_ms": as_float(summary.get("median_iteration_ms")),
            "transfer_ms": as_float(summary.get("median_transfer_ms")),
            "compute_ms": as_float(summary.get("median_compute_ms")),
            "throughput_gib_s": as_float(summary.get("median_gib_per_second")),
            "transfer_bytes": as_int(prefetch.get("bytes")) + as_int(offload.get("bytes")),
            "bytes_completed": as_int(prefetch.get("bytes_completed")) + as_int(offload.get("bytes_completed")),
            "direct_bytes": direct_bytes,
            "relay_bytes": relay_bytes,
            "direct_chunks": direct_chunks,
            "relay_chunks": relay_chunks,
            "decision_ids": join_values(
                [*prefetch.get("decision_ids", ()), *offload.get("decision_ids", ())]
            ),
            "topology_snapshot_ids": join_values(
                [
                    *prefetch.get("topology_snapshot_ids", ()),
                    *offload.get("topology_snapshot_ids", ()),
                ]
            ),
            "ticket_ids": join_values(
                [*prefetch.get("ticket_ids", ()), *offload.get("ticket_ids", ())]
            ),
            "prefetch_decision_ids": join_values(prefetch.get("decision_ids")),
            "offload_decision_ids": join_values(offload.get("decision_ids")),
            "fallback_reason": join_values(
                [*prefetch.get("fallback_reasons", ()), *offload.get("fallback_reasons", ())]
            ),
        }
    ]


def collect_vllm_kv_metrics(summary: dict) -> list[dict[str, object]]:
    config = summary.get("vllm_kv_connector_config", {}) or {}
    save = summary.get("vllm_kv_connector_save", {}) or {}
    restore = summary.get("vllm_kv_connector_restore", {}) or {}
    result = summary.get("vllm_kv_connector_result", {}) or {}
    if not save or not restore:
        return []
    return [
        {
            "workload": "vllm-kv",
            "policy": "daemon-default",
            "iterations": 1,
            "ttft_proxy_ms": as_float(restore.get("total_ms")),
            "transfer_ms": as_float(restore.get("transfer_ms")),
            "throughput_gib_s": _gib_per_second_value(
                as_int(restore.get("bytes")),
                as_float(restore.get("transfer_ms")),
            ),
            "transfer_bytes": as_int(restore.get("bytes")),
            "bytes_completed": as_int(restore.get("bytes")),
            "direct_bytes": as_int(restore.get("direct_bytes")),
            "relay_bytes": as_int(restore.get("relay_bytes")),
            "direct_chunks": as_int(restore.get("direct_chunks")),
            "relay_chunks": as_int(restore.get("relay_chunks")),
            "decision_ids": join_values(restore.get("decision_ids")),
            "topology_snapshot_ids": join_values(restore.get("topology_snapshot_ids")),
            "ticket_ids": join_values(restore.get("ticket_ids")),
            "save_decision_ids": join_values(save.get("decision_ids")),
            "save_topology_snapshot_ids": join_values(save.get("topology_snapshot_ids")),
            "save_ticket_ids": join_values(save.get("ticket_ids")),
            "fallback_reason": join_values(restore.get("fallback_reason")),
            "save_receipt_ids": join_values(save.get("receipt_ids")),
            "restore_receipt_ids": join_values(restore.get("receipt_ids")),
            "save_ms": as_float(save.get("elapsed_ms")),
            "restore_ms": as_float(restore.get("elapsed_ms")),
            "save_layer_count": as_int(save.get("save_layer_count")),
            "save_layer_ranges": as_int(save.get("save_layer_ranges")),
            "restore_layers": as_int(restore.get("layers")),
            "restore_ranges": as_int(restore.get("ranges")),
            "prompt_tokens": as_int(result.get("prompt_tokens")),
            "shared_prefix": str(result.get("shared_prefix", "")),
            "model": str(config.get("model", "")),
        }
    ]


def collect_workload_metrics(workload: str, paths: dict[str, Path]) -> tuple[object, list[dict]]:
    data = read_json(paths["json"], {})
    if workload == "model-loading":
        return data, collect_model_metrics(data)
    if workload == "training-offload":
        return data, collect_training_metrics(data)
    if workload == "vllm-kv":
        if not data:
            data = parse_vllm_kv_summary(paths["log"])
        return data, collect_vllm_kv_metrics(data)
    raise ValueError(f"unsupported workload: {workload}")


def workload_validation_errors(data_path: Path, metrics: list[dict]) -> list[str]:
    errors = []
    if not data_path.exists():
        errors.append("missing_output_file")
    if not metrics:
        errors.append("missing_paper_metrics")
        return errors
    missing_trace = [
        metric["workload"]
        for metric in metrics
        if not metric.get("decision_ids")
        or not metric.get("topology_snapshot_ids")
        or not metric.get("ticket_ids")
    ]
    if missing_trace:
        errors.append("missing_daemon_trace")
    return errors


def vllm_kv_validation_errors(paths: dict[str, Path], metrics: list[dict]) -> list[str]:
    errors = workload_validation_errors(paths["json"], metrics)
    if not paths["log"].exists():
        errors.append("missing_log_file")
        return errors
    summary = parse_vllm_kv_summary(paths["log"])
    for key in (
        "vllm_kv_connector_config",
        "vllm_kv_connector_scenario",
        "vllm_kv_connector_save",
        "vllm_kv_connector_restore",
        "vllm_kv_connector_result",
    ):
        if key not in summary:
            errors.append(f"missing_{key}")
    text = paths["log"].read_text(encoding="utf-8")
    for event in (
        "register_kv_caches",
        "save_layer",
        "wait_for_save_done",
        "save",
        "restore",
        "start_load_done",
    ):
        if f"turbobus_kv_connector_event event={event}" not in text:
            errors.append(f"missing_event_{event}")
    return errors


def workload_status(dry_run: bool, returncode: int, validation_errors: list[str]) -> str:
    if dry_run:
        return "dry-run"
    if returncode != 0:
        return "failed"
    if "invalid_output" in validation_errors:
        return "invalid-output"
    if "missing_output_file" in validation_errors:
        return "missing-output"
    if validation_errors:
        return "missing-metrics"
    return "ok"


def workload_failed(status: str) -> bool:
    return status not in ("ok", "dry-run")


def metric_line(metric: dict) -> str:
    ordered = [
        "workload",
        "policy",
        "iterations",
        "ttft_proxy_ms",
        "iteration_ms",
        "transfer_ms",
        "compute_ms",
        "throughput_gib_s",
        "transfer_bytes",
        "bytes_completed",
        "direct_bytes",
        "relay_bytes",
        "direct_chunks",
        "relay_chunks",
        "decision_ids",
        "topology_snapshot_ids",
        "ticket_ids",
        "prefetch_decision_ids",
        "offload_decision_ids",
        "save_decision_ids",
        "save_topology_snapshot_ids",
        "save_ticket_ids",
        "save_ms",
        "restore_ms",
        "save_layer_count",
        "save_layer_ranges",
        "restore_layers",
        "restore_ranges",
        "prompt_tokens",
        "shared_prefix",
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
            f"session_id={config['session_id']} job_id={config['job_id']} "
            f"cpu_buffer_id={config['cpu_buffer_id']} "
            f"gpu_buffer_id={config['gpu_buffer_id']} "
            f"workloads={','.join(config['workloads'])} "
            f"policy={config['policy']} "
            f"dry_run={config.get('dry_run', False)} output_dir={config['output_dir']}"
        ),
    ]
    for workload in result["workloads"]:
        errors = ",".join(workload.get("validation_errors", []))
        lines.append(
            "paper_workload "
            f"workload={workload['workload']} status={workload['status']} "
            f"returncode={workload['returncode']} summary={workload['summary_path']} "
            f"json={workload['data_path']} validation_errors={errors}"
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
    if "vllm-kv" in workloads and not str(args.vllm_model).strip():
        raise ValueError("--vllm-model is required when workload vllm-kv is selected")
    if "vllm-kv" in workloads and not str(args.daemon_socket_path).strip():
        raise ValueError("--daemon-socket-path is required when workload vllm-kv is selected")
    result = {
        "config": {
            "session_id": args.session_id,
            "job_id": args.job_id,
            "cpu_buffer_id": args.cpu_buffer_id,
            "gpu_buffer_id": args.gpu_buffer_id,
            "workloads": workloads,
            "policy": args.policy,
            "run_id": args.run_id,
            "output_dir": str(output_dir),
            "dry_run": bool(args.dry_run),
            "daemon_socket_path": args.daemon_socket_path,
            "daemon_max_inflight_chunks": args.daemon_max_inflight_chunks,
            "daemon_profile_max_age_seconds": args.daemon_profile_max_age_seconds,
            "vllm_model": args.vllm_model,
        },
        "workloads": [],
    }

    for workload in workloads:
        paths = output_paths(output_dir, workload)
        command = build_workload_command(args, workload, paths)
        data_path = paths["json"]
        validation_errors = []
        if args.dry_run:
            print(
                "paper_validation_dry_run",
                f"workload={workload}",
                " ".join(command),
                flush=True,
            )
            completed = subprocess.CompletedProcess(command, 0, "", "")
            data = {}
            metrics = []
        else:
            clear_workload_outputs(paths)
            print("paper_validation_run", f"workload={workload}", " ".join(command), flush=True)
            completed = run_command(command)
            try:
                data, metrics = collect_workload_metrics(workload, paths)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                data = {}
                metrics = []
                validation_errors.append("invalid_output")
                validation_errors.append(type(exc).__name__)
            if workload == "vllm-kv":
                if data:
                    write_json(data_path, data)
                validation_errors.extend(vllm_kv_validation_errors(paths, metrics))
            else:
                validation_errors.extend(workload_validation_errors(data_path, metrics))
        status = workload_status(args.dry_run, completed.returncode, validation_errors)
        result["workloads"].append(
            {
                "workload": workload,
                "status": status,
                "returncode": completed.returncode,
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "summary_path": str(paths["summary"]),
                "data_path": str(data_path),
                "validation_errors": validation_errors,
                "data": data,
                "metrics": metrics,
            }
        )
        if workload_failed(status) and not args.keep_going:
            break
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TurboBus daemon-first paper validation")
    parser.add_argument(
        "--workloads",
        default="all",
        help="Comma-separated: all, model-loading, training-offload, vllm-kv",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--job-id", default="paper-validation")
    parser.add_argument("--cpu-buffer-id", required=True)
    parser.add_argument("--gpu-buffer-id", required=True)
    parser.add_argument("--policy", default="daemon-default")
    parser.add_argument("--run-id", default="paper-validation")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--active-buckets", type=int)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["packed", "separate"], default="packed")
    parser.add_argument(
        "--training-workload-kind",
        choices=["training_state", "optimizer_state"],
        default="training_state",
    )
    parser.add_argument("--compute-delay-ms", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results/paper_validation")
    parser.add_argument("--json-output")
    parser.add_argument("--summary-output")
    parser.add_argument("--no-copy-summary", action="store_true")
    parser.add_argument("--vllm-model", default="")
    parser.add_argument("--vllm-prompt", default="")
    parser.add_argument("--vllm-prompt-repeat", type=int, default=64)
    parser.add_argument("--vllm-second-prompt-suffix", default=" Italy")
    parser.add_argument("--vllm-prefix-key", default="paper-validation-vllm-kv")
    parser.add_argument("--vllm-restore-blocks", type=int, default=8)
    parser.add_argument("--vllm-matched-tokens", type=int, default=128)
    parser.add_argument("--vllm-wait-timeout-seconds", type=float, default=None)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-enable-multiproc-executor", action="store_true")
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
    if any(workload_failed(item["status"]) for item in result["workloads"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
