from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_summary_line(line: str) -> tuple[str, dict[str, str]]:
    tokens = shlex.split(line)
    if not tokens:
        return "", {}
    values = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return tokens[0], values


def extract_copy_summary(text: str) -> list[str]:
    in_summary = False
    lines = []
    for line in text.splitlines():
        if line.strip() == "COPY_SUMMARY_BEGIN":
            in_summary = True
            continue
        if line.strip() == "COPY_SUMMARY_END":
            break
        if in_summary:
            lines.append(line.strip())
    return lines


def parse_copy_summary(text: str) -> dict[str, dict[str, str]]:
    parsed = {}
    for line in extract_copy_summary(text):
        name, values = parse_summary_line(line)
        if name:
            parsed[name] = values
    return parsed


def gib_per_second(byte_count: str, elapsed_ms: str) -> str:
    try:
        bytes_value = float(byte_count)
        ms_value = float(elapsed_ms)
    except ValueError:
        return "NA"
    if bytes_value <= 0.0 or ms_value <= 0.0:
        return "NA"
    return f"{bytes_value / (1024.0 ** 3) / (ms_value / 1000.0):.3f}"


def run_case(args, case_id: str, restore_blocks: int, matched_tokens: int, log_path: Path):
    script = Path(__file__).with_name("vllm_turbobus_kv_connector.py")
    repo_root = Path(__file__).resolve().parents[1]
    session_id = f"{args.session_id}-{case_id}-blocks{restore_blocks}"
    command = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--job-id",
        args.job_id,
        "--session-id",
        session_id,
        "--cpu-buffer-id",
        args.cpu_buffer_id,
        "--gpu-buffer-id",
        args.gpu_buffer_id,
        "--prompt-repeat",
        str(args.prompt_repeat),
        "--second-prompt-suffix",
        args.second_prompt_suffix,
        "--prefix-key",
        f"{args.prefix_key}-{case_id}-{restore_blocks}",
        "--matched-tokens",
        str(matched_tokens),
        "--restore-blocks",
        str(restore_blocks),
        "--restore-enabled",
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--daemon-socket-path",
        args.daemon_socket_path,
        "--log-output",
        str(log_path),
    ]
    if args.wait_timeout_seconds is not None:
        command.extend(["--wait-timeout-seconds", str(args.wait_timeout_seconds)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.enable_multiproc_executor:
        command.append("--enable-multiproc-executor")

    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    summary = parse_copy_summary(log_text)
    return {
        "case_id": case_id,
        "restore_blocks": restore_blocks,
        "matched_tokens": matched_tokens,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "log_path": str(log_path),
        "summary": summary,
    }


def build_sweep_summary_lines(args, results) -> list[str]:
    case_rows = build_case_rows(args, results)
    lines = ["SWEEP_SUMMARY_BEGIN", _config_line(args)]
    lines.extend(_case_line(row) for row in case_rows)
    lines.append("SWEEP_SUMMARY_END")
    return lines


def build_case_rows(args, results) -> list[dict[str, object]]:
    case_rows = []
    for result in results:
        summary = result["summary"]
        config = summary.get("vllm_kv_connector_config", {})
        save = summary.get("vllm_kv_connector_save", {})
        save_event = _event_from_log(Path(result["log_path"]), "save")
        restore = _event_from_log(Path(result["log_path"]), "restore")
        start_load = _event_from_log(Path(result["log_path"]), "start_load_done")
        output = summary.get("vllm_kv_connector_result", {})
        row = {
            "case_id": result["case_id"],
            "restore_blocks": result["restore_blocks"],
            "matched_tokens": result["matched_tokens"],
            "returncode": result["returncode"],
            "job_id": config.get("job_id", getattr(args, "job_id", "NA")),
            "session_id": config.get("session_id", "NA"),
            "save_ms": save.get("elapsed_ms", "NA"),
            "save_prepare_ms": save_event.get("prepare_ms", "NA"),
            "save_cpu_alloc_ms": save_event.get("cpu_alloc_ms", "NA"),
            "save_adapter_ms": save_event.get("adapter_ms", "NA"),
            "save_refs_ms": save_event.get("refs_ms", "NA"),
            "save_transfer_ms": save_event.get("transfer_ms", save.get("elapsed_ms", "NA")),
            "save_register_ms": save_event.get("register_ms", "NA"),
            "save_total_ms": save_event.get("total_ms", "NA"),
            "save_layer_count": save.get("save_layer_count", "NA"),
            "save_layer_ranges": save.get("save_layer_ranges", "NA"),
            "save_receipt_ids": save.get("receipt_ids", save_event.get("receipt_ids", "NA")),
            "save_decision_ids": save.get("decision_ids", save_event.get("decision_ids", "NA")),
            "save_topology_snapshot_ids": save.get(
                "topology_snapshot_ids",
                save_event.get("topology_snapshot_ids", "NA"),
            ),
            "save_ticket_ids": save.get("ticket_ids", save_event.get("ticket_ids", "NA")),
            "save_fallback_reason": save.get("fallback_reason", save_event.get("fallback_reason", "NA")),
            "restore_ms": restore.get("elapsed_ms", "NA"),
            "restore_prepare_ms": restore.get("prepare_ms", "NA"),
            "restore_transfer_ms": restore.get("transfer_ms", restore.get("elapsed_ms", "NA")),
            "restore_total_ms": restore.get("total_ms", "NA"),
            "start_load_ms": start_load.get("elapsed_ms", "NA"),
            "bytes": restore.get("bytes", save.get("bytes", "NA")),
            "direct_chunks": restore.get("direct_chunks", "NA"),
            "relay_chunks": restore.get("relay_chunks", "NA"),
            "direct_bytes": restore.get("direct_bytes", "NA"),
            "relay_bytes": restore.get("relay_bytes", "NA"),
            "receipt_ids": restore.get("receipt_ids", "NA"),
            "decision_ids": restore.get("decision_ids", "NA"),
            "topology_snapshot_ids": restore.get("topology_snapshot_ids", "NA"),
            "ticket_ids": restore.get("ticket_ids", "NA"),
            "fallback_reason": restore.get("fallback_reason", "NA"),
            "layers": restore.get("layers", "NA"),
            "ranges": restore.get("ranges", "NA"),
            "prompt_tokens": output.get("prompt_tokens", "NA"),
            "shared_prefix": output.get("shared_prefix", "NA"),
            "log": result["log_path"],
        }
        row["restore_gib_s"] = gib_per_second(row["bytes"], row["restore_ms"])
        case_rows.append(row)
    return case_rows


def print_sweep_summary(
    args,
    results,
    output_path: Path | None = None,
    cases_json_output: Path | None = None,
    cases_csv_output: Path | None = None,
) -> None:
    case_rows = build_case_rows(args, results)
    lines = ["SWEEP_SUMMARY_BEGIN", _config_line(args)]
    lines.extend(_case_line(row) for row in case_rows)
    lines.append("SWEEP_SUMMARY_END")
    text = "\n".join(lines)
    print(text)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print("vllm_kv_connector_sweep summary", output_path)
    if cases_json_output is not None:
        cases_json_output.parent.mkdir(parents=True, exist_ok=True)
        cases_json_output.write_text(json.dumps(case_rows, indent=2) + "\n", encoding="utf-8")
        print("vllm_kv_connector_sweep cases_json", cases_json_output)
    if cases_csv_output is not None:
        _write_case_rows_csv(cases_csv_output, case_rows)
        print("vllm_kv_connector_sweep cases_csv", cases_csv_output)


def _config_line(args) -> str:
    return " ".join(
        [
            "vllm_kv_connector_sweep_config",
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            f"model={args.model}",
            f"job_id={args.job_id}",
            f"session_id={args.session_id}",
            f"cpu_buffer_id={args.cpu_buffer_id}",
            f"gpu_buffer_id={args.gpu_buffer_id}",
            f"prompt_repeat={args.prompt_repeat}",
            f"case_ids={','.join(args.case_ids)}",
            f"restore_blocks_list={','.join(str(item) for item in args.restore_blocks_list)}",
            f"chunk_bytes={args.chunk_bytes}",
            f"daemon_socket_path={args.daemon_socket_path}",
            f"wait_timeout_seconds={args.wait_timeout_seconds}",
        ]
    )


def _case_line(row) -> str:
    keys = [
        "case_id",
        "restore_blocks",
        "matched_tokens",
        "returncode",
        "job_id",
        "session_id",
        "save_ms",
        "save_prepare_ms",
        "save_cpu_alloc_ms",
        "save_adapter_ms",
        "save_refs_ms",
        "save_transfer_ms",
        "save_register_ms",
        "save_total_ms",
        "save_layer_count",
        "save_layer_ranges",
        "save_receipt_ids",
        "save_decision_ids",
        "save_topology_snapshot_ids",
        "save_ticket_ids",
        "save_fallback_reason",
        "restore_ms",
        "restore_gib_s",
        "restore_prepare_ms",
        "restore_transfer_ms",
        "restore_total_ms",
        "start_load_ms",
        "bytes",
        "direct_chunks",
        "relay_chunks",
        "direct_bytes",
        "relay_bytes",
        "receipt_ids",
        "decision_ids",
        "topology_snapshot_ids",
        "ticket_ids",
        "fallback_reason",
        "layers",
        "ranges",
        "prompt_tokens",
        "shared_prefix",
        "log",
    ]
    return " ".join(["vllm_kv_connector_sweep_case", *(f"{key}={row[key]}" for key in keys)])


def _write_case_rows_csv(path: Path, case_rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not case_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)


def _event_from_log(log_path: Path, event: str) -> dict[str, str]:
    if not log_path.exists():
        return {}
    prefix = f"turbobus_kv_connector_event event={event} "
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if prefix not in line:
            continue
        _, values = parse_summary_line(line.replace("turbobus_kv_connector_event ", ""))
        return values
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run daemon-scheduled vLLM KV connector cases through TurboBus"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--job-id", default="vllm-kv-sweep-job")
    parser.add_argument("--session-id", default="vllm-kv-sweep-session")
    parser.add_argument("--cpu-buffer-id", default="vllm-kv-cpu-buffer")
    parser.add_argument("--gpu-buffer-id", default="vllm-kv-gpu-buffer")
    parser.add_argument("--prompt-repeat", type=int, default=64)
    parser.add_argument("--second-prompt-suffix", default=" Italy")
    parser.add_argument("--prefix-key", default="qwen3-prefix")
    parser.add_argument("--restore-blocks-list", default="8")
    parser.add_argument("--tokens-per-block", type=int, default=16)
    parser.add_argument("--case-ids", default="daemon")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--daemon-socket-path", required=True)
    parser.add_argument("--wait-timeout-seconds", type=float, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-multiproc-executor", action="store_true")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--cases-json-output", default=None)
    parser.add_argument("--cases-csv-output", default=None)
    args = parser.parse_args()
    args.case_ids = parse_csv_strings(args.case_ids)
    args.restore_blocks_list = parse_csv_ints(args.restore_blocks_list)
    return args


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = (
        Path(args.log_dir)
        if args.log_dir is not None
        else Path("benchmarks") / "results" / f"vllm_kv_connector_sweep_{stamp}"
    )
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_output = Path(args.summary_output) if args.summary_output is not None else log_dir / "sweep_summary.txt"
    if not summary_output.is_absolute():
        summary_output = repo_root / summary_output
    cases_json_output = Path(args.cases_json_output) if args.cases_json_output is not None else log_dir / "sweep_cases.json"
    if not cases_json_output.is_absolute():
        cases_json_output = repo_root / cases_json_output
    cases_csv_output = Path(args.cases_csv_output) if args.cases_csv_output is not None else log_dir / "sweep_cases.csv"
    if not cases_csv_output.is_absolute():
        cases_csv_output = repo_root / cases_csv_output

    results = []
    failed = False
    for restore_blocks in args.restore_blocks_list:
        matched_tokens = restore_blocks * args.tokens_per_block
        for case_id in args.case_ids:
            log_path = log_dir / f"{case_id}_blocks{restore_blocks}.log"
            print(
                "vllm_kv_connector_sweep run",
                f"case_id={case_id}",
                f"restore_blocks={restore_blocks}",
                f"matched_tokens={matched_tokens}",
                f"log={log_path}",
                flush=True,
            )
            result = run_case(args, case_id, restore_blocks, matched_tokens, log_path)
            results.append(result)
            if result["returncode"] != 0:
                failed = True
                break
        if failed:
            break

    print_sweep_summary(
        args,
        results,
        summary_output,
        cases_json_output=cases_json_output,
        cases_csv_output=cases_csv_output,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
