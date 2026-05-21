from __future__ import annotations

import argparse
from datetime import datetime
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


def speedup(numerator_ms: str, denominator_ms: str) -> str:
    try:
        numerator = float(numerator_ms)
        denominator = float(denominator_ms)
    except ValueError:
        return "NA"
    if numerator <= 0.0 or denominator <= 0.0:
        return "NA"
    return f"{numerator / denominator:.3f}"


def run_case(args, mode: str, restore_blocks: int, matched_tokens: int, log_path: Path):
    script = Path(__file__).with_name("vllm_turbobus_kv_connector.py")
    repo_root = Path(__file__).resolve().parents[1]
    min_pool_bytes = getattr(args, "min_pool_bytes", 16 * 1024 * 1024)
    command = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--target-gpu",
        str(args.target_gpu),
        "--relay-gpus",
        args.relay_gpus,
        "--prompt-repeat",
        str(args.prompt_repeat),
        "--second-prompt-suffix",
        args.second_prompt_suffix,
        "--prefix-key",
        f"{args.prefix_key}-{mode}-{restore_blocks}",
        "--matched-tokens",
        str(matched_tokens),
        "--restore-blocks",
        str(restore_blocks),
        "--restore-enabled",
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--profile-bytes",
        str(args.profile_bytes),
        "--min-pool-bytes",
        str(min_pool_bytes),
        "--mode",
        mode,
        "--log-output",
        str(log_path),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.enable_multiproc_executor:
        command.append("--enable-multiproc-executor")
    if args.no_map_physical_gpus:
        command.append("--no-map-physical-gpus")

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
        "mode": mode,
        "restore_blocks": restore_blocks,
        "matched_tokens": matched_tokens,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "log_path": str(log_path),
        "summary": summary,
    }


def build_sweep_summary_lines(args, results) -> list[str]:
    case_rows = []
    min_pool_bytes = getattr(args, "min_pool_bytes", 16 * 1024 * 1024)
    lines = ["SWEEP_SUMMARY_BEGIN"]
    lines.append(
        " ".join(
            [
                "vllm_kv_connector_sweep_config",
                f"target={args.target_gpu}",
                f"relays={parse_csv_ints(args.relay_gpus)}",
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                f"model={args.model}",
                f"prompt_repeat={args.prompt_repeat}",
                f"modes={','.join(args.modes)}",
                f"restore_blocks_list={','.join(str(item) for item in args.restore_blocks_list)}",
                f"chunk_bytes={args.chunk_bytes}",
                f"profile_bytes={args.profile_bytes}",
                f"min_pool_bytes={min_pool_bytes}",
            ]
        )
    )
    for result in results:
        summary = result["summary"]
        config = summary.get("vllm_kv_connector_config", {})
        save = summary.get("vllm_kv_connector_save", {})
        save_event = _event_from_log(Path(result["log_path"]), "save")
        restore = _restore_from_log(Path(result["log_path"]))
        start_load = _event_from_log(Path(result["log_path"]), "start_load_done")
        output = summary.get("vllm_kv_connector_result", {})
        row = {
            "mode": result["mode"],
            "restore_blocks": result["restore_blocks"],
            "matched_tokens": result["matched_tokens"],
            "returncode": result["returncode"],
            "save_ms": save.get("elapsed_ms", "NA"),
            "save_runtime_init_ms": save_event.get("runtime_init_ms", "NA"),
            "save_prepare_ms": save_event.get("prepare_ms", "NA") if save_event else "NA",
            "save_cpu_alloc_ms": save_event.get("cpu_alloc_ms", "NA") if save_event else "NA",
            "save_adapter_ms": save_event.get("adapter_ms", "NA") if save_event else "NA",
            "save_refs_ms": save_event.get("refs_ms", "NA") if save_event else "NA",
            "save_transfer_ms": save_event.get("transfer_ms", save.get("elapsed_ms", "NA")) if save_event else save.get("elapsed_ms", "NA"),
            "save_register_ms": save_event.get("register_ms", "NA") if save_event else "NA",
            "save_total_ms": save_event.get("total_ms", "NA") if save_event else "NA",
            "save_layer_count": save.get("save_layer_count", "NA"),
            "save_layer_ranges": save.get("save_layer_ranges", "NA"),
            "restore_ms": restore.get("elapsed_ms", "NA"),
            "restore_prepare_ms": restore.get("prepare_ms", "NA"),
            "restore_transfer_ms": restore.get("transfer_ms", restore.get("elapsed_ms", "NA")),
            "restore_total_ms": restore.get("total_ms", "NA"),
            "start_load_ms": start_load.get("elapsed_ms", "NA"),
            "bytes": restore.get("bytes", save.get("bytes", "NA")),
            "direct_chunks": restore.get("direct_chunks", "NA"),
            "relay_chunks": restore.get("relay_chunks", "NA"),
            "auto_resolved_mode": restore.get("auto_resolved_mode", "NA"),
            "auto_reason": restore.get("auto_reason", "NA"),
            "auto_request_bytes": restore.get("auto_request_bytes", "NA"),
            "auto_request_chunks": restore.get("auto_request_chunks", "NA"),
            "auto_direct_bw_gbps": restore.get("auto_direct_bw_gbps", "NA"),
            "auto_relay_bw_gbps": restore.get("auto_relay_bw_gbps", "NA"),
            "auto_eligible_relays": restore.get("auto_eligible_relays", "NA"),
            "layers": restore.get("layers", "NA"),
            "ranges": restore.get("ranges", "NA"),
            "prompt_tokens": output.get("prompt_tokens", "NA"),
            "shared_prefix": output.get("shared_prefix", "NA"),
            "child_mode": config.get("mode", result["mode"]),
            "log": result["log_path"],
        }
        row["restore_gib_s"] = gib_per_second(row["bytes"], row["restore_ms"])
        case_rows.append(row)
        lines.append(
            " ".join(
                [
                    "vllm_kv_connector_sweep_case",
                    f"mode={row['mode']}",
                    f"restore_blocks={row['restore_blocks']}",
                    f"matched_tokens={row['matched_tokens']}",
                    f"returncode={row['returncode']}",
                    f"save_ms={row['save_ms']}",
                    f"save_runtime_init_ms={row['save_runtime_init_ms']}",
                    f"save_prepare_ms={row['save_prepare_ms']}",
                    f"save_cpu_alloc_ms={row['save_cpu_alloc_ms']}",
                    f"save_adapter_ms={row['save_adapter_ms']}",
                    f"save_refs_ms={row['save_refs_ms']}",
                    f"save_transfer_ms={row['save_transfer_ms']}",
                    f"save_register_ms={row['save_register_ms']}",
                    f"save_total_ms={row['save_total_ms']}",
                    f"save_layer_count={row['save_layer_count']}",
                    f"save_layer_ranges={row['save_layer_ranges']}",
                    f"restore_ms={row['restore_ms']}",
                    f"restore_gib_s={row['restore_gib_s']}",
                    f"restore_prepare_ms={row['restore_prepare_ms']}",
                    f"restore_transfer_ms={row['restore_transfer_ms']}",
                    f"restore_total_ms={row['restore_total_ms']}",
                    f"start_load_ms={row['start_load_ms']}",
                    f"bytes={row['bytes']}",
                    f"direct_chunks={row['direct_chunks']}",
                    f"relay_chunks={row['relay_chunks']}",
                    f"auto_resolved_mode={row['auto_resolved_mode']}",
                    f"auto_reason={row['auto_reason']}",
                    f"auto_request_bytes={row['auto_request_bytes']}",
                    f"auto_request_chunks={row['auto_request_chunks']}",
                    f"auto_direct_bw_gbps={row['auto_direct_bw_gbps']}",
                    f"auto_relay_bw_gbps={row['auto_relay_bw_gbps']}",
                    f"auto_eligible_relays={row['auto_eligible_relays']}",
                    f"layers={row['layers']}",
                    f"ranges={row['ranges']}",
                    f"prompt_tokens={row['prompt_tokens']}",
                    f"shared_prefix={row['shared_prefix']}",
                    f"child_mode={row['child_mode']}",
                    f"log={row['log']}",
                ]
            )
        )
    lines.extend(_speedup_lines(case_rows))
    lines.append("SWEEP_SUMMARY_END")
    return lines


def print_sweep_summary(args, results, output_path: Path | None = None) -> None:
    lines = build_sweep_summary_lines(args, results)
    text = "\n".join(lines)
    print(text)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print("vllm_kv_connector_sweep summary", output_path)


def _speedup_lines(case_rows) -> list[str]:
    lines = []
    by_blocks = {}
    for row in case_rows:
        by_blocks.setdefault(row["restore_blocks"], {})[row["mode"]] = row
    for restore_blocks in sorted(by_blocks):
        rows = by_blocks[restore_blocks]
        pool = rows.get("pool")
        if pool is None:
            continue
        direct = rows.get("direct")
        relay = rows.get("relay")
        lines.append(
            " ".join(
                [
                    "vllm_kv_connector_sweep_speedup",
                    f"restore_blocks={restore_blocks}",
                    f"direct_over_pool_restore={speedup(direct['restore_ms'], pool['restore_ms']) if direct else 'NA'}",
                    f"relay_over_pool_restore={speedup(relay['restore_ms'], pool['restore_ms']) if relay else 'NA'}",
                ]
            )
        )
    return lines


def _restore_from_log(log_path: Path) -> dict[str, str]:
    return _event_from_log(log_path, "restore")


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
        description="Sweep real vLLM KV connector restores across TurboBus modes"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", default="")
    parser.add_argument("--prompt-repeat", type=int, default=64)
    parser.add_argument("--second-prompt-suffix", default=" Italy")
    parser.add_argument("--prefix-key", default="qwen3-prefix")
    parser.add_argument("--restore-blocks-list", default="8")
    parser.add_argument("--tokens-per-block", type=int, default=16)
    parser.add_argument("--modes", default="auto,direct,relay,pool")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--min-pool-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-multiproc-executor", action="store_true")
    parser.add_argument("--no-map-physical-gpus", action="store_true")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()
    args.modes = parse_csv_strings(args.modes)
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

    results = []
    failed = False
    for restore_blocks in args.restore_blocks_list:
        matched_tokens = restore_blocks * args.tokens_per_block
        for mode in args.modes:
            log_path = log_dir / f"{mode}_blocks{restore_blocks}.log"
            print(
                "vllm_kv_connector_sweep run",
                f"mode={mode}",
                f"restore_blocks={restore_blocks}",
                f"matched_tokens={matched_tokens}",
                f"log={log_path}",
                flush=True,
            )
            result = run_case(args, mode, restore_blocks, matched_tokens, log_path)
            results.append(result)
            if result["returncode"] != 0:
                failed = True
                break
        if failed:
            break

    print_sweep_summary(args, results, summary_output)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
