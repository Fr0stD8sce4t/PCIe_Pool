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


def print_sweep_summary(args, results) -> None:
    case_rows = []
    print("SWEEP_SUMMARY_BEGIN")
    print(
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
    )
    for result in results:
        summary = result["summary"]
        config = summary.get("vllm_kv_connector_config", {})
        save = summary.get("vllm_kv_connector_save", {})
        restore = _restore_from_log(Path(result["log_path"]))
        output = summary.get("vllm_kv_connector_result", {})
        row = {
            "mode": result["mode"],
            "restore_blocks": result["restore_blocks"],
            "matched_tokens": result["matched_tokens"],
            "returncode": result["returncode"],
            "save_ms": save.get("elapsed_ms", "NA"),
            "restore_ms": restore.get("elapsed_ms", "NA"),
            "bytes": restore.get("bytes", save.get("bytes", "NA")),
            "direct_chunks": restore.get("direct_chunks", "NA"),
            "relay_chunks": restore.get("relay_chunks", "NA"),
            "prompt_tokens": output.get("prompt_tokens", "NA"),
            "shared_prefix": output.get("shared_prefix", "NA"),
            "child_mode": config.get("mode", result["mode"]),
            "log": result["log_path"],
        }
        row["restore_gib_s"] = gib_per_second(row["bytes"], row["restore_ms"])
        case_rows.append(row)
        print(
            "vllm_kv_connector_sweep_case",
            f"mode={row['mode']}",
            f"restore_blocks={row['restore_blocks']}",
            f"matched_tokens={row['matched_tokens']}",
            f"returncode={row['returncode']}",
            f"save_ms={row['save_ms']}",
            f"restore_ms={row['restore_ms']}",
            f"restore_gib_s={row['restore_gib_s']}",
            f"bytes={row['bytes']}",
            f"direct_chunks={row['direct_chunks']}",
            f"relay_chunks={row['relay_chunks']}",
            f"prompt_tokens={row['prompt_tokens']}",
            f"shared_prefix={row['shared_prefix']}",
            f"child_mode={row['child_mode']}",
            f"log={row['log']}",
        )
    _print_speedups(case_rows)
    print("SWEEP_SUMMARY_END")


def _print_speedups(case_rows) -> None:
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
        print(
            "vllm_kv_connector_sweep_speedup",
            f"restore_blocks={restore_blocks}",
            f"direct_over_pool_restore={speedup(direct['restore_ms'], pool['restore_ms']) if direct else 'NA'}",
            f"relay_over_pool_restore={speedup(relay['restore_ms'], pool['restore_ms']) if relay else 'NA'}",
        )


def _restore_from_log(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "turbobus_kv_connector_event event=restore " not in line:
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
    parser.add_argument("--modes", default="direct,relay,pool")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-multiproc-executor", action="store_true")
    parser.add_argument("--no-map-physical-gpus", action="store_true")
    parser.add_argument("--log-dir", default=None)
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

    print_sweep_summary(args, results)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
