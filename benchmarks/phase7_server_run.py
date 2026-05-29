from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BASELINE_POLICY = "paper-baseline"
TURBOBUS_POLICY = "turbobus-daemon"


def script_path(name: str) -> str:
    return str(Path("benchmarks") / name)


def as_path(path: str | Path) -> str:
    return str(path)


def run_id(server_class: str, policy: str) -> str:
    suffix = "baseline" if policy == BASELINE_POLICY else "turbobus"
    return f"phase7-{server_class}-{suffix}"


def paper_validation_command(
    *,
    server_class: str,
    policy: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    rid = run_id(server_class, policy)
    command = [
        sys.executable,
        script_path("paper_validation.py"),
        "--workloads",
        args.workloads,
        "--session-id",
        rid,
        "--job-id",
        rid,
        "--cpu-buffer-id",
        f"{rid}-cpu",
        "--gpu-buffer-id",
        f"{rid}-gpu",
        "--daemon-socket-path",
        args.daemon_socket_path,
        "--policy",
        policy,
        "--run-id",
        rid,
        "--bucket-count",
        str(args.bucket_count),
        "--bucket-bytes",
        str(args.bucket_bytes),
        "--chunk-bytes",
        str(args.chunk_bytes),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--vllm-model",
        args.vllm_model,
        "--vllm-job-count",
        str(args.vllm_job_count),
        "--vllm-restore-blocks",
        str(args.vllm_restore_blocks),
        "--vllm-matched-tokens",
        str(args.vllm_matched_tokens),
        "--vllm-prompt-repeat",
        str(args.vllm_prompt_repeat),
        "--output-dir",
        as_path(output_dir),
        "--json-output",
        as_path(output_dir / "result.json"),
        "--summary-output",
        as_path(output_dir / "summary.txt"),
    ]
    if args.vllm_enforce_eager:
        command.append("--vllm-enforce-eager")
    return command


def build_server_run_plan(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    server_root = output_root / args.server_class
    baseline_dir = server_root / BASELINE_POLICY
    turbobus_dir = server_root / TURBOBUS_POLICY
    comparison_path = server_root / "comparison.json"
    evidence_path = turbobus_dir / "evidence.json"
    bundle_gate_path = server_root / "bundle-gate.json"
    manifest_path = output_root / "acceptance-manifest.json"
    inventory_path = output_root / "acceptance-inventory.json"
    baseline_result = baseline_dir / "result.json"
    turbobus_result = turbobus_dir / "result.json"
    baseline_check = baseline_dir / "check.json"
    turbobus_check = turbobus_dir / "check.json"

    evidence_command = [
        sys.executable,
        script_path("phase7_evidence.py"),
        "--result",
        as_path(turbobus_result),
        "--comparison",
        as_path(comparison_path),
        "--json-output",
        as_path(evidence_path),
    ]
    if args.profile:
        evidence_command.extend(["--profile", args.profile])
    else:
        evidence_command.extend(["--daemon-socket-path", args.daemon_socket_path])

    bundle_command = [
        sys.executable,
        script_path("phase7_bundle_gate.py"),
        "--server-class",
        args.server_class,
        "--baseline-result",
        as_path(baseline_result),
        "--turbobus-result",
        as_path(turbobus_result),
        "--baseline-check",
        as_path(baseline_check),
        "--turbobus-check",
        as_path(turbobus_check),
        "--comparison",
        as_path(comparison_path),
        "--evidence",
        as_path(evidence_path),
        "--json-output",
        as_path(bundle_gate_path),
    ]
    if args.correctness:
        bundle_command.extend(["--correctness", args.correctness])

    ingest_command = [
        sys.executable,
        script_path("phase7_ingest_artifacts.py"),
        "--manifest",
        as_path(manifest_path),
        "--server-class",
        args.server_class,
        "--status",
        "accepted",
        "--bundle-gate",
        as_path(bundle_gate_path),
        "--real-artifacts",
        "--inventory-output",
        as_path(inventory_path),
    ]
    if args.allow_incomplete_inventory:
        ingest_command.append("--allow-incomplete-inventory")

    steps = [
        {
            "name": "baseline_paper_validation",
            "command": paper_validation_command(
                server_class=args.server_class,
                policy=BASELINE_POLICY,
                output_dir=baseline_dir,
                args=args,
            ),
            "outputs": [as_path(baseline_result), as_path(baseline_dir / "summary.txt")],
        },
        {
            "name": "baseline_result_check",
            "command": [
                sys.executable,
                script_path("phase7_result_check.py"),
                as_path(baseline_result),
                "--json-output",
                as_path(baseline_check),
            ],
            "outputs": [as_path(baseline_check)],
        },
        {
            "name": "turbobus_paper_validation",
            "command": paper_validation_command(
                server_class=args.server_class,
                policy=TURBOBUS_POLICY,
                output_dir=turbobus_dir,
                args=args,
            ),
            "outputs": [as_path(turbobus_result), as_path(turbobus_dir / "summary.txt")],
        },
        {
            "name": "turbobus_result_check",
            "command": [
                sys.executable,
                script_path("phase7_result_check.py"),
                as_path(turbobus_result),
                "--json-output",
                as_path(turbobus_check),
            ],
            "outputs": [as_path(turbobus_check)],
        },
        {
            "name": "comparison",
            "command": [
                sys.executable,
                script_path("phase7_compare.py"),
                "--baseline",
                as_path(baseline_result),
                "--turbobus",
                as_path(turbobus_result),
                "--json-output",
                as_path(comparison_path),
            ],
            "outputs": [as_path(comparison_path)],
        },
        {
            "name": "daemon_evidence",
            "command": evidence_command,
            "outputs": [as_path(evidence_path)],
        },
        {
            "name": "bundle_gate",
            "command": bundle_command,
            "outputs": [as_path(bundle_gate_path)],
        },
        {
            "name": "acceptance_ingest",
            "command": ingest_command,
            "outputs": [as_path(manifest_path), as_path(inventory_path)],
        },
    ]
    return {
        "ok": True,
        "server_class": args.server_class,
        "output_root": as_path(output_root),
        "steps": steps,
        "run_note": (
            "This command runs existing Phase 7 benchmark/evaluation tools. "
            "The workload commands submit public TransferIntent requests and "
            "read daemon TransferReceipt output; this runner does not select "
            "direct, relay, or pooled physical paths."
        ),
    }


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_plan(plan: dict) -> dict:
    completed = []
    for step in plan["steps"]:
        command = step["command"]
        result = subprocess.run(command, check=False)
        record = {
            "name": step["name"],
            "returncode": result.returncode,
            "command": command,
            "outputs": step.get("outputs", []),
        }
        completed.append(record)
        if result.returncode != 0:
            return {
                "ok": False,
                "failed_step": step["name"],
                "completed_steps": completed,
                "plan": plan,
            }
    return {
        "ok": True,
        "completed_steps": completed,
        "plan": plan,
    }


def selected_workloads(value: str) -> set[str]:
    items = {item.strip() for item in value.split(",") if item.strip()}
    if "all" in items:
        return {"model-loading", "training-offload", "optimizer-offload", "vllm-kv"}
    return items


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if "vllm-kv" in selected_workloads(args.workloads) and not args.vllm_model.strip():
        parser.error("--vllm-model is required for Phase 7 runs that include vLLM KV")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 7 server artifact chain for one server class",
    )
    parser.add_argument("--server-class", required=True, choices=("2gpu", "4gpu", "8gpu"))
    parser.add_argument("--output-root", default="benchmarks/results/phase7")
    parser.add_argument("--daemon-socket-path", default="/tmp/turbobusd.sock")
    parser.add_argument("--profile", help="optional saved daemon PROFILE JSON for evidence")
    parser.add_argument("--correctness", help="optional correctness gate JSON path")
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--bucket-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--vllm-model", default="")
    parser.add_argument("--vllm-job-count", type=int, default=1)
    parser.add_argument("--vllm-restore-blocks", type=int, default=8)
    parser.add_argument("--vllm-matched-tokens", type=int, default=128)
    parser.add_argument("--vllm-prompt-repeat", type=int, default=64)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--allow-incomplete-inventory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", help="optional JSON path for the command plan")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    plan = build_server_run_plan(args)
    if args.plan_output:
        write_json(args.plan_output, plan)
    if args.dry_run:
        sys.stdout.write(json.dumps(plan, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return
    result = execute_plan(plan)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
