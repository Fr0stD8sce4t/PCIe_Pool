from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from daemon_support import add_daemon_options

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPTS = {
    "bandwidth": REPO_ROOT / "benchmarks" / "bandwidth_pool.py",
    "model-loading": REPO_ROOT / "benchmarks" / "model_loading.py",
    "training-offload": REPO_ROOT / "benchmarks" / "training_offload.py",
}


def parse_relay_gpus(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def smoke_socket_path(prefix: str = "turbobusd-smoke") -> str:
    return str(Path(tempfile.gettempdir()) / f"{prefix}-{os.getpid()}.sock")


def build_daemon_command(args, socket_path: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "turbobus.daemon",
        "--socket-path",
        socket_path,
        "--relay-gpus",
        args.relay_gpus,
        "--max-sessions-per-relay",
        str(args.daemon_max_sessions_per_relay),
        "--max-inflight-chunks-per-relay",
        str(args.daemon_max_inflight_chunks_per_relay),
    ]
    session_timeout = float(getattr(args, "daemon_session_timeout_seconds", 0.0) or 0.0)
    profile_max_age = float(getattr(args, "daemon_profile_max_age_seconds", 0.0) or 0.0)
    if session_timeout > 0.0:
        command.extend(["--session-timeout-seconds", str(session_timeout)])
    if profile_max_age > 0.0:
        command.extend(["--profile-max-age-seconds", str(profile_max_age)])
    return command


def build_client_command(args, socket_path: str, workload: str, client_index: int) -> list[str]:
    script = BENCHMARK_SCRIPTS[workload]
    command = [sys.executable, str(script)]
    if workload == "bandwidth":
        command.extend(
            [
                "--target-gpu",
                str(args.target_gpu),
                "--relay-gpus",
                args.relay_gpus,
                "--bytes",
                str(args.bytes),
                "--chunk-bytes",
                str(args.chunk_bytes),
                "--profile-bytes",
                str(args.profile_bytes),
                "--min-pool-bytes",
                str(args.min_pool_bytes),
                "--mode",
                args.mode,
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
            ]
        )
    elif workload == "model-loading":
        command.extend(
            [
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
                "--mode",
                args.mode,
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
            ]
        )
    elif workload == "training-offload":
        command.extend(
            [
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
                "--mode",
                args.mode,
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
            ]
        )
    else:  # pragma: no cover - argparse constrains values
        raise ValueError(f"unsupported workload: {workload}")

    command.extend(
        [
            "--daemon-socket-path",
            socket_path,
            "--daemon-max-inflight-chunks",
            str(args.daemon_max_inflight_chunks),
        ]
    )
    if args.verify:
        command.append("--verify")
    if args.force_profile_first and client_index == 1:
        command.append("--force-profile")
    return command


def parse_status_line(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    head = parts[0]
    data: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key] = value
    return head, data


def collect_summary_fields(output: str) -> dict[str, list[dict[str, str]]]:
    profiles: list[dict[str, str]] = []
    reservations: list[dict[str, str]] = []
    for line in output.splitlines():
        head, data = parse_status_line(line)
        if head == "daemon_profile":
            profiles.append(data)
        elif head == "daemon_reservation":
            reservations.append(data)
    return {"daemon_profiles": profiles, "daemon_reservations": reservations}


def pick_profile(fields: dict[str, list[dict[str, str]]], phase: str) -> dict[str, str]:
    for item in fields.get("daemon_profiles", []):
        if item.get("phase") == phase:
            return item
    return {}


def first_reservation_entry(fields: dict[str, list[dict[str, str]]]) -> dict[str, str]:
    reservations = fields.get("daemon_reservations", [])
    return reservations[0] if reservations else {}


def wait_for_socket(socket_path: str, timeout_seconds: float = 10.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if not os.path.exists(socket_path):
            time.sleep(0.05)
            continue
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(0.2)
            client.connect(socket_path)
            return
        except Exception as exc:  # pragma: no cover - timing dependent
            last_error = exc
            time.sleep(0.05)
        finally:
            client.close()
    raise RuntimeError(
        f"daemon socket did not become ready: {socket_path}"
        + (f" ({last_error})" if last_error is not None else "")
    )


def run_client(command: list[str], label: str, env: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    print(f"client {label} command", " ".join(command))
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"client {label} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def build_smoke_result(
    first_output: str,
    second_output: str,
    workload: str,
) -> dict[str, object]:
    first_fields = collect_summary_fields(first_output)
    second_fields = collect_summary_fields(second_output)
    first_profile = pick_profile(first_fields, "after_profile")
    second_profile = pick_profile(second_fields, "initial")
    first_reservation = first_reservation_entry(first_fields)
    second_reservation = first_reservation_entry(second_fields)
    return {
        "workload": workload,
        "clients": {
            "first": {
                "daemon_profile": first_profile,
                "daemon_reservation": first_reservation,
            },
            "second": {
                "daemon_profile": second_profile,
                "daemon_reservation": second_reservation,
            },
        },
    }


def print_smoke_summary(result: dict[str, object]) -> None:
    print("DAEMON_SMOKE_SUMMARY_BEGIN")
    for label in ("first", "second"):
        client = result["clients"][label]
        profile = client.get("daemon_profile", {})
        reservation = client.get("daemon_reservation", {})
        fields = [
            f"client={label}",
            f"workload={result['workload']}",
        ]
        if profile:
            for key in ("phase", "daemon_profile_status", "daemon_profile_bytes"):
                if key in profile:
                    fields.append(f"{key}={profile[key]}")
        if reservation:
            for key in (
                "daemon_reservation_status",
                "daemon_reserved_relays",
                "daemon_reserved_chunks_per_relay",
                "mode",
            ):
                if key in reservation:
                    fields.append(f"{key}={reservation[key]}")
        print("daemon_smoke_client", *fields)
    print("DAEMON_SMOKE_SUMMARY_END")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TurboBus daemon benchmark smoke")
    parser.add_argument(
        "--workload",
        choices=tuple(BENCHMARK_SCRIPTS),
        default="bandwidth",
    )
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--relay-gpus", required=True)
    parser.add_argument("--mode", choices=["pool", "auto", "direct", "relay"], default="pool")
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--bucket-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--storage-layout", choices=["separate", "packed"], default="packed")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--profile-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--min-pool-bytes", type=int, default=6 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", action="store_false", dest="verify")
    parser.add_argument("--force-profile-first", action="store_true")
    parser.add_argument("--daemon-max-sessions-per-relay", type=int, default=2)
    parser.add_argument("--daemon-max-inflight-chunks-per-relay", type=int, default=128)
    parser.add_argument("--daemon-session-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--daemon-profile-max-age-seconds", type=float, default=0.0)
    add_daemon_options(parser)
    parser.set_defaults(daemon_max_inflight_chunks=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    socket_path = args.daemon_socket_path or smoke_socket_path()
    daemon_command = build_daemon_command(args, socket_path)
    env = os.environ.copy()
    python_path = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = python_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = python_path

    Path(socket_path).unlink(missing_ok=True)
    daemon = subprocess.Popen(
        daemon_command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_socket(socket_path)
        print("daemon", "socket_path", socket_path)
        print("daemon", "command", " ".join(daemon_command))

        first_command = build_client_command(args, socket_path, args.workload, 1)
        first_output = run_client(first_command, "first", env)

        second_command = build_client_command(args, socket_path, args.workload, 2)
        second_output = run_client(second_command, "second", env)

        result = build_smoke_result(first_output, second_output, args.workload)
        print_smoke_summary(result)
    finally:
        daemon.terminate()
        try:
            stdout, stderr = daemon.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()
            stdout, stderr = daemon.communicate(timeout=10)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
        Path(socket_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
