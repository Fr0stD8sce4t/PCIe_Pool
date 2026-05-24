from __future__ import annotations

import argparse
import json

from .startup import DaemonStartupConfig, DaemonStartupError, create_production_daemon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TurboBus daemon state preview")
    parser.add_argument("--topology-provider", default="cuda-nvml")
    parser.add_argument("--target-gpu", type=int, required=True)
    parser.add_argument("--min-relays", type=int, default=1)
    parser.add_argument(
        "--allow-missing-fabric",
        action="store_true",
        help="Start even when the production provider cannot report GPU fabric links.",
    )
    parser.add_argument(
        "--allow-missing-pcie",
        action="store_true",
        help="Start even when the production provider cannot report PCIe paths.",
    )
    parser.add_argument("--max-sessions-per-relay", type=int, default=1)
    parser.add_argument("--max-inflight-chunks-per-relay", type=int, default=8)
    parser.add_argument("--session-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--profile-max-age-seconds", type=float, default=0.0)
    parser.add_argument(
        "--socket-path",
        default=None,
        help="Run the daemon server on this Unix socket path instead of printing state",
    )
    return parser


def startup_config_from_args(args) -> DaemonStartupConfig:
    return DaemonStartupConfig(
        topology_provider=args.topology_provider,
        target_gpu=args.target_gpu,
        min_relay_count=args.min_relays,
        require_fabric=not args.allow_missing_fabric,
        require_pcie=not args.allow_missing_pcie,
        max_sessions_per_relay=args.max_sessions_per_relay,
        max_inflight_chunks_per_relay=args.max_inflight_chunks_per_relay,
        session_timeout_seconds=args.session_timeout_seconds,
        profile_max_age_seconds=args.profile_max_age_seconds,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        daemon = create_production_daemon(startup_config_from_args(args))
    except DaemonStartupError as exc:
        parser.exit(2, f"turbobus daemon startup failed: {exc}\n")
    if args.socket_path:
        daemon.serve_forever(args.socket_path)
        return
    print(json.dumps(daemon.describe().payload, indent=2))


if __name__ == "__main__":
    main()
