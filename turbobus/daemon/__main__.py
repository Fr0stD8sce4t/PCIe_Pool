from __future__ import annotations

import argparse
import json

from .server import TurboBusDaemon


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboBus daemon state preview")
    parser.add_argument("--relay-gpus", default="1", help="Comma-separated relay GPU ids")
    parser.add_argument("--max-sessions-per-relay", type=int, default=1)
    parser.add_argument("--max-inflight-chunks-per-relay", type=int, default=8)
    parser.add_argument("--session-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--profile-max-age-seconds", type=float, default=0.0)
    parser.add_argument(
        "--socket-path",
        default=None,
        help="Run the daemon server on this Unix socket path instead of printing state",
    )
    args = parser.parse_args()

    relays = [int(item) for item in args.relay_gpus.split(",") if item.strip()]
    daemon = TurboBusDaemon(
        relays,
        max_sessions_per_relay=args.max_sessions_per_relay,
        max_inflight_chunks_per_relay=args.max_inflight_chunks_per_relay,
        session_timeout_seconds=args.session_timeout_seconds,
        profile_max_age_seconds=args.profile_max_age_seconds,
    )
    if args.socket_path:
        daemon.serve_forever(args.socket_path)
        return
    print(json.dumps(daemon.describe().payload, indent=2))


if __name__ == "__main__":
    main()
