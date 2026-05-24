from __future__ import annotations

import argparse

from turbobus import TurboBusClient, TransferIntent, WorkloadKind


def build_intent(args) -> TransferIntent:
    return TransferIntent(
        intent_id=args.intent_id,
        job_id=args.job_id,
        session_id=args.session_id,
        source_buffer_id=args.source_buffer_id,
        destination_buffer_id=args.destination_buffer_id,
        direction=args.direction,
        total_bytes=args.bytes,
        ranges=(
            {
                "src_offset": args.src_offset,
                "dst_offset": args.dst_offset,
                "bytes": args.bytes,
            },
        ),
        workload_kind=WorkloadKind.GENERIC,
        policy_hints={},
        metadata={"example": "torch-tensor-fetch", "policy": args.policy},
    )


def receipt_line(receipt) -> str:
    direct_bytes = 0
    relay_bytes = 0
    for path in receipt.path_stats:
        bytes_count = int(path.get("bytes", 0) or 0)
        if str(path.get("kind", "")).lower() == "relay":
            relay_bytes += bytes_count
        else:
            direct_bytes += bytes_count
    return (
        "daemon_receipt "
        f"intent_id={receipt.intent_id} "
        f"decision_id={receipt.decision_id} "
        f"topology_snapshot_id={receipt.topology_snapshot_id} "
        f"ticket_id={receipt.ticket_id} "
        f"state={receipt.state.value} "
        f"bytes_total={receipt.bytes_total} "
        f"bytes_completed={receipt.bytes_completed} "
        f"direct_bytes={direct_bytes} "
        f"relay_bytes={relay_bytes}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit one generic TransferIntent through the public TurboBus client"
    )
    parser.add_argument("--daemon-socket-path", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--job-id", default="example-torch-tensor-fetch")
    parser.add_argument("--intent-id", default="example-torch-tensor-fetch-0")
    parser.add_argument("--source-buffer-id", required=True)
    parser.add_argument("--destination-buffer-id", required=True)
    parser.add_argument("--direction", choices=["h2d", "d2h"], default="h2d")
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--src-offset", type=int, default=0)
    parser.add_argument("--dst-offset", type=int, default=0)
    parser.add_argument("--policy", default="daemon-default")
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    return parser


def validate_args(args) -> None:
    if args.bytes <= 0:
        raise ValueError("--bytes must be positive")
    if args.src_offset < 0:
        raise ValueError("--src-offset must be non-negative")
    if args.dst_offset < 0:
        raise ValueError("--dst-offset must be non-negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    client = TurboBusClient(socket_path=args.daemon_socket_path)
    intent = build_intent(args)
    receipt = client.submit_transfer_intent(intent)
    if args.wait_timeout_seconds is not None:
        receipt = client.wait_transfer_receipt(
            intent.intent_id,
            timeout_seconds=args.wait_timeout_seconds,
        )
    print(receipt_line(receipt))


if __name__ == "__main__":
    main()
