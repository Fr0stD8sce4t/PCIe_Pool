from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from enum import Enum
import os
import uuid

from turbobus import TurboBusClient, TransferIntent, TransferReceipt, WorkloadKind


def add_daemon_options(parser):
    parser.add_argument("--daemon-socket-path")
    parser.add_argument("--daemon-max-inflight-chunks", type=int, default=8)
    parser.add_argument("--daemon-profile-max-age-seconds", type=float, default=3600.0)
    return parser


def runtime_options_kwargs(args) -> dict[str, object]:
    return {
        "daemon_socket_path": getattr(args, "daemon_socket_path", None),
        "daemon_max_inflight_chunks": int(
            getattr(args, "daemon_max_inflight_chunks", 8) or 8
        ),
        "daemon_profile_max_age_seconds": float(
            getattr(args, "daemon_profile_max_age_seconds", 3600.0) or 3600.0
        ),
    }


def daemon_profile_summary(runtime) -> dict[str, object]:
    return dict(getattr(runtime, "last_daemon_profile_dict", lambda: {})() or {})


def collect_daemon_reservation_info(handles: Iterable) -> dict[str, object]:
    seen = set()
    for handle in handles:
        key = id(handle)
        if key in seen:
            continue
        seen.add(key)
        info = getattr(handle, "daemon_reservation_info", None)
        if info:
            return dict(info)
    return {}


def make_benchmark_transfer_intent(
    *,
    workload_kind: WorkloadKind | str,
    job_id: str,
    session_id: str,
    source_buffer_id: str,
    destination_buffer_id: str,
    direction: str,
    total_bytes: int,
    ranges: Iterable[dict[str, int]],
    priority: int = 0,
    policy_hints: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    intent_id: str | None = None,
) -> TransferIntent:
    return TransferIntent(
        intent_id=intent_id or f"benchmark-intent-{uuid.uuid4()}",
        job_id=str(job_id),
        session_id=str(session_id),
        source_buffer_id=str(source_buffer_id),
        destination_buffer_id=str(destination_buffer_id),
        direction=str(direction),
        total_bytes=int(total_bytes),
        ranges=tuple(dict(item) for item in ranges),
        workload_kind=workload_kind,
        priority=int(priority),
        policy_hints={} if policy_hints is None else dict(policy_hints),
        metadata={} if metadata is None else dict(metadata),
    )


def submit_benchmark_transfer_intent(
    intent: TransferIntent,
    *,
    daemon_socket_path: str,
) -> TransferReceipt:
    client = TurboBusClient(socket_path=daemon_socket_path)
    return client.submit_transfer_intent(intent)


def receipt_to_trace(receipt: TransferReceipt) -> dict[str, object]:
    trace = _json_ready(asdict(receipt))
    direct_bytes = 0
    relay_bytes = 0
    direct_chunks = 0
    relay_chunks = 0
    for path in receipt.path_stats:
        bytes_count = int(path.get("bytes", 0) or 0)
        chunks = int(path.get("chunk_count", path.get("chunks", 0)) or 0)
        if str(path.get("kind", "")).lower() == "relay":
            relay_bytes += bytes_count
            relay_chunks += chunks
        else:
            direct_bytes += bytes_count
            direct_chunks += chunks
    trace.update(
        {
            "direct_bytes": direct_bytes,
            "relay_bytes": relay_bytes,
            "direct_chunks": direct_chunks,
            "relay_chunks": relay_chunks,
            "path_split": {
                "direct_bytes": direct_bytes,
                "relay_bytes": relay_bytes,
                "direct_chunks": direct_chunks,
                "relay_chunks": relay_chunks,
            },
            "fallback_reason": str(receipt.metadata.get("fallback_reason", "") or ""),
        }
    )
    return trace


def receipt_trace_line(receipt: TransferReceipt, *, prefix: str = "daemon_receipt") -> str:
    trace = receipt_to_trace(receipt)
    fields = [
        prefix,
        f"intent_id={trace['intent_id']}",
        f"decision_id={trace['decision_id']}",
        f"topology_snapshot_id={trace['topology_snapshot_id']}",
        f"ticket_id={trace['ticket_id']}",
        f"state={receipt.state.value}",
        f"bytes_total={trace['bytes_total']}",
        f"bytes_completed={trace['bytes_completed']}",
        f"direct_bytes={trace['direct_bytes']}",
        f"relay_bytes={trace['relay_bytes']}",
        f"direct_chunks={trace['direct_chunks']}",
        f"relay_chunks={trace['relay_chunks']}",
    ]
    fallback_reason = trace.get("fallback_reason")
    if fallback_reason:
        fields.append(f"fallback_reason={str(fallback_reason).replace(' ', '_')}")
    return " ".join(fields)


def benchmark_job_id(workload: str) -> str:
    return f"benchmark-{workload}-{os.getpid()}"


def daemon_profile_line(profile: dict[str, object]) -> str:
    return _format_status_line("daemon_profile", profile)


def daemon_reservation_line(reservation: dict[str, object]) -> str:
    return _format_status_line("daemon_reservation", reservation)


def _format_status_line(prefix: str, data: dict[str, object]) -> str:
    if not data:
        return ""
    fields = []
    for key in sorted(data):
        value = data[key]
        if value in (None, ""):
            continue
        fields.append(f"{key}={value}")
    if not fields:
        return ""
    return " ".join([prefix, *fields])


def _json_ready(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
