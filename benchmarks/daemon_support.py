from __future__ import annotations

from collections.abc import Iterable


def add_daemon_options(parser):
    parser.add_argument("--daemon-socket-path")
    parser.add_argument("--daemon-max-inflight-chunks", type=int, default=8)
    return parser


def runtime_options_kwargs(args) -> dict[str, object]:
    return {
        "daemon_socket_path": getattr(args, "daemon_socket_path", None),
        "daemon_max_inflight_chunks": int(
            getattr(args, "daemon_max_inflight_chunks", 8) or 8
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
