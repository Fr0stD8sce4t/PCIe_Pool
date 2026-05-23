from __future__ import annotations

import argparse
from threading import Event
from typing import Sequence

from ..daemon import TurboBusDaemonClient
from .endpoint import WorkerServiceEndpoint
from .transport import WorkerServiceUnixSocketTransport


def build_worker_helper_transport(
    daemon_socket_path: str,
    socket_path: str,
) -> WorkerServiceUnixSocketTransport:
    daemon_client = TurboBusDaemonClient(str(daemon_socket_path))
    endpoint = WorkerServiceEndpoint(daemon_client=daemon_client)
    return WorkerServiceUnixSocketTransport(
        endpoint=endpoint,
        socket_path=str(socket_path),
    )


def run_worker_helper_process(
    daemon_socket_path: str,
    socket_path: str,
    stop_event: Event | None = None,
    max_requests: int | None = None,
) -> None:
    transport = build_worker_helper_transport(daemon_socket_path, socket_path)
    transport.serve_forever(stop_event=stop_event, max_requests=max_requests)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TurboBus worker helper process",
    )
    parser.add_argument(
        "--daemon-socket-path",
        required=True,
        help="Unix socket path for the daemon control plane",
    )
    parser.add_argument(
        "--socket-path",
        required=True,
        help="Unix socket path for the worker helper service",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after this many requests; mainly useful for smoke tests",
    )
    args = parser.parse_args(argv)
    run_worker_helper_process(
        args.daemon_socket_path,
        args.socket_path,
        max_requests=args.max_requests,
    )
    return 0


__all__ = [
    "build_worker_helper_transport",
    "main",
    "run_worker_helper_process",
]
