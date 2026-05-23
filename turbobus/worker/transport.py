from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any
from threading import Event

from .endpoint import WorkerServiceEndpoint


@dataclass
class WorkerServiceUnixSocketTransport:
    endpoint: WorkerServiceEndpoint
    socket_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, WorkerServiceEndpoint):
            raise TypeError("endpoint must be a WorkerServiceEndpoint")
        socket_path = str(self.socket_path)
        if not socket_path.strip():
            raise ValueError("socket_path must be non-empty")
        self.socket_path = socket_path

    def handle_message(self, message: str | bytes) -> str:
        return self._send(message)

    def serve_forever(
        self,
        stop_event: Event | None = None,
        max_requests: int | None = None,
    ) -> None:
        if max_requests is not None:
            max_requests = int(max_requests)
            if max_requests <= 0:
                raise ValueError("max_requests must be positive")
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen()
        server.settimeout(0.1)

        try:
            request_count = 0
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if max_requests is not None and request_count >= max_requests:
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    data = read_worker_socket_message(conn)
                    if not data:
                        continue
                    response = self.endpoint.handle_message(data)
                    conn.sendall((response + "\n").encode("utf-8"))
                    request_count += 1
        finally:
            server.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def _send(self, message: str | bytes) -> str:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
            client.sendall(_ensure_bytes(message) + b"\n")
            data = read_worker_socket_message(client)
        finally:
            client.close()
        return data.decode("utf-8")


__all__ = [
    "WorkerServiceUnixSocketTransport",
    "read_worker_socket_message",
]


def _ensure_bytes(message: str | bytes) -> bytes:
    if isinstance(message, bytes):
        return message
    if isinstance(message, str):
        return message.encode("utf-8")
    raise TypeError("message must be str or bytes")


def read_worker_socket_message(conn: Any) -> bytes:
    data = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    return data.partition(b"\n")[0]
