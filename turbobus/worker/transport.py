from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any
from typing import Protocol, runtime_checkable
from threading import Event

from .codec import (
    WorkerMessageCodecError,
    decode_worker_observability_request_envelope,
)
from .endpoint import WorkerServiceEndpoint


@runtime_checkable
class WorkerServiceTransport(Protocol):
    def handle_message(self, message: str | bytes) -> str:
        raise NotImplementedError

    def handle_observability_message(self, message: str | bytes) -> str:
        raise NotImplementedError


@dataclass
class WorkerServiceLoopbackTransport:
    endpoint: WorkerServiceTransport

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, WorkerServiceTransport):
            raise TypeError("endpoint must be a WorkerServiceTransport")

    def handle_message(self, message: str | bytes) -> str:
        return self.endpoint.handle_message(message)

    def handle_observability_message(self, message: str | bytes) -> str:
        return self.endpoint.handle_observability_message(message)


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

    def handle_observability_message(self, message: str | bytes) -> str:
        return self._send(message)

    def serve_forever(self, stop_event: Event | None = None) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen()
        server.settimeout(0.1)

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    data = _read_message(conn)
                    if not data:
                        continue
                    response = self._handle_wire_message(data)
                    conn.sendall((response + "\n").encode("utf-8"))
        finally:
            server.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def _handle_wire_message(self, message: bytes) -> str:
        try:
            decode_worker_observability_request_envelope(message)
        except WorkerMessageCodecError:
            return self.endpoint.handle_message(message)
        return self.endpoint.handle_observability_message(message)

    def _send(self, message: str | bytes) -> str:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
            client.sendall(_ensure_bytes(message) + b"\n")
            data = _read_message(client)
        finally:
            client.close()
        return data.decode("utf-8")


__all__ = [
    "WorkerServiceLoopbackTransport",
    "WorkerServiceTransport",
    "WorkerServiceUnixSocketTransport",
]


def _ensure_bytes(message: str | bytes) -> bytes:
    if isinstance(message, bytes):
        return message
    if isinstance(message, str):
        return message.encode("utf-8")
    raise TypeError("message must be str or bytes")


def _read_message(conn: Any) -> bytes:
    data = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    return data.partition(b"\n")[0]
