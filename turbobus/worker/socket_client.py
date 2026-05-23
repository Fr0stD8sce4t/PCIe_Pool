from __future__ import annotations

import socket
from dataclasses import dataclass

from .codec import decode_worker_response_envelope, encode_worker_request_envelope
from .helper import (
    WorkerDataPlaneCompletionEnvelope,
    WorkerServiceRequestEnvelope,
)
from .transport import read_worker_socket_message


@dataclass
class WorkerServiceSocketClient:
    socket_path: str

    def __post_init__(self) -> None:
        socket_path = str(self.socket_path)
        if not socket_path.strip():
            raise ValueError("socket_path must be non-empty")
        self.socket_path = socket_path

    def submit_envelope(
        self,
        envelope: WorkerServiceRequestEnvelope,
    ) -> WorkerDataPlaneCompletionEnvelope:
        if not isinstance(envelope, WorkerServiceRequestEnvelope):
            raise TypeError("envelope must be a WorkerServiceRequestEnvelope")
        response_message = _send_worker_socket_message(
            self.socket_path,
            encode_worker_request_envelope(envelope),
        )
        response = decode_worker_response_envelope(response_message)
        if response.completion is None:
            return WorkerDataPlaneCompletionEnvelope(
                ok=response.ok,
                final_state=response.final_state,
                error=response.error,
            )
        completion = dict(response.completion)
        completion["ok"] = bool(response.ok and completion.get("ok", True))
        if response.final_state is not None and completion.get("final_state") is None:
            completion["final_state"] = response.final_state
        if response.error is not None and completion.get("error") is None:
            completion["error"] = response.error
        return WorkerDataPlaneCompletionEnvelope(**completion)


def _send_worker_socket_message(socket_path: str, message: str) -> str:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(socket_path)
        client.sendall(message.encode("utf-8") + b"\n")
        data = read_worker_socket_message(client)
    finally:
        client.close()
    return data.decode("utf-8")


__all__ = ["WorkerServiceSocketClient"]
