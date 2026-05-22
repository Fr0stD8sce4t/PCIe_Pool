from __future__ import annotations

import json
import socket
from dataclasses import asdict

from .protocol import DaemonRequest, DaemonResponse, RequestType


class TurboBusDaemonClient:
    def __init__(self, socket_path: str) -> None:
        self.socket_path = str(socket_path)

    def send(self, request: DaemonRequest) -> DaemonResponse:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
            client.sendall((json.dumps(asdict(request)) + "\n").encode("utf-8"))
            data = b""
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
        finally:
            client.close()

        response_data = json.loads(data.decode("utf-8"))
        return DaemonResponse(
            ok=bool(response_data["ok"]),
            payload=response_data.get("payload", {}),
            error=response_data.get("error"),
        )

    def register_session(
        self,
        target_gpu: int,
        relay_gpus: list[int],
        max_inflight_chunks: int = 8,
    ) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.REGISTER_SESSION,
                payload={
                    "target_gpu": int(target_gpu),
                    "relay_gpus": [int(gpu) for gpu in relay_gpus],
                    "max_inflight_chunks": int(max_inflight_chunks),
                },
            )
        )

    def close_session(self, session_id: str) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.CLOSE_SESSION,
                session_id=str(session_id),
            )
        )

    def reserve_transfer(
        self,
        session_id: str,
        relay_gpu: int,
        chunks: int,
        bytes_: int = 0,
        direction: str = "unknown",
    ) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.RESERVE_TRANSFER,
                session_id=str(session_id),
                payload={
                    "relay_gpu": int(relay_gpu),
                    "chunks": int(chunks),
                    "bytes": int(bytes_),
                    "direction": str(direction),
                },
            )
        )

    def release_transfer(self, reservation_id: str) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.RELEASE_TRANSFER,
                payload={"reservation_id": str(reservation_id)},
            )
        )
