from __future__ import annotations

import json
import socket
from dataclasses import asdict

from ..transfer import TransferRequest
from .protocol import (
    DaemonRequest,
    DaemonResponse,
    RequestType,
    WorkerTransferAuthorizationRequest,
)


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

    def register_job(
        self,
        job_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        container_id: str | None = None,
        process_id: int | None = None,
    ) -> DaemonResponse:
        payload: dict[str, object] = {"job_id": str(job_id)}
        if user_id is not None:
            payload["user_id"] = str(user_id)
        if session_id is not None:
            payload["session_id"] = str(session_id)
        if container_id is not None:
            payload["container_id"] = str(container_id)
        if process_id is not None:
            payload["process_id"] = int(process_id)
        return self.send(
            DaemonRequest(
                request_type=RequestType.REGISTER_JOB,
                payload=payload,
            )
        )

    def register_buffer(
        self,
        buffer_id: str,
        job_id: str,
        kind: str,
        size_bytes: int,
        device_index: int | None = None,
        address: int | None = None,
        pinned: bool = False,
    ) -> DaemonResponse:
        payload: dict[str, object] = {
            "buffer_id": str(buffer_id),
            "job_id": str(job_id),
            "kind": str(kind),
            "size_bytes": int(size_bytes),
            "pinned": bool(pinned),
        }
        if device_index is not None:
            payload["device_index"] = int(device_index)
        if address is not None:
            payload["address"] = int(address)
        return self.send(
            DaemonRequest(
                request_type=RequestType.REGISTER_BUFFER,
                payload=payload,
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

    def plan_transfer(
        self,
        session_id: str,
        total_bytes: int,
        chunk_bytes: int,
        mode: str = "pool",
        direction: str = "h2d",
        job_id: str | None = None,
    ) -> DaemonResponse:
        request = TransferRequest(
            total_bytes=total_bytes,
            chunk_bytes=chunk_bytes,
            mode=mode,
            direction=direction,
            job_id=job_id,
        )
        return self.plan_transfer_request(session_id, request)

    def plan_transfer_request(
        self,
        session_id: str,
        request: TransferRequest,
        mode: str | None = None,
    ) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=str(session_id),
                payload=request.daemon_payload(mode=mode),
            )
        )

    def release_transfer(self, reservation_id: str) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.RELEASE_TRANSFER,
                payload={"reservation_id": str(reservation_id)},
            )
        )

    def cleanup(
        self,
        target_kind: str,
        target_id: str,
        reason: str = "manual",
        force: bool = False,
    ) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.CLEANUP,
                payload={
                    "target_kind": str(target_kind),
                    "target_id": str(target_id),
                    "reason": str(reason),
                    "force": bool(force),
                },
            )
        )

    def transfer_status(
        self,
        transfer_id: str,
        state: str | None = None,
        bytes_completed: int | None = None,
        error: str | None = None,
    ) -> DaemonResponse:
        payload: dict[str, object] = {"transfer_id": str(transfer_id)}
        if state is not None:
            payload["state"] = str(state)
        if bytes_completed is not None:
            payload["bytes_completed"] = int(bytes_completed)
        if error is not None:
            payload["error"] = str(error)
        return self.send(
            DaemonRequest(
                request_type=RequestType.TRANSFER_STATUS,
                payload=payload,
            )
        )

    def validate_lease(
        self,
        lease_id: str,
        token: str,
        session_id: str | None = None,
        relay_gpu: int | None = None,
        job_id: str | None = None,
        buffer_ids: list[str] | None = None,
    ) -> DaemonResponse:
        payload: dict[str, object] = {
            "lease_id": str(lease_id),
            "token": str(token),
        }
        if session_id is not None:
            payload["session_id"] = str(session_id)
        if relay_gpu is not None:
            payload["relay_gpu"] = int(relay_gpu)
        if job_id is not None:
            payload["job_id"] = str(job_id)
        if buffer_ids is not None:
            payload["buffer_ids"] = [str(buffer_id) for buffer_id in buffer_ids]
        return self.send(
            DaemonRequest(
                request_type=RequestType.VALIDATE_LEASE,
                payload=payload,
            )
        )

    def authorize_worker_transfer(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.AUTHORIZE_WORKER_TRANSFER,
                payload=asdict(request),
            )
        )

    def describe(self) -> DaemonResponse:
        return self.send(DaemonRequest(request_type=RequestType.PROFILE))

    def get_inventory(self) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.GET_INVENTORY,
            )
        )

    def discover_relays(
        self,
        target_gpu: int | None = None,
        relay_gpus: list[int] | None = None,
    ) -> DaemonResponse:
        payload: dict[str, object] = {}
        if target_gpu is not None:
            payload["target_gpu"] = int(target_gpu)
        if relay_gpus is not None:
            payload["relay_gpus"] = [int(gpu) for gpu in relay_gpus]
        return self.send(
            DaemonRequest(
                request_type=RequestType.DISCOVER_RELAYS,
                payload=payload,
            )
        )

    def get_profile(self, target_gpu: int, relay_gpus: list[int]) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.GET_PROFILE,
                payload={
                    "target_gpu": int(target_gpu),
                    "relay_gpus": [int(gpu) for gpu in relay_gpus],
                },
            )
        )

    def put_profile(
        self,
        target_gpu: int,
        relay_gpus: list[int],
        profile: dict,
        profile_bytes: int = 0,
    ) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.PUT_PROFILE,
                payload={
                    "target_gpu": int(target_gpu),
                    "relay_gpus": [int(gpu) for gpu in relay_gpus],
                    "profile": profile,
                    "profile_bytes": int(profile_bytes),
                },
            )
        )

    def invalidate_profile(self, target_gpu: int, relay_gpus: list[int]) -> DaemonResponse:
        return self.send(
            DaemonRequest(
                request_type=RequestType.INVALIDATE_PROFILE,
                payload={
                    "target_gpu": int(target_gpu),
                    "relay_gpus": [int(gpu) for gpu in relay_gpus],
                },
            )
        )
