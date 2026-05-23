from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping

from ..schema import BufferRegistration, WorkerTransferAuthorization


class WorkerTransferState(str, Enum):
    UNSUPPORTED = "unsupported"


class UnsupportedWorkerExecution(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerTransferRequest:
    authorization: WorkerTransferAuthorization

    @classmethod
    def from_authorization_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "WorkerTransferRequest":
        authorization_payload = payload.get("authorization", payload)
        if not isinstance(authorization_payload, Mapping):
            raise ValueError("authorization payload must be a mapping")
        return cls(
            authorization=WorkerTransferAuthorization(
                transfer_id=str(authorization_payload["transfer_id"]),
                lease_id=str(authorization_payload["lease_id"]),
                session_id=str(authorization_payload["session_id"]),
                job_id=str(authorization_payload["job_id"]),
                src_buffer=_buffer_from_payload(authorization_payload["src_buffer"]),
                dst_buffer=_buffer_from_payload(authorization_payload["dst_buffer"]),
                direction=str(authorization_payload["direction"]),
                ranges=tuple(authorization_payload.get("ranges", ())),
                relay_gpu=authorization_payload.get("relay_gpu"),
            )
        )

    @property
    def transfer_id(self) -> str:
        return self.authorization.transfer_id

    def as_dict(self) -> dict[str, object]:
        return {"authorization": asdict(self.authorization)}


@dataclass(frozen=True)
class WorkerTransferResult:
    transfer_id: str
    state: WorkerTransferState
    error: str | None = None
    bytes_completed: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        bytes_completed = int(self.bytes_completed)
        if bytes_completed < 0:
            raise ValueError("bytes_completed must be non-negative")
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "state", WorkerTransferState(self.state))
        object.__setattr__(self, "bytes_completed", bytes_completed)
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return {
            "transfer_id": self.transfer_id,
            "state": self.state.value,
            "error": self.error,
            "bytes_completed": self.bytes_completed,
            "metadata": dict(self.metadata),
        }


class WorkerTransferUnsupportedExecutor:
    def execute(self, request: WorkerTransferRequest) -> WorkerTransferResult:
        if not isinstance(request, WorkerTransferRequest):
            raise TypeError("request must be a WorkerTransferRequest")
        return WorkerTransferResult(
            transfer_id=request.transfer_id,
            state=WorkerTransferState.UNSUPPORTED,
            error="worker execution is not implemented yet",
            bytes_completed=0,
            metadata={
                "relay_gpu": request.authorization.relay_gpu,
                "src_buffer_id": request.authorization.src_buffer.buffer_id,
                "dst_buffer_id": request.authorization.dst_buffer.buffer_id,
            },
        )

    def execute_or_raise(self, request: WorkerTransferRequest) -> WorkerTransferResult:
        result = self.execute(request)
        raise UnsupportedWorkerExecution(result.error or "worker execution is unsupported")


def _buffer_from_payload(payload: object) -> BufferRegistration:
    if not isinstance(payload, Mapping):
        raise ValueError("buffer payload must be a mapping")
    return BufferRegistration(
        buffer_id=str(payload["buffer_id"]),
        job_id=str(payload["job_id"]),
        kind=str(payload["kind"]),
        size_bytes=int(payload["size_bytes"]),
        device_index=payload.get("device_index"),
        address=payload.get("address"),
        pinned=bool(payload.get("pinned", False)),
    )
