from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from ..schema import WorkerDataPlaneRequest


class WorkerStagingPoolError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerStagingSlot:
    slot_id: str
    transfer_id: str
    lease_id: str
    session_id: str
    job_id: str
    relay_gpu: int
    requested_bytes: int
    allocated_bytes: int
    max_chunk_bytes: int
    chunk_count: int
    alignment_bytes: int
    buffer_kind: str
    active: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_request(
        cls,
        request: WorkerDataPlaneRequest,
        *,
        slot_id: str,
    ) -> "WorkerStagingSlot":
        if not isinstance(request, WorkerDataPlaneRequest):
            raise TypeError("request must be a WorkerDataPlaneRequest")
        staging = request.staging
        return cls(
            slot_id=slot_id,
            transfer_id=request.transfer_id,
            lease_id=request.lease_id,
            session_id=request.session_id,
            job_id=request.job_id,
            relay_gpu=request.relay_gpu,
            requested_bytes=staging.total_bytes,
            allocated_bytes=_align_up(staging.total_bytes, staging.alignment_bytes),
            max_chunk_bytes=staging.max_chunk_bytes,
            chunk_count=staging.chunk_count,
            alignment_bytes=staging.alignment_bytes,
            buffer_kind=staging.buffer_kind,
            metadata={
                "src_buffer_id": request.src_handle.buffer_id,
                "dst_buffer_id": request.dst_handle.buffer_id,
                "direction": request.direction,
            },
        )

    def __post_init__(self) -> None:
        if not str(self.slot_id).strip():
            raise ValueError("slot_id must be non-empty")
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        if not str(self.lease_id).strip():
            raise ValueError("lease_id must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        relay_gpu = int(self.relay_gpu)
        if relay_gpu < 0:
            raise ValueError("relay_gpu must be non-negative")
        requested_bytes = int(self.requested_bytes)
        allocated_bytes = int(self.allocated_bytes)
        max_chunk_bytes = int(self.max_chunk_bytes)
        chunk_count = int(self.chunk_count)
        alignment_bytes = int(self.alignment_bytes)
        if requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")
        if allocated_bytes < requested_bytes:
            raise ValueError("allocated_bytes cannot be smaller than requested_bytes")
        if max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        if chunk_count <= 0:
            raise ValueError("chunk_count must be positive")
        if alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be positive")
        if allocated_bytes % alignment_bytes != 0:
            raise ValueError("allocated_bytes must be alignment-sized")
        if not str(self.buffer_kind).strip():
            raise ValueError("buffer_kind must be non-empty")
        object.__setattr__(self, "slot_id", str(self.slot_id))
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "relay_gpu", relay_gpu)
        object.__setattr__(self, "requested_bytes", requested_bytes)
        object.__setattr__(self, "allocated_bytes", allocated_bytes)
        object.__setattr__(self, "max_chunk_bytes", max_chunk_bytes)
        object.__setattr__(self, "chunk_count", chunk_count)
        object.__setattr__(self, "alignment_bytes", alignment_bytes)
        object.__setattr__(self, "buffer_kind", str(self.buffer_kind))
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class WorkerStagingPool:
    def __init__(
        self,
        slot_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._slot_id_factory = slot_id_factory
        self._next_slot_id = 1
        self._slots: dict[str, WorkerStagingSlot] = {}

    def allocate(self, request: WorkerDataPlaneRequest) -> WorkerStagingSlot:
        slot_id = self._new_slot_id()
        if slot_id in self._slots:
            raise WorkerStagingPoolError("staging slot id already exists")
        slot = WorkerStagingSlot.from_request(request, slot_id=slot_id)
        self._slots[slot.slot_id] = slot
        return slot

    def describe(self, slot_id: str | None = None) -> dict[str, object]:
        if slot_id is not None:
            slot = self._slots.get(str(slot_id))
            if slot is None:
                raise WorkerStagingPoolError("unknown staging slot")
            return slot.as_dict()
        return {
            "active_slots": {
                slot_id: slot.as_dict() for slot_id, slot in sorted(self._slots.items())
            }
        }

    def validate_slot_for_request(
        self,
        slot_id: str,
        request: WorkerDataPlaneRequest,
    ) -> WorkerStagingSlot:
        if not isinstance(request, WorkerDataPlaneRequest):
            raise TypeError("request must be a WorkerDataPlaneRequest")
        slot = self._slots.get(str(slot_id))
        if slot is None:
            raise WorkerStagingPoolError("unknown staging slot")
        if slot.transfer_id != request.transfer_id:
            raise WorkerStagingPoolError("staging slot transfer mismatch")
        if slot.lease_id != request.lease_id:
            raise WorkerStagingPoolError("staging slot lease mismatch")
        if slot.session_id != request.session_id:
            raise WorkerStagingPoolError("staging slot session mismatch")
        if slot.job_id != request.job_id:
            raise WorkerStagingPoolError("staging slot job mismatch")
        if slot.relay_gpu != request.relay_gpu:
            raise WorkerStagingPoolError("staging slot relay mismatch")
        return slot

    def release(
        self,
        slot_id: str,
        request: WorkerDataPlaneRequest | None = None,
    ) -> WorkerStagingSlot:
        if request is not None:
            self.validate_slot_for_request(slot_id, request)
        slot = self._slots.pop(str(slot_id), None)
        if slot is None:
            raise WorkerStagingPoolError("unknown staging slot")
        return replace(slot, active=False)

    def _new_slot_id(self) -> str:
        if self._slot_id_factory is not None:
            slot_id = str(self._slot_id_factory())
            if not slot_id.strip():
                raise WorkerStagingPoolError("slot id factory returned an empty id")
            return slot_id
        slot_id = f"staging-{self._next_slot_id}"
        self._next_slot_id += 1
        return slot_id


def _align_up(value: int, alignment: int) -> int:
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


__all__ = [
    "WorkerStagingPool",
    "WorkerStagingPoolError",
    "WorkerStagingSlot",
]
