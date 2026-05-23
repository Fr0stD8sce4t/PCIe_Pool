from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TransferMode(str, Enum):
    AUTO = "auto"
    POOL = "pool"
    DIRECT = "direct"
    RELAY = "relay"


@dataclass(frozen=True)
class AutoTransferDecision:
    requested_mode: TransferMode
    resolved_mode: TransferMode
    request_bytes: int
    request_chunks: int
    direct_h2d_bw_gbps: float
    relay_effective_bw_gbps: float
    eligible_relay_devices: tuple[int, ...]
    reason: str


class RequestType(str, Enum):
    REGISTER_JOB = "REGISTER_JOB"
    REGISTER_SESSION = "REGISTER_SESSION"
    REGISTER_BUFFER = "REGISTER_BUFFER"
    GET_INVENTORY = "GET_INVENTORY"
    DISCOVER_RELAYS = "DISCOVER_RELAYS"
    REAP_EXPIRED_LEASES = "REAP_EXPIRED_LEASES"
    PROFILE = "PROFILE"
    GET_PROFILE = "GET_PROFILE"
    PUT_PROFILE = "PUT_PROFILE"
    INVALIDATE_PROFILE = "INVALIDATE_PROFILE"
    PLAN_TRANSFER = "PLAN_TRANSFER"
    TRANSFER_STATUS = "TRANSFER_STATUS"
    RESERVE_TRANSFER = "RESERVE_TRANSFER"
    ISSUE_LEASE = "ISSUE_LEASE"
    VALIDATE_LEASE = "VALIDATE_LEASE"
    AUTHORIZE_WORKER_TRANSFER = "AUTHORIZE_WORKER_TRANSFER"
    RELEASE_TRANSFER = "RELEASE_TRANSFER"
    CLEANUP = "CLEANUP"
    CLOSE_SESSION = "CLOSE_SESSION"


class TransferStatusState(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class JobIdentity:
    job_id: str
    user_id: str | None = None
    session_id: str | None = None
    container_id: str | None = None
    process_id: int | None = None

    def __post_init__(self) -> None:
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        if self.process_id is not None and int(self.process_id) < 0:
            raise ValueError("process_id must be non-negative")
        object.__setattr__(self, "job_id", str(self.job_id))
        if self.user_id is not None:
            object.__setattr__(self, "user_id", str(self.user_id))
        if self.session_id is not None:
            object.__setattr__(self, "session_id", str(self.session_id))
        if self.container_id is not None:
            object.__setattr__(self, "container_id", str(self.container_id))
        if self.process_id is not None:
            object.__setattr__(self, "process_id", int(self.process_id))


@dataclass(frozen=True)
class BufferRegistration:
    buffer_id: str
    job_id: str
    kind: str
    size_bytes: int
    device_index: int | None = None
    address: int | None = None
    pinned: bool = False
    handle_type: str = "registered_buffer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.buffer_id).strip():
            raise ValueError("buffer_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        size_bytes = int(self.size_bytes)
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.device_index is not None and int(self.device_index) < 0:
            raise ValueError("device_index must be non-negative")
        if self.address is not None and int(self.address) < 0:
            raise ValueError("address must be non-negative")
        if not str(self.kind).strip():
            raise ValueError("kind must be non-empty")
        handle_type = str(self.handle_type).lower()
        metadata = _normalize_buffer_handle_metadata(
            kind=str(self.kind),
            pinned=bool(self.pinned),
            device_index=self.device_index,
            size_bytes=size_bytes,
            handle_type=handle_type,
            metadata=self.metadata,
        )
        object.__setattr__(self, "buffer_id", str(self.buffer_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "handle_type", handle_type)
        object.__setattr__(self, "metadata", metadata)
        if self.device_index is not None:
            object.__setattr__(self, "device_index", int(self.device_index))
        if self.address is not None:
            object.__setattr__(self, "address", int(self.address))


@dataclass(frozen=True)
class LeaseToken:
    lease_id: str
    session_id: str
    relay_gpu: int
    token: str
    buffer_ids: tuple[str, ...] = field(default_factory=tuple)
    job_id: str | None = None
    issued_at: float = 0.0
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.lease_id).strip():
            raise ValueError("lease_id must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")
        if int(self.relay_gpu) < 0:
            raise ValueError("relay_gpu must be non-negative")
        token = str(self.token)
        if not token.strip():
            raise ValueError("token must be non-empty")
        buffer_ids = tuple(str(buffer_id) for buffer_id in self.buffer_ids)
        if any(not buffer_id.strip() for buffer_id in buffer_ids):
            raise ValueError("buffer_ids must be non-empty when provided")
        issued_at = float(self.issued_at)
        expires_at = float(self.expires_at)
        if expires_at and expires_at < issued_at:
            raise ValueError("expires_at must not be earlier than issued_at")
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "relay_gpu", int(self.relay_gpu))
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "buffer_ids", buffer_ids)
        if self.job_id is not None:
            object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True)
class TransferStatus:
    transfer_id: str
    job_id: str
    state: TransferStatusState
    bytes_total: int
    bytes_completed: int = 0
    session_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        bytes_total = int(self.bytes_total)
        bytes_completed = int(self.bytes_completed)
        if bytes_total < 0:
            raise ValueError("bytes_total must be non-negative")
        if bytes_completed < 0:
            raise ValueError("bytes_completed must be non-negative")
        if bytes_completed > bytes_total:
            raise ValueError("bytes_completed cannot exceed bytes_total")
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "state", TransferStatusState(self.state))
        object.__setattr__(self, "bytes_total", bytes_total)
        object.__setattr__(self, "bytes_completed", bytes_completed)
        if self.session_id is not None:
            object.__setattr__(self, "session_id", str(self.session_id))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))


@dataclass(frozen=True)
class WorkerTransferAuthorizationRequest:
    transfer_id: str
    lease_id: str
    token: str
    session_id: str
    job_id: str
    src_buffer_id: str
    dst_buffer_id: str
    direction: str
    ranges: tuple[dict[str, int], ...] = field(default_factory=tuple)
    relay_gpu: int | None = None

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        if not str(self.lease_id).strip():
            raise ValueError("lease_id must be non-empty")
        if not str(self.token).strip():
            raise ValueError("token must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        if not str(self.src_buffer_id).strip():
            raise ValueError("src_buffer_id must be non-empty")
        if not str(self.dst_buffer_id).strip():
            raise ValueError("dst_buffer_id must be non-empty")
        direction = str(self.direction).lower()
        if direction not in {"h2d", "d2h"}:
            raise ValueError("direction must be h2d or d2h")
        normalized_ranges = _normalize_worker_ranges(self.ranges)
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "token", str(self.token))
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "src_buffer_id", str(self.src_buffer_id))
        object.__setattr__(self, "dst_buffer_id", str(self.dst_buffer_id))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "ranges", tuple(normalized_ranges))
        if self.relay_gpu is not None:
            object.__setattr__(self, "relay_gpu", int(self.relay_gpu))
            if self.relay_gpu < 0:
                raise ValueError("relay_gpu must be non-negative")


@dataclass(frozen=True)
class WorkerTransferAuthorization:
    transfer_id: str
    lease_id: str
    session_id: str
    job_id: str
    src_buffer: BufferRegistration
    dst_buffer: BufferRegistration
    direction: str
    ranges: tuple[dict[str, int], ...] = field(default_factory=tuple)
    relay_gpu: int | None = None
    plan: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        if not str(self.lease_id).strip():
            raise ValueError("lease_id must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        direction = str(self.direction).lower()
        if direction not in {"h2d", "d2h"}:
            raise ValueError("direction must be h2d or d2h")
        normalized_ranges = _normalize_worker_ranges(self.ranges)
        if self.src_buffer.job_id != str(self.job_id):
            raise ValueError("src buffer job does not match authorization job")
        if self.dst_buffer.job_id != str(self.job_id):
            raise ValueError("dst buffer job does not match authorization job")
        if not isinstance(self.plan, Mapping):
            raise TypeError("plan must be a mapping")
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "ranges", tuple(normalized_ranges))
        object.__setattr__(self, "plan", dict(self.plan))
        if self.relay_gpu is not None:
            object.__setattr__(self, "relay_gpu", int(self.relay_gpu))
            if self.relay_gpu < 0:
                raise ValueError("relay_gpu must be non-negative")


@dataclass(frozen=True)
class WorkerBufferHandle:
    buffer_id: str
    job_id: str
    kind: str
    size_bytes: int
    access: str
    device_index: int | None = None
    address: int | None = None
    pinned: bool = False
    handle_type: str = "registered_buffer"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_buffer_registration(
        cls,
        buffer: BufferRegistration,
        access: str,
    ) -> "WorkerBufferHandle":
        if not isinstance(buffer, BufferRegistration):
            raise TypeError("buffer must be a BufferRegistration")
        return cls(
            buffer_id=buffer.buffer_id,
            job_id=buffer.job_id,
            kind=buffer.kind,
            size_bytes=buffer.size_bytes,
            access=access,
            device_index=buffer.device_index,
            address=buffer.address,
            pinned=buffer.pinned,
            handle_type=buffer.handle_type,
            metadata=buffer.metadata,
        )

    def __post_init__(self) -> None:
        if not str(self.buffer_id).strip():
            raise ValueError("buffer_id must be non-empty")
        if not str(self.job_id).strip():
            raise ValueError("job_id must be non-empty")
        if not str(self.kind).strip():
            raise ValueError("kind must be non-empty")
        size_bytes = int(self.size_bytes)
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        access = str(self.access).lower()
        if access not in {"read", "write", "read_write"}:
            raise ValueError("access must be read, write, or read_write")
        handle_type = str(self.handle_type).lower()
        if not handle_type.strip():
            raise ValueError("handle_type must be non-empty")
        if self.device_index is not None and int(self.device_index) < 0:
            raise ValueError("device_index must be non-negative")
        if self.address is not None and int(self.address) < 0:
            raise ValueError("address must be non-negative")
        metadata = _normalize_buffer_handle_metadata(
            kind=str(self.kind),
            pinned=bool(self.pinned),
            device_index=self.device_index,
            size_bytes=size_bytes,
            handle_type=handle_type,
            metadata=self.metadata,
        )
        object.__setattr__(self, "buffer_id", str(self.buffer_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "access", access)
        object.__setattr__(self, "handle_type", handle_type)
        object.__setattr__(self, "metadata", metadata)
        if self.device_index is not None:
            object.__setattr__(self, "device_index", int(self.device_index))
        if self.address is not None:
            object.__setattr__(self, "address", int(self.address))


@dataclass(frozen=True)
class WorkerStagingBufferRequirement:
    relay_gpu: int
    total_bytes: int
    max_chunk_bytes: int
    chunk_count: int
    alignment_bytes: int = 256
    buffer_kind: str = "relay_staging"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relay_gpu = int(self.relay_gpu)
        if relay_gpu < 0:
            raise ValueError("relay_gpu must be non-negative")
        total_bytes = int(self.total_bytes)
        max_chunk_bytes = int(self.max_chunk_bytes)
        chunk_count = int(self.chunk_count)
        alignment_bytes = int(self.alignment_bytes)
        if total_bytes <= 0:
            raise ValueError("staging total_bytes must be positive")
        if max_chunk_bytes <= 0:
            raise ValueError("staging max_chunk_bytes must be positive")
        if max_chunk_bytes > total_bytes:
            raise ValueError("staging max_chunk_bytes cannot exceed total_bytes")
        if chunk_count <= 0:
            raise ValueError("staging chunk_count must be positive")
        if alignment_bytes <= 0:
            raise ValueError("staging alignment_bytes must be positive")
        if not str(self.buffer_kind).strip():
            raise ValueError("staging buffer_kind must be non-empty")
        object.__setattr__(self, "relay_gpu", relay_gpu)
        object.__setattr__(self, "total_bytes", total_bytes)
        object.__setattr__(self, "max_chunk_bytes", max_chunk_bytes)
        object.__setattr__(self, "chunk_count", chunk_count)
        object.__setattr__(self, "alignment_bytes", alignment_bytes)
        object.__setattr__(self, "buffer_kind", str(self.buffer_kind))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class WorkerDataPlaneRequest:
    transfer_id: str
    lease_id: str
    session_id: str
    job_id: str
    relay_gpu: int
    direction: str
    src_handle: WorkerBufferHandle
    dst_handle: WorkerBufferHandle
    staging: WorkerStagingBufferRequirement
    ranges: tuple[dict[str, int], ...] = field(default_factory=tuple)
    plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_authorization(
        cls,
        authorization: WorkerTransferAuthorization,
        *,
        staging_alignment_bytes: int = 256,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkerDataPlaneRequest":
        if not isinstance(authorization, WorkerTransferAuthorization):
            raise TypeError("authorization must be a WorkerTransferAuthorization")
        if authorization.relay_gpu is None:
            raise ValueError("relay_gpu is required for worker data-plane request")
        ranges = tuple(authorization.ranges)
        if not ranges:
            raise ValueError("worker data-plane request requires chunk ranges")
        total_bytes = sum(item["bytes"] for item in ranges)
        return cls(
            transfer_id=authorization.transfer_id,
            lease_id=authorization.lease_id,
            session_id=authorization.session_id,
            job_id=authorization.job_id,
            relay_gpu=authorization.relay_gpu,
            direction=authorization.direction,
            src_handle=WorkerBufferHandle.from_buffer_registration(
                authorization.src_buffer,
                access="read",
            ),
            dst_handle=WorkerBufferHandle.from_buffer_registration(
                authorization.dst_buffer,
                access="write",
            ),
            staging=WorkerStagingBufferRequirement(
                relay_gpu=authorization.relay_gpu,
                total_bytes=total_bytes,
                max_chunk_bytes=max(item["bytes"] for item in ranges),
                chunk_count=len(ranges),
                alignment_bytes=staging_alignment_bytes,
            ),
            ranges=ranges,
            plan=authorization.plan,
            metadata={} if metadata is None else metadata,
        )

    def __post_init__(self) -> None:
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
        direction = str(self.direction).lower()
        if direction not in {"h2d", "d2h"}:
            raise ValueError("direction must be h2d or d2h")
        if not isinstance(self.src_handle, WorkerBufferHandle):
            raise TypeError("src_handle must be a WorkerBufferHandle")
        if not isinstance(self.dst_handle, WorkerBufferHandle):
            raise TypeError("dst_handle must be a WorkerBufferHandle")
        if not isinstance(self.staging, WorkerStagingBufferRequirement):
            raise TypeError("staging must be a WorkerStagingBufferRequirement")
        if not isinstance(self.plan, Mapping):
            raise TypeError("plan must be a mapping")
        ranges = _normalize_worker_ranges(self.ranges)
        if not ranges:
            raise ValueError("worker data-plane request requires chunk ranges")
        total_bytes = sum(item["bytes"] for item in ranges)
        if total_bytes > self.staging.total_bytes:
            raise ValueError("ranges exceed staging total_bytes")
        if max(item["bytes"] for item in ranges) > self.staging.max_chunk_bytes:
            raise ValueError("ranges exceed staging max_chunk_bytes")
        if self.staging.chunk_count < len(ranges):
            raise ValueError("staging chunk_count cannot be smaller than range count")
        if self.src_handle.job_id != str(self.job_id):
            raise ValueError("src handle job does not match request job")
        if self.dst_handle.job_id != str(self.job_id):
            raise ValueError("dst handle job does not match request job")
        if self.staging.relay_gpu != relay_gpu:
            raise ValueError("staging relay does not match request relay")
        _validate_worker_handle_direction(
            direction,
            src_handle=self.src_handle,
            dst_handle=self.dst_handle,
        )
        _validate_worker_range_bounds(
            ranges,
            src_size_bytes=self.src_handle.size_bytes,
            dst_size_bytes=self.dst_handle.size_bytes,
        )
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "job_id", str(self.job_id))
        object.__setattr__(self, "relay_gpu", relay_gpu)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "ranges", tuple(ranges))
        object.__setattr__(self, "plan", dict(self.plan))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class WorkerDataPlaneCompletion:
    transfer_id: str
    lease_id: str
    state: TransferStatusState
    bytes_completed: int
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.transfer_id).strip():
            raise ValueError("transfer_id must be non-empty")
        if not str(self.lease_id).strip():
            raise ValueError("lease_id must be non-empty")
        state = TransferStatusState(self.state)
        if state not in {TransferStatusState.COMPLETE, TransferStatusState.FAILED}:
            raise ValueError("worker completion state must be complete or failed")
        bytes_completed = int(self.bytes_completed)
        if bytes_completed < 0:
            raise ValueError("bytes_completed must be non-negative")
        if state == TransferStatusState.FAILED and self.error is None:
            raise ValueError("failed worker completion requires error")
        if state == TransferStatusState.COMPLETE and self.error is not None:
            raise ValueError("complete worker completion cannot include error")
        object.__setattr__(self, "transfer_id", str(self.transfer_id))
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "bytes_completed", bytes_completed)
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _normalize_worker_ranges(ranges: tuple[dict[str, int], ...]) -> tuple[dict[str, int], ...]:
    normalized_ranges: list[dict[str, int]] = []
    for item in ranges:
        src_offset = int(item["src_offset"])
        dst_offset = int(item["dst_offset"])
        bytes_count = int(item["bytes"])
        if src_offset < 0 or dst_offset < 0:
            raise ValueError("range offsets must be non-negative")
        if bytes_count <= 0:
            raise ValueError("range bytes must be positive")
        normalized_ranges.append(
            {
                "src_offset": src_offset,
                "dst_offset": dst_offset,
                "bytes": bytes_count,
            }
        )
    return tuple(normalized_ranges)


def _validate_worker_range_bounds(
    ranges: tuple[dict[str, int], ...],
    *,
    src_size_bytes: int,
    dst_size_bytes: int,
) -> None:
    src_size = int(src_size_bytes)
    dst_size = int(dst_size_bytes)
    for item in ranges:
        if item["src_offset"] + item["bytes"] > src_size:
            raise ValueError("worker range exceeds src buffer size")
        if item["dst_offset"] + item["bytes"] > dst_size:
            raise ValueError("worker range exceeds dst buffer size")


def _validate_worker_handle_direction(
    direction: str,
    *,
    src_handle: WorkerBufferHandle,
    dst_handle: WorkerBufferHandle,
) -> None:
    if direction == "h2d":
        if src_handle.handle_type != "shared_pinned_cpu":
            raise ValueError("h2d worker source must be shared_pinned_cpu")
        if dst_handle.handle_type != "cuda_ipc_device":
            raise ValueError("h2d worker destination must be cuda_ipc_device")
        return
    if src_handle.handle_type != "cuda_ipc_device":
        raise ValueError("d2h worker source must be cuda_ipc_device")
    if dst_handle.handle_type != "shared_pinned_cpu":
        raise ValueError("d2h worker destination must be shared_pinned_cpu")


def _normalize_buffer_handle_metadata(
    *,
    kind: str,
    pinned: bool,
    device_index: int | None,
    size_bytes: int,
    handle_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not handle_type.strip():
        raise ValueError("handle_type must be non-empty")
    if handle_type == "shared_pinned_cpu":
        if kind != "cpu_pinned":
            raise ValueError("shared_pinned_cpu handles require cpu_pinned buffers")
        if not pinned:
            raise ValueError("shared_pinned_cpu handles require pinned buffers")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        required = ("shared_memory_name", "offset_bytes")
        for field_name in required:
            if field_name not in metadata:
                raise ValueError(f"shared_pinned_cpu metadata requires {field_name}")
        if int(metadata["offset_bytes"]) < 0:
            raise ValueError("shared_pinned_cpu offset_bytes must be non-negative")
        span = int(metadata["offset_bytes"]) + int(size_bytes)
        if "shared_memory_size_bytes" in metadata:
            shared_size = int(metadata["shared_memory_size_bytes"])
            if shared_size < span:
                raise ValueError(
                    "shared_pinned_cpu shared_memory_size_bytes is smaller than buffer span"
                )
    elif handle_type == "cuda_ipc_device":
        if kind != "gpu":
            raise ValueError("cuda_ipc_device handles require gpu buffers")
        if device_index is None:
            raise ValueError("cuda_ipc_device handles require device_index")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        if "cuda_ipc_handle" not in metadata:
            raise ValueError("cuda_ipc_device metadata requires cuda_ipc_handle")
    elif not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    return dict(metadata)


@dataclass(frozen=True)
class CleanupRequest:
    target_kind: str
    target_id: str
    reason: str
    force: bool = False

    def __post_init__(self) -> None:
        if not str(self.target_kind).strip():
            raise ValueError("target_kind must be non-empty")
        if not str(self.target_id).strip():
            raise ValueError("target_id must be non-empty")
        if not str(self.reason).strip():
            raise ValueError("reason must be non-empty")
        object.__setattr__(self, "target_kind", str(self.target_kind))
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "reason", str(self.reason))


@dataclass
class Session:
    session_id: str
    target_gpu: int
    relay_gpus: list[int]
    max_inflight_chunks: int
    active_chunks: int = 0
    active: bool = True
    created_at: float = 0.0
    last_seen: float = 0.0
    closed_at: float | None = None


@dataclass
class RelayQuota:
    relay_gpu: int
    max_sessions: int = 1
    max_inflight_chunks: int = 8
    sessions: set[str] = field(default_factory=set)
    active_chunks: int = 0

    def can_attach(self) -> bool:
        return len(self.sessions) < self.max_sessions

    def can_reserve(self, chunks: int) -> bool:
        return self.active_chunks + chunks <= self.max_inflight_chunks


@dataclass
class TransferReservation:
    reservation_id: str
    session_id: str
    relay_gpu: int
    chunks: int
    bytes: int = 0
    direction: str = "unknown"


@dataclass
class DaemonRequest:
    request_type: RequestType
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DaemonResponse:
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


__all__ = [
    "AutoTransferDecision",
    "BufferRegistration",
    "CleanupRequest",
    "DaemonRequest",
    "DaemonResponse",
    "JobIdentity",
    "LeaseToken",
    "RelayQuota",
    "RequestType",
    "Session",
    "TransferMode",
    "TransferReservation",
    "TransferStatus",
    "TransferStatusState",
    "WorkerTransferAuthorization",
    "WorkerTransferAuthorizationRequest",
    "WorkerBufferHandle",
    "WorkerDataPlaneCompletion",
    "WorkerDataPlaneRequest",
    "WorkerStagingBufferRequirement",
]
