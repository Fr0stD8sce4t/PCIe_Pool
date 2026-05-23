from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .schema import TransferMode


@dataclass(frozen=True)
class PlannerDevice:
    device_id: int
    kind: str = "gpu"
    memory_bytes: int = 0
    numa_node: int | None = None
    pci_bus_id: str | None = None
    name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerLink:
    src_device: int
    dst_device: int
    kind: str
    bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0
    enabled: bool = True
    fabric_kind: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerChunk:
    src_offset: int
    dst_offset: int
    bytes: int
    relay_device: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerPath:
    kind: str
    direction: str
    target_device: int
    relay_device: int = -1
    h2d_bw_gbps: float = 0.0
    d2h_bw_gbps: float = 0.0
    p2p_bw_gbps: float = 0.0
    effective_bw_gbps: float = 0.0
    enabled: bool = True
    fabric_kind: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerPathAssignment:
    path: PlannerPath
    chunks: tuple[PlannerChunk, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        chunks = [chunk.as_dict() for chunk in self.chunks]
        return {
            "path": self.path.as_dict(),
            "chunks": chunks,
            "bytes": sum(chunk["bytes"] for chunk in chunks),
            "chunk_count": len(chunks),
        }


@dataclass(frozen=True)
class PlannerTransferPlan:
    total_bytes: int = 0
    chunk_bytes: int = 16 * 1024 * 1024
    assignments: tuple[PlannerPathAssignment, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_bytes": self.total_bytes,
            "chunk_bytes": self.chunk_bytes,
            "assignments": [assignment.as_dict() for assignment in self.assignments],
        }


@dataclass(frozen=True)
class PlannerLease:
    lease_id: str
    session_id: str
    relay_device: int
    chunk_limit: int
    bytes_limit: int = 0
    direction: str = "unknown"
    granted_at: float = 0.0
    expires_at: float = 0.0
    active: bool = True
    job_id: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerStats:
    bytes: int = 0
    direct_bytes: int = 0
    relay_bytes: int = 0
    direct_chunks: int = 0
    relay_chunks: int = 0
    path_count: int = 0
    relay_path_count: int = 0
    fallback_reason: str | None = None
    requested_mode: TransferMode | str = TransferMode.POOL
    resolved_mode: TransferMode | str = TransferMode.POOL

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        if isinstance(self.requested_mode, TransferMode):
            data["requested_mode"] = self.requested_mode.value
        if isinstance(self.resolved_mode, TransferMode):
            data["resolved_mode"] = self.resolved_mode.value
        return data


__all__ = [
    "PlannerChunk",
    "PlannerDevice",
    "PlannerLease",
    "PlannerLink",
    "PlannerPath",
    "PlannerPathAssignment",
    "PlannerStats",
    "PlannerTransferPlan",
]
