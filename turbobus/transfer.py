from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from .schema import TransferMode


class TransferDirection(str, Enum):
    H2D = "h2d"
    D2H = "d2h"


@dataclass(frozen=True)
class TransferRange:
    src_offset: int
    dst_offset: int
    bytes: int

    def __post_init__(self) -> None:
        if int(self.src_offset) < 0 or int(self.dst_offset) < 0:
            raise ValueError("range offsets must be non-negative")
        if int(self.bytes) <= 0:
            raise ValueError("range bytes must be positive")
        object.__setattr__(self, "src_offset", int(self.src_offset))
        object.__setattr__(self, "dst_offset", int(self.dst_offset))
        object.__setattr__(self, "bytes", int(self.bytes))

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TransferRequest:
    total_bytes: int
    chunk_bytes: int
    direction: TransferDirection | str
    mode: TransferMode | str = TransferMode.POOL
    request_chunks: int | None = None
    ranges: tuple[TransferRange, ...] = field(default_factory=tuple)
    job_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total_bytes = int(self.total_bytes)
        chunk_bytes = int(self.chunk_bytes)
        if total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")

        ranges = tuple(_coerce_range(item) for item in self.ranges)
        range_bytes = sum(item.bytes for item in ranges)
        if ranges and range_bytes != total_bytes:
            raise ValueError("total_bytes must equal the sum of range bytes")
        direction = TransferDirection(self.direction)
        mode = TransferMode(self.mode)
        metadata = {} if self.metadata is None else dict(self.metadata)
        request_chunks = self.request_chunks
        computed_chunks = _chunk_count(total_bytes, chunk_bytes, ranges)
        if request_chunks is None:
            request_chunks = computed_chunks
        request_chunks = int(request_chunks)
        if request_chunks <= 0:
            raise ValueError("request_chunks must be positive")
        if ranges and request_chunks < computed_chunks:
            raise ValueError("request_chunks cannot be smaller than range chunk count")

        object.__setattr__(self, "total_bytes", total_bytes)
        object.__setattr__(self, "chunk_bytes", chunk_bytes)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "request_chunks", request_chunks)
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_ranges(
        cls,
        ranges: Iterable[TransferRange | tuple[int, int, int] | dict],
        *,
        chunk_bytes: int,
        direction: TransferDirection | str,
        mode: TransferMode | str = TransferMode.POOL,
        job_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "TransferRequest":
        normalized = tuple(_coerce_range(item) for item in ranges)
        total_bytes = sum(item.bytes for item in normalized)
        return cls(
            total_bytes=total_bytes,
            chunk_bytes=chunk_bytes,
            direction=direction,
            mode=mode,
            ranges=normalized,
            job_id=job_id,
            metadata={} if metadata is None else dict(metadata),
        )

    def with_mode(self, mode: TransferMode | str) -> "TransferRequest":
        return TransferRequest(
            total_bytes=self.total_bytes,
            chunk_bytes=self.chunk_bytes,
            direction=self.direction,
            mode=mode,
            request_chunks=self.request_chunks,
            ranges=self.ranges,
            job_id=self.job_id,
            metadata=self.metadata,
        )

    def daemon_payload(self, mode: TransferMode | str | None = None) -> dict[str, object]:
        transfer_mode = self.mode if mode is None else TransferMode(mode)
        payload: dict[str, object] = {
            "total_bytes": self.total_bytes,
            "chunk_bytes": self.chunk_bytes,
            "mode": TransferMode(transfer_mode).value,
            "direction": self.direction.value,
            "request_chunks": self.request_chunks,
        }
        if self.job_id is not None:
            payload["job_id"] = self.job_id
        if self.ranges:
            payload["ranges"] = [item.as_dict() for item in self.ranges]
        buffer_ids = self.metadata.get("buffer_ids")
        if buffer_ids is not None:
            payload["buffer_ids"] = [str(buffer_id) for buffer_id in buffer_ids]
        return payload

    def as_dict(self) -> dict[str, object]:
        data = self.daemon_payload()
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


def _coerce_range(item) -> TransferRange:
    if isinstance(item, TransferRange):
        return item
    if isinstance(item, Mapping):
        return TransferRange(
            src_offset=int(item["src_offset"]),
            dst_offset=int(item["dst_offset"]),
            bytes=int(item["bytes"]),
        )
    if isinstance(item, tuple) or isinstance(item, list):
        if len(item) != 3:
            raise ValueError("range tuples must be (src_offset, dst_offset, bytes)")
        return TransferRange(
            src_offset=int(item[0]),
            dst_offset=int(item[1]),
            bytes=int(item[2]),
        )
    return TransferRange(
        src_offset=int(getattr(item, "src_offset")),
        dst_offset=int(getattr(item, "dst_offset")),
        bytes=int(getattr(item, "bytes")),
    )


def _chunk_count(
    total_bytes: int,
    chunk_bytes: int,
    ranges: tuple[TransferRange, ...],
) -> int:
    if ranges:
        return sum(max(1, math.ceil(item.bytes / chunk_bytes)) for item in ranges)
    return max(1, math.ceil(total_bytes / chunk_bytes))


__all__ = ["TransferDirection", "TransferRange", "TransferRequest"]
