from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .planner_types import (
    PlannerChunk,
    PlannerPath,
    PlannerPathAssignment,
    PlannerTransferPlan,
)
from .schema import TransferMode


@dataclass(frozen=True)
class PlannerEngineOptions:
    min_chunks_for_relay: int = 2
    relay_min_effective_bw_gbps: float = 0.0
    relay_min_direct_ratio: float = 0.0


class PlannerEngine:
    def __init__(self, options: PlannerEngineOptions | None = None) -> None:
        self.options = options or PlannerEngineOptions()

    def plan(
        self,
        total_bytes: int,
        chunk_bytes: int,
        profile,
        mode: TransferMode | str = TransferMode.POOL,
        *,
        direction: str = "h2d",
    ) -> PlannerTransferPlan:
        total_bytes = max(0, int(total_bytes))
        if total_bytes == 0:
            return PlannerTransferPlan()
        chunk_bytes = int(chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be greater than zero")

        chunks = _make_chunks(total_bytes, chunk_bytes)
        transfer_mode = TransferMode(mode)
        if transfer_mode is TransferMode.POOL and len(chunks) < self.options.min_chunks_for_relay:
            transfer_mode = TransferMode.DIRECT

        paths = self._build_paths(profile, transfer_mode, direction)
        if not paths:
            raise RuntimeError("no enabled transfer path is available")
        return self._plan_chunks(chunks, total_bytes, chunk_bytes, paths)

    def plan_ranges(
        self,
        ranges: Iterable,
        chunk_bytes: int,
        profile,
        mode: TransferMode | str = TransferMode.POOL,
        *,
        direction: str = "h2d",
    ) -> PlannerTransferPlan:
        chunk_bytes = int(chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be greater than zero")

        chunks: list[PlannerChunk] = []
        total_bytes = 0
        for range_item in ranges:
            src_offset, dst_offset, bytes_ = _range_fields(range_item)
            if bytes_ <= 0:
                continue
            total_bytes += bytes_
            consumed = 0
            while consumed < bytes_:
                chunk_bytes_this = min(chunk_bytes, bytes_ - consumed)
                chunks.append(
                    PlannerChunk(
                        src_offset=src_offset + consumed,
                        dst_offset=dst_offset + consumed,
                        bytes=chunk_bytes_this,
                    )
                )
                consumed += chunk_bytes_this
        if not chunks:
            return PlannerTransferPlan()

        transfer_mode = TransferMode(mode)
        if transfer_mode is TransferMode.POOL and len(chunks) < self.options.min_chunks_for_relay:
            transfer_mode = TransferMode.DIRECT

        paths = self._build_paths(profile, transfer_mode, direction)
        if not paths:
            raise RuntimeError("no enabled transfer path is available")
        return self._plan_chunks(chunks, total_bytes, chunk_bytes, paths)

    def _build_paths(
        self,
        profile,
        mode: TransferMode,
        direction: str,
    ) -> list[PlannerPath]:
        paths: list[PlannerPath] = []
        direct_bw = _direct_bandwidth(profile, direction)
        if mode is not TransferMode.RELAY and direct_bw > 0.0:
            paths.append(
                PlannerPath(
                    kind="direct",
                    direction=direction,
                    target_device=int(getattr(profile, "target_device", 0)),
                    relay_device=-1,
                    h2d_bw_gbps=float(getattr(profile, "direct_h2d_bw_gbps", 0.0) or 0.0),
                    d2h_bw_gbps=float(
                        getattr(profile, "direct_d2h_bw_gbps", 0.0) or direct_bw
                    ),
                    p2p_bw_gbps=0.0,
                    effective_bw_gbps=float(direct_bw),
                    enabled=True,
                )
            )

        if mode is TransferMode.DIRECT:
            return paths

        for relay in getattr(profile, "relays", []) or []:
            relay_effective_bw = _relay_effective_bandwidth(relay, direction)
            if not getattr(relay, "p2p_enabled", False) or relay_effective_bw <= 0.0:
                continue
            if relay_effective_bw < self.options.relay_min_effective_bw_gbps:
                continue
            if (
                direct_bw > 0.0
                and self.options.relay_min_direct_ratio > 0.0
                and relay_effective_bw < direct_bw * self.options.relay_min_direct_ratio
            ):
                continue
            paths.append(
                PlannerPath(
                    kind="relay",
                    direction=direction,
                    target_device=int(getattr(relay, "target_device", getattr(profile, "target_device", 0))),
                    relay_device=int(getattr(relay, "relay_device", -1)),
                    h2d_bw_gbps=float(getattr(relay, "h2d_bw_gbps", 0.0) or 0.0),
                    d2h_bw_gbps=float(
                        getattr(relay, "d2h_bw_gbps", 0.0) or getattr(relay, "h2d_bw_gbps", 0.0)
                    ),
                    p2p_bw_gbps=float(getattr(relay, "p2p_bw_gbps", 0.0) or 0.0),
                    effective_bw_gbps=float(relay_effective_bw),
                    enabled=True,
                )
            )
        return paths

    @staticmethod
    def _plan_chunks(
        chunks: Sequence[PlannerChunk],
        total_bytes: int,
        chunk_bytes: int,
        paths: Sequence[PlannerPath],
    ) -> PlannerTransferPlan:
        total_bw = sum(path.effective_bw_gbps for path in paths)
        if total_bw <= 0.0:
            raise RuntimeError("enabled paths have zero effective bandwidth")

        assignments = [PlannerPathAssignment(path=path, chunks=tuple()) for path in paths]
        assigned_scores = [0.0 for _ in paths]

        for chunk in chunks:
            selected = 0
            best_score = math.inf
            for index, path in enumerate(paths):
                score = assigned_scores[index] / max(path.effective_bw_gbps, 1e-12)
                if score < best_score:
                    best_score = score
                    selected = index
            assignments[selected] = PlannerPathAssignment(
                path=assignments[selected].path,
                chunks=assignments[selected].chunks + (chunk,),
            )
            assigned_scores[selected] += float(chunk.bytes)

        return PlannerTransferPlan(
            total_bytes=int(total_bytes),
            chunk_bytes=int(chunk_bytes),
            assignments=tuple(
                assignment for assignment in assignments if assignment.chunks
            ),
        )


def plan_transfer(
    total_bytes: int,
    chunk_bytes: int,
    profile,
    mode: TransferMode | str = TransferMode.POOL,
    *,
    direction: str = "h2d",
    options: PlannerEngineOptions | None = None,
) -> PlannerTransferPlan:
    return PlannerEngine(options).plan(total_bytes, chunk_bytes, profile, mode, direction=direction)


def plan_transfer_ranges(
    ranges: Iterable,
    chunk_bytes: int,
    profile,
    mode: TransferMode | str = TransferMode.POOL,
    *,
    direction: str = "h2d",
    options: PlannerEngineOptions | None = None,
) -> PlannerTransferPlan:
    return PlannerEngine(options).plan_ranges(ranges, chunk_bytes, profile, mode, direction=direction)


def _direct_bandwidth(profile, direction: str) -> float:
    direct_attr = "direct_h2d_bw_gbps" if direction == "h2d" else "direct_d2h_bw_gbps"
    direct_bw = max(0.0, float(getattr(profile, direct_attr, 0.0) or 0.0))
    if direction != "h2d" and direct_bw <= 0.0:
        direct_bw = max(0.0, float(getattr(profile, "direct_h2d_bw_gbps", 0.0) or 0.0))
    return direct_bw


def _relay_effective_bandwidth(relay, direction: str) -> float:
    relay_attr = "effective_bw_gbps" if direction == "h2d" else "effective_d2h_bw_gbps"
    effective_bw = max(0.0, float(getattr(relay, relay_attr, 0.0) or 0.0))
    if direction != "h2d" and effective_bw <= 0.0:
        effective_bw = max(0.0, float(getattr(relay, "effective_bw_gbps", 0.0) or 0.0))
    return effective_bw


def _make_chunks(total_bytes: int, chunk_bytes: int) -> list[PlannerChunk]:
    chunks = []
    for offset in range(0, total_bytes, chunk_bytes):
        size = min(chunk_bytes, total_bytes - offset)
        chunks.append(PlannerChunk(src_offset=offset, dst_offset=offset, bytes=size))
    return chunks


def _range_fields(range_item) -> tuple[int, int, int]:
    if isinstance(range_item, dict):
        return (
            int(range_item["src_offset"]),
            int(range_item["dst_offset"]),
            int(range_item["bytes"]),
        )
    if isinstance(range_item, tuple) or isinstance(range_item, list):
        if len(range_item) != 3:
            raise ValueError("range tuples must be (src_offset, dst_offset, bytes)")
        return int(range_item[0]), int(range_item[1]), int(range_item[2])
    return (
        int(getattr(range_item, "src_offset")),
        int(getattr(range_item, "dst_offset")),
        int(getattr(range_item, "bytes")),
    )


__all__ = [
    "PlannerEngine",
    "PlannerEngineOptions",
    "plan_transfer",
    "plan_transfer_ranges",
]
