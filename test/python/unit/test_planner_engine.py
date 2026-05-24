from __future__ import annotations

from dataclasses import dataclass
import unittest

from turbobus.planner_engine import PlannerEngine, PlannerEngineOptions, plan_transfer_ranges
from turbobus.schema import TransferMode


@dataclass(frozen=True)
class RelayProfile:
    relay_device: int
    target_device: int = 0
    h2d_bw_gbps: float = 7.5
    d2h_bw_gbps: float = 7.0
    p2p_bw_gbps: float = 40.0
    effective_bw_gbps: float = 7.5
    effective_d2h_bw_gbps: float = 7.0
    p2p_enabled: bool = True


@dataclass(frozen=True)
class Profile:
    target_device: int = 0
    direct_h2d_bw_gbps: float = 7.5
    direct_d2h_bw_gbps: float = 6.5
    relays: tuple[RelayProfile, ...] = ()


class PlannerEngineTest(unittest.TestCase):
    def test_direct_plan_uses_only_direct_path(self) -> None:
        planner = PlannerEngine()
        plan = planner.plan(
            total_bytes=64,
            chunk_bytes=16,
            profile=Profile(relays=(RelayProfile(1),)),
            mode=TransferMode.DIRECT,
        )

        self.assertEqual(plan.total_bytes, 64)
        self.assertEqual(len(plan.assignments), 1)
        self.assertEqual(plan.assignments[0].path.kind, "direct")
        self.assertEqual(len(plan.assignments[0].chunks), 4)

    def test_relay_plan_uses_only_eligible_relay_path(self) -> None:
        planner = PlannerEngine(
            PlannerEngineOptions(
                relay_min_effective_bw_gbps=7.0,
                relay_min_direct_ratio=0.5,
            )
        )
        plan = planner.plan(
            total_bytes=64,
            chunk_bytes=16,
            profile=Profile(
                relays=(
                    RelayProfile(1, effective_bw_gbps=6.0),
                    RelayProfile(2, effective_bw_gbps=8.0),
                )
            ),
            mode=TransferMode.RELAY,
        )

        self.assertEqual(len(plan.assignments), 1)
        self.assertEqual(plan.assignments[0].path.kind, "relay")
        self.assertEqual(plan.assignments[0].path.relay_device, 2)

    def test_pool_plan_splits_chunks_across_direct_and_relay_paths(self) -> None:
        planner = PlannerEngine()
        plan = planner.plan(
            total_bytes=64,
            chunk_bytes=16,
            profile=Profile(relays=(RelayProfile(1, effective_bw_gbps=7.5),)),
            mode=TransferMode.POOL,
        )

        self.assertEqual(len(plan.assignments), 2)
        self.assertEqual({assignment.path.kind for assignment in plan.assignments}, {"direct", "relay"})
        self.assertEqual(sum(len(assignment.chunks) for assignment in plan.assignments), 4)
        self.assertEqual(sum(assignment.as_dict()["bytes"] for assignment in plan.assignments), 64)

    def test_pool_plan_falls_back_to_direct_for_small_requests(self) -> None:
        planner = PlannerEngine(PlannerEngineOptions(min_chunks_for_relay=4))
        plan = planner.plan(
            total_bytes=16,
            chunk_bytes=16,
            profile=Profile(relays=(RelayProfile(1),)),
            mode=TransferMode.POOL,
        )

        self.assertEqual(len(plan.assignments), 1)
        self.assertEqual(plan.assignments[0].path.kind, "direct")

    def test_range_plan_keeps_offsets(self) -> None:
        plan = plan_transfer_ranges(
            [{"src_offset": 32, "dst_offset": 64, "bytes": 32}],
            chunk_bytes=16,
            profile=Profile(relays=(RelayProfile(1),)),
            mode=TransferMode.DIRECT,
        )

        chunks = plan.assignments[0].chunks
        self.assertEqual([(chunk.src_offset, chunk.dst_offset, chunk.bytes) for chunk in chunks], [(32, 64, 16), (48, 80, 16)])


if __name__ == "__main__":
    unittest.main()
