from __future__ import annotations

import json
import unittest

from turbobus import (
    PlannerChunk,
    PlannerDevice,
    PlannerLease,
    PlannerLink,
    PlannerPath,
    PlannerPathAssignment,
    PlannerStats,
    PlannerTransferPlan,
)
from turbobus.plan_trace import transfer_plan_to_dict
from turbobus.schema import TransferMode


class PlannerTypesTest(unittest.TestCase):
    def test_planner_types_are_serializable(self) -> None:
        device = PlannerDevice(device_id=0, memory_bytes=40 * 1024 * 1024 * 1024, name="gpu0")
        link = PlannerLink(0, 1, kind="nvlink", bandwidth_gbps=50.0, fabric_kind="cuda")
        path = PlannerPath(
            kind="relay",
            direction="h2d",
            target_device=6,
            relay_device=1,
            h2d_bw_gbps=7.5,
            p2p_bw_gbps=50.0,
            effective_bw_gbps=7.0,
        )
        chunks = (
            PlannerChunk(0, 0, 16),
            PlannerChunk(16, 16, 16),
        )
        assignment = PlannerPathAssignment(path=path, chunks=chunks)
        plan = PlannerTransferPlan(total_bytes=32, chunk_bytes=16, assignments=(assignment,))
        lease = PlannerLease(
            lease_id="lease-1",
            session_id="session-1",
            relay_device=1,
            chunk_limit=4,
            bytes_limit=128,
            direction="h2d",
            granted_at=1.0,
            expires_at=10.0,
            job_id="job-1",
        )
        stats = PlannerStats(
            bytes=32,
            direct_bytes=16,
            relay_bytes=16,
            direct_chunks=1,
            relay_chunks=1,
            path_count=2,
            relay_path_count=1,
            fallback_reason="pool",
            requested_mode=TransferMode.AUTO,
            resolved_mode=TransferMode.POOL,
        )

        payload = {
            "device": device.as_dict(),
            "link": link.as_dict(),
            "plan": plan.as_dict(),
            "lease": lease.as_dict(),
            "stats": stats.as_dict(),
        }

        encoded = json.dumps(payload)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["device"]["device_id"], 0)
        self.assertEqual(decoded["link"]["fabric_kind"], "cuda")
        self.assertEqual(decoded["plan"]["assignments"][0]["chunk_count"], 2)
        self.assertEqual(decoded["lease"]["relay_device"], 1)
        self.assertEqual(decoded["stats"]["resolved_mode"], "pool")

    def test_transfer_plan_to_dict_accepts_planner_model(self) -> None:
        path = PlannerPath(kind="direct", direction="h2d", target_device=6)
        plan = PlannerTransferPlan(
            total_bytes=32,
            chunk_bytes=16,
            assignments=(
                PlannerPathAssignment(
                    path=path,
                    chunks=(
                        PlannerChunk(0, 0, 16),
                        PlannerChunk(16, 16, 16),
                    ),
                ),
            ),
        )

        payload = transfer_plan_to_dict(plan)

        self.assertEqual(payload["total_bytes"], 32)
        self.assertEqual(payload["chunk_bytes"], 16)
        self.assertEqual(payload["assignments"][0]["path"]["kind"], "direct")
        self.assertEqual(payload["assignments"][0]["chunk_count"], 2)
        self.assertEqual(payload["assignments"][0]["bytes"], 32)


if __name__ == "__main__":
    unittest.main()
