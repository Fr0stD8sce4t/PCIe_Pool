from __future__ import annotations

import unittest

from turbobus.scheduler import DaemonScheduler
from turbobus.schema import (
    RelayQuota,
    SchedulingDecision,
    SchedulingDecisionState,
    Session,
    TransferMode,
)


def profile_entry() -> dict:
    return {
        "profile": {
            "target_device": 0,
            "direct_h2d_bw_gbps": 7.5,
            "direct_d2h_bw_gbps": 6.5,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 7.5,
                    "d2h_bw_gbps": 6.5,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 7.5,
                    "effective_d2h_bw_gbps": 6.5,
                    "p2p_enabled": True,
                }
            ],
        }
    }


class DaemonSchedulerTest(unittest.TestCase):
    def make_scheduler(self) -> DaemonScheduler:
        lease_counter = {"value": 0}
        decision_counter = {"value": 0}

        def lease_id() -> str:
            lease_counter["value"] += 1
            return f"lease-{lease_counter['value']}"

        def decision_id() -> str:
            decision_counter["value"] += 1
            return f"decision-{decision_counter['value']}"

        return DaemonScheduler(
            lease_id_factory=lease_id,
            decision_id_factory=decision_id,
            lease_seconds=10.0,
        )

    def make_session(self) -> Session:
        return Session(
            session_id="session-1",
            target_gpu=0,
            relay_gpus=[1],
            max_inflight_chunks=8,
            active_chunks=0,
        )

    def test_pool_plan_issues_relay_lease(self) -> None:
        scheduler = self.make_scheduler()
        session = self.make_session()
        quotas = {1: RelayQuota(relay_gpu=1, max_inflight_chunks=8)}

        decision = scheduler.plan_transfer(
            session=session,
            profile_entry=profile_entry(),
            relay_quotas=quotas,
            total_bytes=64,
            chunk_bytes=16,
            mode=TransferMode.POOL,
            direction="h2d",
            now=100.0,
            job_id="job-1",
            intent_id="intent-1",
            topology_snapshot_id="topology-1",
        )

        self.assertIsInstance(decision, SchedulingDecision)
        self.assertEqual(decision.state, SchedulingDecisionState.PLANNED)
        self.assertEqual(decision.decision_id, "decision-1")
        self.assertEqual(decision.intent_id, "intent-1")
        self.assertEqual(decision.topology_snapshot_id, "topology-1")
        self.assertEqual(decision.job_id, "job-1")
        self.assertEqual(decision.metadata["stats"]["resolved_mode"], "pool")
        self.assertEqual(decision.metadata["stats"]["direct_bytes"], 32)
        self.assertEqual(decision.metadata["stats"]["relay_bytes"], 32)
        self.assertEqual(len(decision.metadata["leases"]), 1)
        lease = decision.metadata["leases"][0]
        self.assertEqual(lease["lease_id"], "lease-1")
        self.assertEqual(lease["relay_device"], 1)
        self.assertEqual(lease["chunk_limit"], 2)
        self.assertEqual(lease["bytes_limit"], 32)
        self.assertEqual(lease["expires_at"], 110.0)
        self.assertEqual(decision.path_summary[0]["kind"], "direct")
        self.assertEqual(decision.path_summary[1]["kind"], "relay")

    def test_quota_denial_returns_direct_fallback(self) -> None:
        scheduler = self.make_scheduler()
        session = self.make_session()
        quotas = {1: RelayQuota(relay_gpu=1, max_inflight_chunks=1)}

        decision = scheduler.plan_transfer(
            session=session,
            profile_entry=profile_entry(),
            relay_quotas=quotas,
            total_bytes=64,
            chunk_bytes=16,
            mode=TransferMode.POOL,
            direction="h2d",
        )

        self.assertEqual(decision.state, SchedulingDecisionState.FALLBACK)
        self.assertEqual(decision.metadata["stats"]["resolved_mode"], "direct")
        self.assertEqual(decision.metadata["leases"], [])
        self.assertIn("quota", decision.fallback_reason)
        self.assertEqual(
            {
                item["path"]["kind"]
                for item in decision.plan["assignments"]
            },
            {"direct"},
        )

    def test_missing_profile_returns_direct_fallback(self) -> None:
        scheduler = self.make_scheduler()
        session = self.make_session()
        quotas = {1: RelayQuota(relay_gpu=1, max_inflight_chunks=8)}

        decision = scheduler.plan_transfer(
            session=session,
            profile_entry=None,
            relay_quotas=quotas,
            total_bytes=64,
            chunk_bytes=16,
            mode=TransferMode.POOL,
            direction="h2d",
        )

        self.assertEqual(decision.state, SchedulingDecisionState.FALLBACK)
        self.assertEqual(decision.metadata["stats"]["resolved_mode"], "direct")
        self.assertEqual(decision.metadata["leases"], [])
        self.assertIn("profile miss", decision.fallback_reason)

    def test_invalid_request_values_are_rejected(self) -> None:
        scheduler = self.make_scheduler()
        session = self.make_session()
        quotas = {1: RelayQuota(relay_gpu=1, max_inflight_chunks=8)}

        with self.assertRaises(ValueError):
            scheduler.plan_transfer(
                session=session,
                profile_entry=profile_entry(),
                relay_quotas=quotas,
                total_bytes=-1,
                chunk_bytes=16,
            )
        with self.assertRaises(ValueError):
            scheduler.plan_transfer(
                session=session,
                profile_entry=profile_entry(),
                relay_quotas=quotas,
                total_bytes=64,
                chunk_bytes=0,
            )
        with self.assertRaises(ValueError):
            scheduler.plan_transfer(
                session=session,
                profile_entry=profile_entry(),
                relay_quotas=quotas,
                total_bytes=64,
                chunk_bytes=16,
                direction="sideways",
            )


if __name__ == "__main__":
    unittest.main()
