from __future__ import annotations

import unittest

from turbobus.daemon.scheduler import DaemonScheduler
from turbobus.schema import RelayQuota, Session, TransferMode


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
        counter = {"value": 0}

        def lease_id() -> str:
            counter["value"] += 1
            return f"lease-{counter['value']}"

        return DaemonScheduler(lease_id_factory=lease_id, lease_seconds=10.0)

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
        )

        self.assertEqual(decision.stats.resolved_mode, TransferMode.POOL)
        self.assertEqual(decision.stats.direct_bytes, 32)
        self.assertEqual(decision.stats.relay_bytes, 32)
        self.assertEqual(len(decision.leases), 1)
        self.assertEqual(decision.leases[0].lease_id, "lease-1")
        self.assertEqual(decision.leases[0].relay_device, 1)
        self.assertEqual(decision.leases[0].chunk_limit, 2)
        self.assertEqual(decision.leases[0].bytes_limit, 32)
        self.assertEqual(decision.leases[0].expires_at, 110.0)

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

        self.assertEqual(decision.stats.resolved_mode, TransferMode.DIRECT)
        self.assertEqual(decision.leases, ())
        self.assertIn("quota", decision.stats.fallback_reason)
        self.assertEqual({item.path.kind for item in decision.plan.assignments}, {"direct"})

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

        self.assertEqual(decision.stats.resolved_mode, TransferMode.DIRECT)
        self.assertEqual(decision.leases, ())
        self.assertIn("profile miss", decision.stats.fallback_reason)

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
