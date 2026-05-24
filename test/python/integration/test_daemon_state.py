from __future__ import annotations

import time
import unittest

from turbobus.daemon.protocol import (
    DaemonRequest,
    RequestType,
    WorkerTransferAuthorizationRequest,
)
from turbobus.daemon.server import TurboBusDaemon
from turbobus.schema import (
    TransferIntent,
    TransferReceipt,
    TransferStatusState,
    WorkloadKind,
)
from turbobus.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
)
from test.python.fixtures.topology import (
    StaticTopologyProvider,
)


CUDA_IPC_TARGET_HANDLE = (b"t" * 64).hex()


def _daemon(*args, **kwargs) -> TurboBusDaemon:
    if "topology_provider" not in kwargs:
        relay_gpus = kwargs.get("relay_gpus", args[0] if args else [])
        kwargs["topology_provider"] = StaticTopologyProvider.from_relay_gpus(relay_gpus)
    return TurboBusDaemon(*args, **kwargs)


def _relay_ranges(plan: dict, relay_gpu: int) -> tuple[dict[str, int], ...]:
    ranges = []
    for assignment in plan["assignments"]:
        path = assignment["path"]
        if path["kind"] != "relay" or int(path["relay_device"]) != int(relay_gpu):
            continue
        ranges.extend(
            {
                "src_offset": int(chunk["src_offset"]),
                "dst_offset": int(chunk["dst_offset"]),
                "bytes": int(chunk["bytes"]),
            }
            for chunk in assignment["chunks"]
        )
    return tuple(ranges)


class DaemonStateTest(unittest.TestCase):
    def test_session_lifecycle_releases_quota(self) -> None:
        daemon = _daemon(relay_gpus=[1], max_sessions_per_relay=1)

        first = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(first.ok)
        session_id = first.payload["session"]["session_id"]

        second = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertFalse(second.ok)
        self.assertIn("unavailable", second.error)

        closed = daemon.close_session(session_id)
        self.assertTrue(closed.ok)

        third = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(third.ok)

    def test_register_session_normalizes_duplicate_relays(self) -> None:
        daemon = _daemon(relay_gpus=[1], max_sessions_per_relay=1)

        registered = daemon.register_session(target_gpu=0, requested_relays=[1, 1])

        self.assertTrue(registered.ok)
        session_id = registered.payload["session"]["session_id"]
        self.assertEqual(registered.payload["session"]["relay_gpus"], [1])
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["sessions"], [session_id])

    def test_register_session_rejects_invalid_session_chunk_limit(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        response = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=0,
        )

        self.assertFalse(response.ok)
        self.assertIn("max_inflight_chunks", response.error)

    def test_job_and_buffer_registration_are_tracked_and_cleaned_up(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        session = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(session.ok)
        session_id = session.payload["session"]["session_id"]

        job = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_JOB,
                payload={
                    "job_id": "job-1",
                    "user_id": "user-1",
                    "session_id": session_id,
                },
            )
        )
        self.assertTrue(job.ok)
        self.assertEqual(job.payload["job"]["job_id"], "job-1")

        buffer_registration = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_BUFFER,
                payload={
                    "buffer_id": "buffer-1",
                    "job_id": "job-1",
                    "kind": "cpu_pinned",
                    "size_bytes": 4096,
                    "device_index": 0,
                    "pinned": True,
                },
            )
        )
        self.assertTrue(buffer_registration.ok)
        self.assertEqual(buffer_registration.payload["buffer"]["buffer_id"], "buffer-1")

        snapshot = daemon.describe().payload
        self.assertEqual(snapshot["jobs"]["job-1"]["user_id"], "user-1")
        self.assertEqual(snapshot["buffers"]["buffer-1"]["kind"], "cpu_pinned")

        cleanup = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.CLEANUP,
                payload={
                    "target_kind": "job",
                    "target_id": "job-1",
                    "reason": "manual",
                },
            )
        )
        self.assertTrue(cleanup.ok)
        self.assertEqual(cleanup.payload["removed"]["jobs"], 1)
        self.assertEqual(cleanup.payload["removed"]["buffers"], 1)
        self.assertNotIn("job-1", daemon.describe().payload["jobs"])
        self.assertNotIn("buffer-1", daemon.describe().payload["buffers"])

    def test_close_session_removes_session_jobs_and_buffers(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        session = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(session.ok)
        session_id = session.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job("job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )
        self.assertTrue(daemon.register_job("detached-job").ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="detached-buffer",
                job_id="detached-job",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )

        closed = daemon.close_session(session_id)
        snapshot = daemon.describe().payload

        self.assertTrue(closed.ok)
        self.assertNotIn("job-1", snapshot["jobs"])
        self.assertNotIn("cpu-buffer", snapshot["buffers"])
        self.assertIn("detached-job", snapshot["jobs"])
        self.assertIn("detached-buffer", snapshot["buffers"])

    def test_register_job_rejects_unknown_session(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        registered = daemon.register_job("job-1", session_id="missing-session")

        self.assertFalse(registered.ok)
        self.assertIn("unknown session", registered.error)
        self.assertEqual(daemon.describe().payload["jobs"], {})

    def test_cleanup_session_reports_removed_jobs_and_buffers(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        session = daemon.register_session(target_gpu=0, requested_relays=[1])
        self.assertTrue(session.ok)
        session_id = session.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job("job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )

        cleanup = daemon.cleanup(
            target_kind="session",
            target_id=session_id,
            reason="test_session_cleanup",
            force=True,
        )

        self.assertTrue(cleanup.ok)
        self.assertEqual(cleanup.payload["removed"]["sessions"], 1)
        self.assertEqual(cleanup.payload["removed"]["jobs"], 1)
        self.assertEqual(cleanup.payload["removed"]["buffers"], 1)
        self.assertEqual(daemon.describe().payload["jobs"], {})
        self.assertEqual(daemon.describe().payload["buffers"], {})

    def test_handle_request_profile(self) -> None:
        daemon = _daemon(relay_gpus=[1, 2], max_sessions_per_relay=2)
        register = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_SESSION,
                payload={"target_gpu": 0, "relay_gpus": [1]},
            )
        )
        self.assertTrue(register.ok)

        profile = daemon.handle_request(DaemonRequest(request_type=RequestType.PROFILE))
        self.assertTrue(profile.ok)
        self.assertIn("sessions", profile.payload)
        self.assertEqual(len(profile.payload["sessions"]), 1)

        session_id = register.payload["session"]["session_id"]
        closed = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.CLOSE_SESSION,
                session_id=session_id,
            )
        )
        self.assertTrue(closed.ok)

    def test_daemon_exposes_injected_resource_inventory(self) -> None:
        inventory = DaemonResourceInventory(
            gpus=(
                GpuInventoryRecord(
                    device_id=0,
                    backend="cuda",
                    vendor="nvidia",
                    pci_bus_id="0000:01:00.0",
                    numa_node=0,
                    role="target",
                ),
                GpuInventoryRecord(
                    device_id=1,
                    backend="cuda",
                    vendor="nvidia",
                    pci_bus_id="0000:02:00.0",
                    numa_node=0,
                    role="relay",
                    visible=False,
                ),
            ),
            pcie_paths=(
                PciePathRecord(
                    device_id=0,
                    numa_node=0,
                    root_complex="rc0",
                    link_generation=5,
                    link_width=16,
                    bandwidth_gbps=63.0,
                ),
                PciePathRecord(
                    device_id=1,
                    numa_node=0,
                    root_complex="rc0",
                    link_generation=5,
                    link_width=16,
                    bandwidth_gbps=63.0,
                ),
            ),
            fabric_links=(
                FabricLinkRecord(
                    src_device_id=1,
                    dst_device_id=0,
                    fabric="nvlink",
                    bandwidth_gbps=100.0,
                    enabled=True,
                ),
            ),
            source="test",
            discovered_at=1.0,
        )
        daemon = _daemon(
            relay_gpus=[1],
            topology_provider=StaticTopologyProvider(inventory),
        )

        response = daemon.handle_request(
            DaemonRequest(request_type=RequestType.GET_INVENTORY)
        )

        self.assertTrue(response.ok)
        payload = response.payload["inventory"]
        self.assertEqual(payload["source"], "test")
        self.assertEqual(payload["gpus"][0]["role"], "target")
        self.assertFalse(payload["gpus"][1]["visible"])
        self.assertEqual(payload["pcie_paths"][1]["device_id"], 1)
        self.assertEqual(payload["fabric_links"][0]["fabric"], "nvlink")

    def test_fixture_inventory_comes_from_explicit_static_provider(self) -> None:
        daemon = _daemon(relay_gpus=[2, 1])

        inventory = daemon.get_inventory()

        self.assertTrue(inventory.ok)
        payload = inventory.payload["inventory"]
        self.assertEqual(payload["source"], "test_fixture_static")
        self.assertEqual(payload["metadata"]["discovery"], "static test fixture")
        self.assertEqual([gpu["device_id"] for gpu in payload["gpus"]], [1, 2])
        self.assertEqual([gpu["role"] for gpu in payload["gpus"]], ["relay", "relay"])

    def test_discover_relays_reports_cross_job_lease_bookkeeping(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=2,
            max_inflight_chunks_per_relay=8,
        )
        first = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        second = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        first_session_id = first.payload["session"]["session_id"]
        second_session_id = second.payload["session"]["session_id"]
        daemon.register_job(
            job_id="job-1",
            user_id="user-1",
            session_id=first_session_id,
        )
        daemon.register_job(
            job_id="job-2",
            user_id="user-2",
            session_id=second_session_id,
        )
        daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata={
                "shared_memory_name": "tb-job-1-src",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 64,
            },
        )
        daemon.register_buffer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=64,
            device_index=0,
            handle_type="cuda_ipc_device",
            metadata={"cuda_ipc_handle": CUDA_IPC_TARGET_HANDLE},
        )
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.plan_transfer(
            session_id=first_session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
            buffer_ids=["cpu-buffer", "gpu-buffer"],
        )
        self.assertTrue(planned.ok)
        lease_token = planned.payload["lease_tokens"][0]

        discovered = daemon.discover_relays(target_gpu=0, requested_relays=[1])

        self.assertTrue(discovered.ok)
        payload = discovered.payload["relay_discovery"]
        self.assertEqual(payload["summary"]["relay_count"], 1)
        self.assertEqual(payload["summary"]["active_session_count"], 2)
        self.assertEqual(payload["summary"]["active_reservation_count"], 1)
        self.assertEqual(payload["summary"]["active_lease_count"], 1)
        relay = payload["relays"][0]
        self.assertEqual(relay["relay_gpu"], 1)
        self.assertTrue(relay["configured"])
        self.assertEqual(relay["eligibility"]["reason"], "eligible")
        self.assertEqual(relay["quota"]["active_sessions"], 2)
        self.assertEqual(relay["quota"]["available_sessions"], 0)
        self.assertEqual(relay["quota"]["active_chunks"], 2)
        self.assertEqual(relay["quota"]["available_chunks"], 6)
        self.assertEqual(
            sorted(session["job_ids"][0] for session in relay["sessions"]),
            ["job-1", "job-2"],
        )
        self.assertEqual(relay["reservations"][0]["job_id"], "job-1")
        self.assertEqual(
            relay["reservations"][0]["reservation_id"],
            lease_token["lease_id"],
        )
        self.assertEqual(relay["leases"][0]["lease_id"], lease_token["lease_id"])
        self.assertEqual(relay["leases"][0]["job_id"], "job-1")
        self.assertEqual(
            relay["leases"][0]["buffer_ids"],
            ("cpu-buffer", "gpu-buffer"),
        )
        self.assertNotIn("token", relay["leases"][0])

    def test_plan_transfer_uses_inventory_eligible_relays_for_profile_lookup(self) -> None:
        inventory = DaemonResourceInventory(
            gpus=(
                GpuInventoryRecord(device_id=0, role="target"),
                GpuInventoryRecord(device_id=1, role="relay"),
                GpuInventoryRecord(device_id=2, role="relay"),
            ),
            pcie_paths=(
                PciePathRecord(device_id=1),
                PciePathRecord(device_id=2),
            ),
            fabric_links=(
                FabricLinkRecord(
                    src_device_id=1,
                    dst_device_id=0,
                    fabric="nvlink",
                    bandwidth_gbps=100.0,
                    enabled=True,
                ),
                FabricLinkRecord(
                    src_device_id=2,
                    dst_device_id=0,
                    fabric="nvlink",
                    bandwidth_gbps=100.0,
                    enabled=False,
                ),
            ),
            source="test",
        )
        daemon = _daemon(
            relay_gpus=[1, 2],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
            topology_provider=StaticTopologyProvider(inventory),
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1, 2],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
        )

        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "pool")
        self.assertEqual(
            planned.payload["reservations"][0]["relay_gpu"],
            1,
        )
        planning = planned.payload["planning"]
        self.assertEqual(planning["target_gpu"], 0)
        self.assertEqual(planning["profile_key"], "target=0;relays=1")
        self.assertEqual(planning["relay_eligibility"]["requested_relays"], [1, 2])
        self.assertEqual(
            planning["relay_eligibility"]["eligible_relays"],
            [{"relay_gpu": 1, "reason": "eligible"}],
        )
        self.assertEqual(
            planning["relay_eligibility"]["filtered_relays"],
            [{"relay_gpu": 2, "reason": "missing enabled fabric link"}],
        )
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 2)
        self.assertEqual(daemon.describe().payload["relay_quotas"][2]["active_chunks"], 0)

    def test_plan_transfer_falls_back_direct_when_inventory_has_no_fabric_path(self) -> None:
        inventory = DaemonResourceInventory(
            gpus=(
                GpuInventoryRecord(device_id=0, role="target"),
                GpuInventoryRecord(device_id=1, role="relay"),
            ),
            pcie_paths=(PciePathRecord(device_id=1),),
            fabric_links=(
                FabricLinkRecord(
                    src_device_id=1,
                    dst_device_id=0,
                    fabric="nvlink",
                    enabled=False,
                ),
            ),
            source="test",
        )
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
            topology_provider=StaticTopologyProvider(inventory),
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        discovered = daemon.discover_relays(target_gpu=0, requested_relays=[1])
        self.assertTrue(discovered.ok)
        self.assertFalse(
            discovered.payload["relay_discovery"]["relays"][0]["eligibility"]["eligible"]
        )
        self.assertEqual(
            discovered.payload["relay_discovery"]["relays"][0]["eligibility"]["reason"],
            "missing enabled fabric link",
        )

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
        )

        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "direct")
        self.assertEqual(planned.payload["reservations"], [])
        self.assertEqual(planned.payload["stats"]["relay_bytes"], 0)
        self.assertEqual(
            planned.payload["planning"]["relay_eligibility"]["filtered_relays"],
            [{"relay_gpu": 1, "reason": "missing enabled fabric link"}],
        )
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_profile_cache_get_put_round_trip(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        profile = {
            "target_device": 0,
            "direct_h2d_bw_gbps": 7.5,
            "direct_d2h_bw_gbps": 8.5,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 7.6,
                    "d2h_bw_gbps": 8.6,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 7.6,
                    "effective_d2h_bw_gbps": 8.6,
                    "p2p_enabled": True,
                }
            ],
        }

        missing = daemon.get_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(missing.ok)
        self.assertIsNone(missing.payload["profile"])

        stored = daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile=profile,
            profile_bytes=1234,
            updated_at=time.time(),
        )
        self.assertTrue(stored.ok)

        loaded = daemon.get_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(loaded.ok)
        self.assertEqual(loaded.payload["profile"]["profile_bytes"], 1234)
        self.assertEqual(
            loaded.payload["profile"]["profile"]["relays"][0]["relay_device"],
            1,
        )

    def test_profile_cache_can_be_invalidated_explicitly(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        profile = {
            "target_device": 0,
            "direct_h2d_bw_gbps": 7.5,
            "direct_d2h_bw_gbps": 8.5,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 7.6,
                    "d2h_bw_gbps": 8.6,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 7.6,
                    "effective_d2h_bw_gbps": 8.6,
                    "p2p_enabled": True,
                }
            ],
        }

        daemon.put_profile(target_gpu=0, relay_gpus=[1], profile=profile, profile_bytes=1234)

        invalidated = daemon.invalidate_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(invalidated.ok)
        self.assertTrue(invalidated.payload["removed"])
        self.assertIsNone(daemon.get_profile(target_gpu=0, relay_gpus=[1]).payload["profile"])

    def test_profile_cache_purges_stale_entries_on_access(self) -> None:
        daemon = _daemon(relay_gpus=[1], profile_max_age_seconds=1.0)
        profile = {
            "target_device": 0,
            "direct_h2d_bw_gbps": 7.5,
            "direct_d2h_bw_gbps": 8.5,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 7.6,
                    "d2h_bw_gbps": 8.6,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 7.6,
                    "effective_d2h_bw_gbps": 8.6,
                    "p2p_enabled": True,
                }
            ],
        }

        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile=profile,
            profile_bytes=1234,
            updated_at=time.time() - 10.0,
        )

        loaded = daemon.get_profile(target_gpu=0, relay_gpus=[1])
        self.assertTrue(loaded.ok)
        self.assertIsNone(loaded.payload["profile"])
        self.assertEqual(daemon.describe().payload["profile_cache"], {})

    def test_handle_request_rejects_invalid_profile_cache_update(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        response = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PUT_PROFILE,
                payload={
                    "target_gpu": 0,
                    "relay_gpus": [1],
                    "profile": {"direct_h2d_bw_gbps": 0.0},
                },
            )
        )

        self.assertFalse(response.ok)
        self.assertIn("direct_h2d", response.error)

    def test_handle_request_rejects_missing_required_fields(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        response = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REGISTER_SESSION,
                payload={"relay_gpus": [1]},
            )
        )

        self.assertFalse(response.ok)
        self.assertIn("invalid request", response.error)

    def test_wire_message_errors_do_not_mutate_state(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        malformed = daemon.handle_wire_message("{not-json")
        missing_type = daemon.handle_wire_message("{}")
        good = daemon.handle_wire_message(
            '{"request_type":"REGISTER_SESSION","payload":{"target_gpu":0,"relay_gpus":[1]}}'
        )

        self.assertFalse(malformed.ok)
        self.assertFalse(missing_type.ok)
        self.assertTrue(good.ok)
        self.assertEqual(len(daemon.describe().payload["sessions"]), 1)

    def test_transfer_reservation_uses_relay_chunk_quota(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=2,
            max_inflight_chunks_per_relay=4,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=4,
        )
        session_id = register.payload["session"]["session_id"]

        first = daemon.reserve_transfer(
            session_id,
            relay_gpu=1,
            chunks=3,
            bytes_=1024,
            direction="h2d",
        )
        self.assertTrue(first.ok)

        blocked = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=2)
        self.assertFalse(blocked.ok)
        self.assertIn("quota", blocked.error)

        reservation_id = first.payload["reservation"]["reservation_id"]
        released = daemon.release_transfer(reservation_id)
        self.assertTrue(released.ok)

        second = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=2)
        self.assertTrue(second.ok)

    def test_transfer_reservation_rejects_invalid_payload_values(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]

        negative_bytes = daemon.reserve_transfer(
            session_id,
            relay_gpu=1,
            chunks=1,
            bytes_=-1,
        )
        invalid_direction = daemon.reserve_transfer(
            session_id,
            relay_gpu=1,
            chunks=1,
            direction="sideways",
        )

        self.assertFalse(negative_bytes.ok)
        self.assertIn("bytes", negative_bytes.error)
        self.assertFalse(invalid_direction.ok)
        self.assertIn("direction", invalid_direction.error)
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_stale_session_reap_releases_reservations_and_quota(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=4,
            session_timeout_seconds=1.0,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=4,
        )
        session_id = register.payload["session"]["session_id"]
        reserved = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=4)
        self.assertTrue(reserved.ok)
        reservation_id = reserved.payload["reservation"]["reservation_id"]

        daemon._sessions[session_id].last_seen = time.time() - 10.0
        expired = daemon.reap_stale_sessions(now=time.time())

        self.assertEqual(expired, [session_id])
        profile = daemon.describe().payload
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        self.assertEqual(profile["relay_quotas"][1]["sessions"], [])
        self.assertEqual(profile["sessions"], {})
        self.assertIn(
            {
                "target_kind": "session",
                "target_id": session_id,
                "reason": "stale_session_timeout",
                "force": True,
            },
            profile["system_cleanup_events"],
        )
        self.assertIn(
            {
                "target_kind": "reservation",
                "target_id": reservation_id,
                "reason": "stale_session_timeout",
                "force": True,
            },
            profile["system_cleanup_events"],
        )

        reopened = daemon.register_session(target_gpu=0, requested_relays=[1], max_inflight_chunks=4)
        self.assertTrue(reopened.ok)

    def test_close_session_releases_transfer_reservations(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=4,
        )
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]
        reserved = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=4)
        self.assertTrue(reserved.ok)
        reservation_id = reserved.payload["reservation"]["reservation_id"]

        closed = daemon.close_session(session_id)
        self.assertTrue(closed.ok)

        profile = daemon.describe()
        self.assertEqual(profile.payload["relay_quotas"][1]["active_chunks"], 0)
        self.assertIn(
            {
                "target_kind": "session",
                "target_id": session_id,
                "reason": "session_closed",
                "force": True,
            },
            profile.payload["system_cleanup_events"],
        )
        self.assertIn(
            {
                "target_kind": "reservation",
                "target_id": reservation_id,
                "reason": "session_closed",
                "force": True,
            },
            profile.payload["system_cleanup_events"],
        )

    def test_transfer_reservation_uses_session_chunk_quota(self) -> None:
        daemon = _daemon(
            relay_gpus=[1, 2],
            max_sessions_per_relay=2,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1, 2],
            max_inflight_chunks=4,
        )
        session_id = register.payload["session"]["session_id"]

        first = daemon.reserve_transfer(session_id, relay_gpu=1, chunks=3)
        self.assertTrue(first.ok)

        blocked = daemon.reserve_transfer(session_id, relay_gpu=2, chunks=2)
        self.assertFalse(blocked.ok)
        self.assertIn("session chunk quota", blocked.error)

        reservation_id = first.payload["reservation"]["reservation_id"]
        released = daemon.release_transfer(reservation_id)
        self.assertTrue(released.ok)

        second = daemon.reserve_transfer(session_id, relay_gpu=2, chunks=2)
        self.assertTrue(second.ok)

    def test_plan_transfer_uses_cached_profile_and_reserves_leases(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
        )

        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "pool")
        self.assertEqual(len(planned.payload["leases"]), 1)
        self.assertEqual(len(planned.payload["reservations"]), 1)
        reservation = planned.payload["reservations"][0]
        self.assertEqual(
            reservation["reservation_id"],
            planned.payload["leases"][0]["lease_id"],
        )
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 2)

        cleaned = daemon.cleanup(
            target_kind="reservation",
            target_id=reservation["reservation_id"],
            reason="test_cleanup",
            force=True,
        )
        self.assertTrue(cleaned.ok)
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_expired_plan_lease_reap_releases_reservation_and_quota(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=2,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )
        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "pool")
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 2)
        transfer_id = planned.payload["transfer_id"]
        lease_token = planned.payload["lease_tokens"][0]

        expired = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.REAP_EXPIRED_LEASES,
                payload={"now": lease_token["expires_at"] + 1.0},
            )
        )

        self.assertTrue(expired.ok)
        self.assertEqual(expired.payload["expired_lease_ids"], [lease_token["lease_id"]])
        self.assertEqual(expired.payload["expired_count"], 1)
        profile = daemon.describe().payload
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        self.assertEqual(profile["reservations"], {})
        self.assertIn(
            {
                "target_kind": "reservation",
                "target_id": lease_token["lease_id"],
                "reason": "lease_expired",
                "force": True,
            },
            profile["system_cleanup_events"],
        )
        status = daemon.transfer_status(transfer_id)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "canceled")
        discovered = daemon.discover_relays(target_gpu=0, requested_relays=[1])
        self.assertEqual(
            discovered.payload["relay_discovery"]["summary"]["active_reservation_count"],
            0,
        )
        self.assertEqual(
            discovered.payload["relay_discovery"]["summary"]["active_lease_count"],
            0,
        )
        self.assertEqual(
            discovered.payload["relay_discovery"]["relays"][0]["quota"]["available_chunks"],
            2,
        )

        replanned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )
        self.assertTrue(replanned.ok)
        self.assertEqual(replanned.payload["stats"]["resolved_mode"], "pool")

    def test_plan_transfer_issues_validatable_lease_tokens(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )

        self.assertTrue(planned.ok)
        lease_token = planned.payload["lease_tokens"][0]
        self.assertEqual(
            lease_token["lease_id"],
            planned.payload["reservations"][0]["reservation_id"],
        )
        self.assertEqual(lease_token["session_id"], session_id)
        self.assertEqual(lease_token["relay_gpu"], 1)
        self.assertEqual(lease_token["job_id"], "job-1")
        self.assertTrue(lease_token["token"])
        self.assertNotIn("lease_tokens", daemon.describe().payload)

        validated = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
        )
        self.assertTrue(validated.ok)

        wrong_token = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token="wrong",
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
        )
        self.assertFalse(wrong_token.ok)
        self.assertIn("invalid lease token", wrong_token.error)

        cleaned = daemon.cleanup(
            target_kind="reservation",
            target_id=lease_token["lease_id"],
            reason="test_cleanup",
            force=True,
        )
        self.assertTrue(cleaned.ok)
        inactive = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
        )
        self.assertFalse(inactive.ok)
        self.assertIn("unknown lease", inactive.error)

    def test_plan_transfer_lease_validation_checks_registered_buffer_ownership(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        job = daemon.register_job(job_id="job-1", session_id=session_id)
        self.assertTrue(job.ok)
        cpu_buffer = daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
        )
        gpu_buffer = daemon.register_buffer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=64,
            device_index=0,
        )
        self.assertTrue(cpu_buffer.ok)
        self.assertTrue(gpu_buffer.ok)
        daemon.register_job(job_id="other-job", session_id=session_id)
        other_buffer = daemon.register_buffer(
            buffer_id="other-buffer",
            job_id="other-job",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
        )
        self.assertTrue(other_buffer.ok)
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "job-1",
                    "buffer_ids": ["cpu-buffer", "gpu-buffer"],
                },
            )
        )

        self.assertTrue(planned.ok)
        lease_token = planned.payload["lease_tokens"][0]
        self.assertEqual(
            lease_token["buffer_ids"],
            ("cpu-buffer", "gpu-buffer"),
        )
        validated = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
            buffer_ids=["cpu-buffer", "gpu-buffer"],
        )
        self.assertTrue(validated.ok)

        partial_buffers = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
            buffer_ids=["cpu-buffer"],
        )
        self.assertFalse(partial_buffers.ok)
        self.assertIn("lease buffer mismatch", partial_buffers.error)

        wrong_buffer = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
            buffer_ids=["other-buffer"],
        )
        self.assertFalse(wrong_buffer.ok)
        self.assertIn("lease buffer mismatch", wrong_buffer.error)

        wrong_owner = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "job-1",
                    "buffer_ids": ["other-buffer"],
                },
            )
        )
        self.assertFalse(wrong_owner.ok)
        self.assertIn("buffer owner", wrong_owner.error)

        other_session = daemon.register_session(
            target_gpu=2,
            requested_relays=[],
            max_inflight_chunks=8,
        )
        other_session_id = other_session.payload["session"]["session_id"]
        cross_session_job = daemon.register_job(
            job_id="cross-session-job",
            session_id=other_session_id,
        )
        self.assertTrue(cross_session_job.ok)
        cross_session_buffer = daemon.register_buffer(
            buffer_id="cross-session-buffer",
            job_id="cross-session-job",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
        )
        self.assertTrue(cross_session_buffer.ok)
        wrong_session = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "cross-session-job",
                    "buffer_ids": ["cross-session-buffer"],
                },
            )
        )
        self.assertFalse(wrong_session.ok)
        self.assertIn("job session", wrong_session.error)

        detached_job = daemon.register_job(job_id="detached-job")
        self.assertTrue(detached_job.ok)
        detached_buffer = daemon.register_buffer(
            buffer_id="detached-buffer",
            job_id="detached-job",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
        )
        self.assertTrue(detached_buffer.ok)
        detached_owner = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "detached-job",
                    "buffer_ids": ["detached-buffer"],
                },
            )
        )
        self.assertFalse(detached_owner.ok)
        self.assertIn("job session", detached_owner.error)

    def test_plan_transfer_infers_job_id_from_registered_buffer_owner(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job(job_id="other-job", session_id=session_id).ok)
        self.assertTrue(daemon.register_job(job_id="job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
            ).ok
        )
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "buffer_ids": ["cpu-buffer", "gpu-buffer"],
                },
            )
        )

        self.assertTrue(planned.ok)
        transfer_id = planned.payload["transfer_id"]
        lease_token = planned.payload["lease_tokens"][0]
        self.assertEqual(planned.payload["transfer_status"]["job_id"], "job-1")
        self.assertEqual(lease_token["job_id"], "job-1")
        validated = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
            buffer_ids=["cpu-buffer", "gpu-buffer"],
        )
        self.assertTrue(validated.ok)

        authorized = daemon.authorize_worker_transfer(
            WorkerTransferAuthorizationRequest(
                transfer_id=transfer_id,
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="h2d",
                relay_gpu=1,
            )
        )
        self.assertTrue(authorized.ok)
        self.assertEqual(authorized.payload["authorization"]["job_id"], "job-1")

    def test_register_buffer_rejects_overwrite_while_buffer_has_active_lease(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job(job_id="job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
            ).ok
        )
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "job-1",
                    "buffer_ids": ["cpu-buffer", "gpu-buffer"],
                },
            )
        )
        self.assertTrue(planned.ok)
        reservation_id = planned.payload["reservations"][0]["reservation_id"]

        overwrite = daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=128,
            pinned=True,
        )
        self.assertFalse(overwrite.ok)
        self.assertIn("active lease", overwrite.error)

        cleaned = daemon.cleanup(
            target_kind="reservation",
            target_id=reservation_id,
            reason="test_cleanup",
            force=True,
        )
        self.assertTrue(cleaned.ok)
        replaced = daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=128,
            pinned=True,
        )
        self.assertTrue(replaced.ok)
        self.assertEqual(replaced.payload["buffer"]["size_bytes"], 128)

    def test_worker_transfer_authorization_packages_validated_transfer_context(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.register_job(job_id="job-1", session_id=session_id)
        daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata={
                "shared_memory_name": "tb-job-1-src",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 64,
            },
        )
        daemon.register_buffer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=64,
            device_index=0,
            handle_type="cuda_ipc_device",
            metadata={"cuda_ipc_handle": CUDA_IPC_TARGET_HANDLE},
        )
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "job-1",
                    "buffer_ids": ["cpu-buffer", "gpu-buffer"],
                },
            )
        )
        self.assertTrue(planned.ok)
        transfer_id = planned.payload["transfer_id"]
        lease_token = planned.payload["lease_tokens"][0]

        authorized = daemon.authorize_worker_transfer(
            WorkerTransferAuthorizationRequest(
                transfer_id=transfer_id,
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="h2d",
                relay_gpu=1,
            )
        )

        self.assertTrue(authorized.ok)
        authorization = authorized.payload["authorization"]
        self.assertEqual(authorization["transfer_id"], transfer_id)
        self.assertEqual(authorization["src_buffer"]["buffer_id"], "cpu-buffer")
        self.assertEqual(authorization["dst_buffer"]["buffer_id"], "gpu-buffer")
        self.assertEqual(authorization["src_buffer"]["handle_type"], "shared_pinned_cpu")
        self.assertEqual(
            authorization["src_buffer"]["metadata"]["shared_memory_name"],
            "tb-job-1-src",
        )
        self.assertEqual(authorization["dst_buffer"]["handle_type"], "cuda_ipc_device")
        self.assertEqual(
            authorization["dst_buffer"]["metadata"]["cuda_ipc_handle"],
            CUDA_IPC_TARGET_HANDLE,
        )
        self.assertEqual(authorization["ranges"], _relay_ranges(planned.payload["plan"], 1))
        self.assertEqual(authorization["relay_gpu"], 1)
        self.assertEqual(authorization["plan"], planned.payload["plan"])
        self.assertEqual(authorized.payload["decision"]["decision_id"], planned.payload["decision_id"])
        ticket = authorized.payload["ticket"]
        self.assertEqual(ticket["decision_id"], planned.payload["decision_id"])
        self.assertEqual(ticket["topology_snapshot_id"], planned.payload["topology_snapshot_id"])
        self.assertEqual(ticket["job_id"], "job-1")
        self.assertEqual(ticket["session_id"], session_id)
        self.assertEqual(ticket["source_buffer_id"], "cpu-buffer")
        self.assertEqual(ticket["destination_buffer_id"], "gpu-buffer")
        self.assertEqual(ticket["plan"], planned.payload["plan"])
        self.assertEqual(ticket["lease_ids"], (lease_token["lease_id"],))
        self.assertEqual(ticket["metadata"]["transfer_id"], transfer_id)

        wrong_transfer = daemon.authorize_worker_transfer(
            WorkerTransferAuthorizationRequest(
                transfer_id="missing-transfer",
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="h2d",
                relay_gpu=1,
            )
        )
        self.assertFalse(wrong_transfer.ok)
        self.assertIn("unknown transfer", wrong_transfer.error)

        wrong_buffer = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.AUTHORIZE_WORKER_TRANSFER,
                payload={
                    "transfer_id": transfer_id,
                    "lease_id": lease_token["lease_id"],
                    "token": lease_token["token"],
                    "session_id": session_id,
                    "job_id": "job-1",
                    "src_buffer_id": "cpu-buffer",
                    "dst_buffer_id": "missing-buffer",
                    "direction": "h2d",
                    "relay_gpu": 1,
                },
            )
        )
        self.assertFalse(wrong_buffer.ok)
        self.assertIn("lease buffer mismatch", wrong_buffer.error)

        swapped_buffers = daemon.authorize_worker_transfer(
            WorkerTransferAuthorizationRequest(
                transfer_id=transfer_id,
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="gpu-buffer",
                dst_buffer_id="cpu-buffer",
                direction="h2d",
                relay_gpu=1,
            )
        )
        self.assertFalse(swapped_buffers.ok)
        self.assertIn("lease buffer mismatch", swapped_buffers.error)

        mismatched_ranges = daemon.authorize_worker_transfer(
            WorkerTransferAuthorizationRequest(
                transfer_id=transfer_id,
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="h2d",
                relay_gpu=1,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
            )
        )
        self.assertFalse(mismatched_ranges.ok)
        self.assertIn("worker ranges do not match daemon plan", mismatched_ranges.error)

    def test_worker_transfer_authorization_rejects_terminal_transfer_status(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job(job_id="job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
                handle_type="shared_pinned_cpu",
                metadata={
                    "shared_memory_name": "tb-job-1-src",
                    "offset_bytes": 0,
                    "shared_memory_size_bytes": 64,
                },
            ).ok
        )
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
                handle_type="cuda_ipc_device",
                metadata={"cuda_ipc_handle": CUDA_IPC_TARGET_HANDLE},
            ).ok
        )
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "job-1",
                    "buffer_ids": ["cpu-buffer", "gpu-buffer"],
                },
            )
        )
        self.assertTrue(planned.ok)
        transfer_id = planned.payload["transfer_id"]
        lease_token = planned.payload["lease_tokens"][0]
        failed = daemon.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed=0,
            error="worker_failed_before_authorization",
        )
        self.assertTrue(failed.ok)

        authorized = daemon.authorize_worker_transfer(
            WorkerTransferAuthorizationRequest(
                transfer_id=transfer_id,
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="h2d",
                relay_gpu=1,
            )
        )
        self.assertFalse(authorized.ok)
        self.assertIn("transfer is terminal", authorized.error)

    def test_lease_validation_rejects_terminal_transfer_status(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job(job_id="job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
            ).ok
        )
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                    "job_id": "job-1",
                    "buffer_ids": ["cpu-buffer", "gpu-buffer"],
                },
            )
        )
        self.assertTrue(planned.ok)
        transfer_id = planned.payload["transfer_id"]
        lease_token = planned.payload["lease_tokens"][0]

        failed = daemon.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed=0,
            error="worker_failed_before_lease_validation",
        )
        self.assertTrue(failed.ok)

        validated = daemon.validate_lease(
            lease_id=lease_token["lease_id"],
            token=lease_token["token"],
            session_id=session_id,
            relay_gpu=1,
            job_id="job-1",
            buffer_ids=["cpu-buffer", "gpu-buffer"],
        )

        self.assertFalse(validated.ok)
        self.assertIn("transfer is terminal", validated.error)

    def test_plan_transfer_requires_completion_status_before_release_completes(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )

        transfer_id = planned.payload["transfer_id"]
        reservation_id = planned.payload["reservations"][0]["reservation_id"]
        self.assertEqual(planned.payload["transfer_status"]["state"], "submitted")
        self.assertEqual(planned.payload["transfer_status"]["job_id"], "job-1")

        status = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.TRANSFER_STATUS,
                payload={"transfer_id": transfer_id},
            )
        )
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "submitted")

        released = daemon.release_transfer(reservation_id)
        self.assertFalse(released.ok)
        self.assertIn("transfer is not complete", released.error)

        still_submitted = daemon.transfer_status(transfer_id)
        self.assertTrue(still_submitted.ok)
        self.assertEqual(still_submitted.payload["status"]["state"], "submitted")
        self.assertEqual(still_submitted.payload["status"]["bytes_completed"], 0)

    def test_plan_transfer_records_and_completes_transfer_status(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )

        transfer_id = planned.payload["transfer_id"]
        reservation_id = planned.payload["reservations"][0]["reservation_id"]
        self.assertEqual(planned.payload["transfer_status"]["state"], "submitted")
        self.assertEqual(planned.payload["transfer_status"]["job_id"], "job-1")

        reported = daemon.transfer_status(
            transfer_id,
            state="complete",
            bytes_completed=64,
        )
        self.assertTrue(reported.ok)

        released = daemon.release_transfer(reservation_id)
        self.assertTrue(released.ok)

        completed = daemon.transfer_status(transfer_id)
        self.assertTrue(completed.ok)
        self.assertEqual(completed.payload["status"]["state"], "complete")
        self.assertEqual(completed.payload["status"]["bytes_completed"], 64)

    def test_close_session_marks_planned_transfer_canceled_and_reports_cleanup(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )
        transfer_id = planned.payload["transfer_id"]
        reservation_id = planned.payload["reservations"][0]["reservation_id"]

        closed = daemon.close_session(session_id)

        self.assertTrue(closed.ok)
        status = daemon.transfer_status(transfer_id)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "canceled")
        self.assertEqual(status.payload["status"]["bytes_completed"], 0)
        cleanup_events = daemon.describe().payload["system_cleanup_events"]
        self.assertIn(
            {
                "target_kind": "session",
                "target_id": session_id,
                "reason": "session_closed",
                "force": True,
            },
            cleanup_events,
        )
        self.assertIn(
            {
                "target_kind": "reservation",
                "target_id": reservation_id,
                "reason": "session_closed",
                "force": True,
            },
            cleanup_events,
        )

    def test_transfer_status_can_be_updated_explicitly(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="direct",
            direction="h2d",
            job_id="job-1",
        )
        transfer_id = planned.payload["transfer_id"]

        updated = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.TRANSFER_STATUS,
                payload={
                    "transfer_id": transfer_id,
                    "state": "running",
                    "bytes_completed": 32,
                },
            )
        )

        self.assertTrue(updated.ok)
        self.assertEqual(updated.payload["status"]["state"], "running")
        self.assertEqual(updated.payload["status"]["bytes_completed"], 32)

    def test_transfer_status_rejects_incomplete_complete_update(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="direct",
            direction="h2d",
            job_id="job-1",
        )
        transfer_id = planned.payload["transfer_id"]

        rejected = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.TRANSFER_STATUS,
                payload={
                    "transfer_id": transfer_id,
                    "state": "complete",
                    "bytes_completed": 32,
                },
            )
        )

        self.assertFalse(rejected.ok)
        self.assertIn("bytes_total completed", rejected.error)
        status = daemon.transfer_status(transfer_id)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "submitted")
        self.assertEqual(status.payload["status"]["bytes_completed"], 0)

    def test_transfer_status_rejects_terminal_state_rewrite(self) -> None:
        daemon = _daemon(relay_gpus=[1])
        register = daemon.register_session(target_gpu=0, requested_relays=[1])
        session_id = register.payload["session"]["session_id"]

        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="direct",
            direction="h2d",
            job_id="job-1",
        )
        transfer_id = planned.payload["transfer_id"]
        failed = daemon.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed=0,
            error="worker failed",
        )
        self.assertTrue(failed.ok)

        idempotent = daemon.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed=0,
            error="worker failed",
        )
        rewritten = daemon.transfer_status(
            transfer_id,
            state="complete",
            bytes_completed=64,
        )
        malformed = daemon.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed="bad",
        )
        status = daemon.transfer_status(transfer_id)

        self.assertTrue(idempotent.ok)
        self.assertFalse(rewritten.ok)
        self.assertFalse(malformed.ok)
        self.assertIn("terminal transfer status", rewritten.error)
        self.assertIn("terminal transfer status", malformed.error)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "failed")
        self.assertEqual(status.payload["status"]["bytes_completed"], 0)
        self.assertEqual(status.payload["status"]["error"], "worker failed")

    def test_releasing_reservation_does_not_rewrite_failed_transfer(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )
        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
        )
        transfer_id = planned.payload["transfer_id"]
        reservation_id = planned.payload["reservations"][0]["reservation_id"]
        failed = daemon.transfer_status(
            transfer_id,
            state="failed",
            bytes_completed=0,
            error="worker failed",
        )

        cleaned = daemon.cleanup(
            target_kind="reservation",
            target_id=reservation_id,
            reason="worker failed",
            force=True,
        )
        status = daemon.transfer_status(transfer_id)

        self.assertTrue(failed.ok)
        self.assertTrue(cleaned.ok)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "failed")
        self.assertEqual(status.payload["status"]["bytes_completed"], 0)
        self.assertEqual(status.payload["status"]["error"], "worker failed")

    def test_plan_transfer_falls_back_direct_when_relay_quota_is_unavailable(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=1,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        daemon.put_profile(
            target_gpu=0,
            relay_gpus=[1],
            profile={
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
            },
        )

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id=session_id,
                payload={
                    "total_bytes": 64,
                    "chunk_bytes": 16,
                    "mode": "pool",
                    "direction": "h2d",
                },
            )
        )

        self.assertTrue(planned.ok)
        self.assertEqual(planned.payload["stats"]["resolved_mode"], "direct")
        self.assertIn("quota", planned.payload["stats"]["fallback_reason"])
        self.assertEqual(planned.payload["leases"], [])
        self.assertEqual(planned.payload["reservations"], [])
        self.assertEqual(daemon.describe().payload["relay_quotas"][1]["active_chunks"], 0)

    def test_plan_transfer_rejects_unknown_session(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        planned = daemon.handle_request(
            DaemonRequest(
                request_type=RequestType.PLAN_TRANSFER,
                session_id="missing",
                payload={"total_bytes": 64, "chunk_bytes": 16},
            )
        )

        self.assertFalse(planned.ok)
        self.assertIn("unknown session", planned.error)

    def test_submit_transfer_intent_returns_ticketed_receipt(self) -> None:
        daemon = _daemon(
            relay_gpus=[1],
            max_sessions_per_relay=1,
            max_inflight_chunks_per_relay=8,
        )
        register = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        session_id = register.payload["session"]["session_id"]
        self.assertTrue(daemon.register_job(job_id="job-1", session_id=session_id).ok)
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
            ).ok
        )
        self.assertTrue(
            daemon.register_buffer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=64,
                device_index=0,
            ).ok
        )
        self.assertTrue(
            daemon.put_profile(
                target_gpu=0,
                relay_gpus=[1],
                profile={
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
                },
            ).ok
        )
        intent = TransferIntent(
            intent_id="intent-1",
            job_id="job-1",
            session_id=session_id,
            source_buffer_id="cpu-buffer",
            destination_buffer_id="gpu-buffer",
            direction="h2d",
            total_bytes=64,
            ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 64},),
            workload_kind=WorkloadKind.MODEL_WEIGHTS,
            policy_hints={"latency_sensitive": True},
        )

        submitted = daemon.submit_transfer_intent(intent)

        self.assertTrue(submitted.ok)
        receipt = TransferReceipt(**submitted.payload["receipt"])
        ticket = submitted.payload["ticket"]
        transfer_id = submitted.payload["transfer_id"]
        self.assertEqual(receipt.intent_id, intent.intent_id)
        self.assertEqual(receipt.state, TransferStatusState.SUBMITTED)
        self.assertEqual(receipt.bytes_total, 64)
        self.assertEqual(receipt.bytes_completed, 0)
        self.assertEqual(receipt.ticket_id, ticket["ticket_id"])
        self.assertEqual(ticket["intent_id"], intent.intent_id)
        self.assertEqual(ticket["decision_id"], receipt.decision_id)
        self.assertEqual(ticket["topology_snapshot_id"], receipt.topology_snapshot_id)
        self.assertEqual(ticket["source_buffer_id"], "cpu-buffer")
        self.assertEqual(ticket["destination_buffer_id"], "gpu-buffer")
        self.assertTrue(receipt.path_stats)
        self.assertTrue(receipt.topology_snapshot_id.startswith("topology-"))

        self.assertTrue(
            daemon.transfer_status(
                transfer_id,
                state="complete",
                bytes_completed=64,
            ).ok
        )
        waited = daemon.wait_transfer_receipt(intent.intent_id)

        self.assertTrue(waited.ok)
        completed = TransferReceipt(**waited.payload["receipt"])
        self.assertEqual(completed.state, TransferStatusState.COMPLETE)
        self.assertEqual(completed.bytes_completed, 64)
        self.assertEqual(completed.decision_id, receipt.decision_id)

    def test_submit_transfer_intent_rejects_physical_policy_hints(self) -> None:
        daemon = _daemon(relay_gpus=[1])

        with self.assertRaisesRegex(ValueError, "physical paths"):
            TransferIntent(
                intent_id="intent-1",
                job_id="job-1",
                session_id="session-1",
                source_buffer_id="cpu-buffer",
                destination_buffer_id="gpu-buffer",
                direction="h2d",
                total_bytes=64,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 64},),
                policy_hints={"relay_gpus": [1]},
            )


if __name__ == "__main__":
    unittest.main()
