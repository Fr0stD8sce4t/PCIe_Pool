from __future__ import annotations

from dataclasses import asdict
import json
import unittest

from turbobus.schema import (
    AutoTransferDecision,
    BufferRegistration,
    CleanupRequest,
    DaemonRequest,
    DaemonResponse,
    JobIdentity,
    LeaseToken,
    RelayQuota,
    RequestType,
    Session,
    TransferMode,
    TransferReservation,
    TransferStatus,
    TransferStatusState,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)
from turbobus.daemon.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
)


class SchemaTest(unittest.TestCase):
    def test_auto_transfer_decision_is_json_serializable(self) -> None:
        decision = AutoTransferDecision(
            requested_mode=TransferMode.AUTO,
            resolved_mode=TransferMode.POOL,
            request_bytes=1024,
            request_chunks=2,
            direct_h2d_bw_gbps=7.5,
            relay_effective_bw_gbps=8.0,
            eligible_relay_devices=(1, 2),
            reason="pool speedup",
        )

        payload = json.loads(json.dumps(asdict(decision)))

        self.assertEqual(payload["resolved_mode"], "pool")
        self.assertEqual(payload["eligible_relay_devices"], [1, 2])

    def test_daemon_protocol_round_trip(self) -> None:
        request = DaemonRequest(
            request_type=RequestType.RESERVE_TRANSFER,
            session_id="session-1",
            payload={"relay_gpu": 1, "chunks": 2},
        )
        session = Session(
            session_id="session-1",
            target_gpu=0,
            relay_gpus=[1],
            max_inflight_chunks=4,
        )
        reservation = TransferReservation(
            reservation_id="reservation-1",
            session_id="session-1",
            relay_gpu=1,
            chunks=2,
            bytes=4096,
            direction="h2d",
        )
        response = DaemonResponse(
            ok=True,
            payload={
                "session": asdict(session),
                "reservation": asdict(reservation),
            },
        )

        request_payload = json.loads(json.dumps(asdict(request)))
        response_payload = json.loads(json.dumps(asdict(response)))

        self.assertEqual(request_payload["request_type"], "RESERVE_TRANSFER")
        self.assertEqual(response_payload["payload"]["session"]["session_id"], "session-1")
        self.assertEqual(
            response_payload["payload"]["reservation"]["reservation_id"],
            "reservation-1",
        )

    def test_plan_transfer_request_is_serializable(self) -> None:
        request = DaemonRequest(
            request_type=RequestType.PLAN_TRANSFER,
            session_id="session-1",
            payload={
                "total_bytes": 64,
                "chunk_bytes": 16,
                "mode": "pool",
                "direction": "h2d",
            },
        )

        payload = json.loads(json.dumps(asdict(request)))

        self.assertEqual(payload["request_type"], "PLAN_TRANSFER")
        self.assertEqual(payload["payload"]["mode"], "pool")

    def test_relay_quota_limits(self) -> None:
        quota = RelayQuota(relay_gpu=1, max_sessions=1, max_inflight_chunks=4)

        self.assertTrue(quota.can_attach())
        self.assertTrue(quota.can_reserve(4))

        quota.sessions.add("session-1")
        quota.active_chunks = 2

        self.assertFalse(quota.can_attach())
        self.assertTrue(quota.can_reserve(2))
        self.assertFalse(quota.can_reserve(3))

    def test_daemon_baseline_message_shapes_are_serializable(self) -> None:
        job = JobIdentity(
            job_id="job-1",
            user_id="user-1",
            session_id="session-1",
            container_id="container-1",
            process_id=42,
        )
        buffer_registration = BufferRegistration(
            buffer_id="buffer-1",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=4096,
            device_index=0,
            address=1024,
            pinned=True,
        )
        lease = LeaseToken(
            lease_id="lease-1",
            session_id="session-1",
            relay_gpu=1,
            token="token-1",
            buffer_ids=("cpu-buffer", "gpu-buffer"),
            job_id="job-1",
            issued_at=1.5,
            expires_at=2.5,
        )
        status = TransferStatus(
            transfer_id="transfer-1",
            job_id="job-1",
            state=TransferStatusState.RUNNING,
            bytes_total=4096,
            bytes_completed=1024,
            session_id="session-1",
        )
        cleanup = CleanupRequest(
            target_kind="session",
            target_id="session-1",
            reason="timeout",
            force=True,
        )
        worker_request = WorkerTransferAuthorizationRequest(
            transfer_id="transfer-1",
            lease_id="lease-1",
            token="token-1",
            session_id="session-1",
            job_id="job-1",
            src_buffer_id="cpu-buffer",
            dst_buffer_id="gpu-buffer",
            direction="h2d",
            relay_gpu=1,
            ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 4096},),
        )
        worker_authorization = WorkerTransferAuthorization(
            transfer_id="transfer-1",
            lease_id="lease-1",
            session_id="session-1",
            job_id="job-1",
            src_buffer=buffer_registration,
            dst_buffer=BufferRegistration(
                buffer_id="gpu-buffer",
                job_id="job-1",
                kind="gpu",
                size_bytes=4096,
                device_index=0,
            ),
            direction="h2d",
            relay_gpu=1,
            ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 4096},),
        )

        payload = json.loads(
            json.dumps(
                {
                    "job": asdict(job),
                    "buffer_registration": asdict(buffer_registration),
                    "lease": asdict(lease),
                    "status": asdict(status),
                    "cleanup": asdict(cleanup),
                    "worker_request": asdict(worker_request),
                    "worker_authorization": asdict(worker_authorization),
                }
            )
        )

        self.assertEqual(payload["job"]["process_id"], 42)
        self.assertEqual(payload["buffer_registration"]["kind"], "cpu_pinned")
        self.assertEqual(payload["lease"]["relay_gpu"], 1)
        self.assertEqual(payload["lease"]["token"], "token-1")
        self.assertEqual(payload["lease"]["buffer_ids"], ["cpu-buffer", "gpu-buffer"])
        self.assertEqual(payload["status"]["state"], "running")
        self.assertTrue(payload["cleanup"]["force"])
        self.assertEqual(payload["worker_request"]["direction"], "h2d")
        self.assertEqual(
            payload["worker_authorization"]["src_buffer"]["buffer_id"],
            "buffer-1",
        )

    def test_daemon_baseline_message_validation_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            JobIdentity(job_id="", process_id=1)
        with self.assertRaises(ValueError):
            BufferRegistration(
                buffer_id="buffer-1",
                job_id="job-1",
                kind="",
                size_bytes=1,
            )
        with self.assertRaises(ValueError):
            LeaseToken(
                lease_id="lease-1",
                session_id="session-1",
                relay_gpu=1,
                token="token-1",
                issued_at=5.0,
                expires_at=4.0,
            )
        with self.assertRaises(ValueError):
            LeaseToken(
                lease_id="lease-1",
                session_id="session-1",
                relay_gpu=1,
                token="",
            )
        with self.assertRaises(ValueError):
            LeaseToken(
                lease_id="lease-1",
                session_id="session-1",
                relay_gpu=1,
                token="token-1",
                buffer_ids=("",),
            )
        with self.assertRaises(ValueError):
            TransferStatus(
                transfer_id="transfer-1",
                job_id="job-1",
                state=TransferStatusState.SUBMITTED,
                bytes_total=1,
                bytes_completed=2,
            )
        with self.assertRaises(ValueError):
            WorkerTransferAuthorizationRequest(
                transfer_id="transfer-1",
                lease_id="lease-1",
                token="token-1",
                session_id="session-1",
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="sideways",
            )
        with self.assertRaises(ValueError):
            CleanupRequest(target_kind="", target_id="session-1", reason="timeout")

    def test_daemon_resource_inventory_is_serializable(self) -> None:
        inventory = DaemonResourceInventory(
            gpus=(
                GpuInventoryRecord(
                    device_id=0,
                    backend="cuda",
                    vendor="nvidia",
                    pci_bus_id="0000:01:00.0",
                    numa_node=0,
                    memory_bytes=80 * 1024 * 1024 * 1024,
                    role="target",
                ),
                GpuInventoryRecord(
                    device_id=1,
                    backend="rocm",
                    vendor="amd",
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
            metadata={"note": "injected"},
        )

        payload = json.loads(json.dumps(inventory.as_dict()))

        self.assertEqual(payload["gpus"][0]["backend"], "cuda")
        self.assertEqual(payload["gpus"][1]["vendor"], "amd")
        self.assertFalse(payload["gpus"][1]["visible"])
        self.assertEqual(payload["pcie_paths"][0]["link_width"], 16)
        self.assertEqual(payload["fabric_links"][0]["fabric"], "nvlink")
        self.assertEqual(payload["metadata"]["note"], "injected")

    def test_daemon_resource_inventory_validation_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            GpuInventoryRecord(device_id=-1)
        with self.assertRaises(ValueError):
            PciePathRecord(device_id=0, bandwidth_gbps=-1.0)
        with self.assertRaises(ValueError):
            FabricLinkRecord(src_device_id=0, dst_device_id=0)
        with self.assertRaises(ValueError):
            DaemonResourceInventory(source="")

    def test_inventory_filters_relay_eligibility_by_fabric_links(self) -> None:
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
                    enabled=True,
                ),
                FabricLinkRecord(
                    src_device_id=2,
                    dst_device_id=0,
                    fabric="nvlink",
                    enabled=False,
                ),
            ),
        )

        self.assertEqual(
            inventory.eligible_relay_devices(target_device=0, requested_relays=[1, 2]),
            (1,),
        )
        eligibility = inventory.relay_eligibility(
            target_device=0,
            requested_relays=[1, 2],
        )
        self.assertEqual(eligibility["requested_relays"], [1, 2])
        self.assertEqual(eligibility["eligible_relays"], [{"relay_gpu": 1, "reason": "eligible"}])
        self.assertEqual(
            eligibility["filtered_relays"],
            [{"relay_gpu": 2, "reason": "missing enabled fabric link"}],
        )


if __name__ == "__main__":
    unittest.main()
