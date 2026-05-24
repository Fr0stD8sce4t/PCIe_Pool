from __future__ import annotations

from dataclasses import asdict
import json
import unittest

from turbobus.schema import (
    BufferHandle,
    BufferKind,
    ExecutionTicket,
    SchedulingDecision,
    SchedulingDecisionState,
    TopologySnapshot,
    TransferIntent,
    TransferReceipt,
    TransferStatusState,
    WorkloadKind,
)


class ContractSchemaTest(unittest.TestCase):
    def test_daemon_first_contract_objects_are_serializable(self) -> None:
        src = BufferHandle(
            buffer_id="cpu-buffer",
            job_id="job-1",
            session_id="session-1",
            kind=BufferKind.CPU_PINNED,
            size_bytes=4096,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata={
                "shared_memory_name": "tb-job-1-src",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 4096,
            },
        )
        dst = BufferHandle(
            buffer_id="gpu-buffer",
            job_id="job-1",
            session_id="session-1",
            kind="gpu",
            size_bytes=4096,
            device_index=0,
        )
        intent = TransferIntent(
            intent_id="intent-1",
            job_id="job-1",
            session_id="session-1",
            source_buffer_id=src.buffer_id,
            destination_buffer_id=dst.buffer_id,
            direction="h2d",
            total_bytes=4096,
            ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 4096},),
            workload_kind=WorkloadKind.KV_CACHE,
            priority=10,
            policy_hints={"latency_sensitive": True},
        )
        topology = TopologySnapshot(
            snapshot_id="topology-1",
            source="daemon-nvml",
            discovered_at=1.0,
            version=2,
            devices=(
                {"device_id": 0, "kind": "gpu", "pci_bus_id": "0000:01:00.0"},
                {"device_id": 1, "kind": "gpu", "pci_bus_id": "0000:02:00.0"},
            ),
            pcie_links=({"device_id": 1, "bandwidth_gbps": 63.0},),
            fabric_links=(
                {
                    "src_device_id": 1,
                    "dst_device_id": 0,
                    "fabric": "nvlink",
                    "bandwidth_gbps": 100.0,
                },
            ),
        )
        decision = SchedulingDecision(
            decision_id="decision-1",
            intent_id=intent.intent_id,
            topology_snapshot_id=topology.snapshot_id,
            job_id="job-1",
            session_id="session-1",
            state=SchedulingDecisionState.PLANNED,
            plan={
                "assignments": [
                    {
                        "path": {
                            "kind": "relay",
                            "relay_device": 1,
                            "target_device": 0,
                        },
                        "chunks": [
                            {"src_offset": 0, "dst_offset": 0, "bytes": 4096}
                        ],
                    }
                ]
            },
            path_summary=({"kind": "relay", "bytes": 4096},),
            issued_at=1.5,
        )
        ticket = ExecutionTicket(
            ticket_id="ticket-1",
            decision_id=decision.decision_id,
            intent_id=intent.intent_id,
            topology_snapshot_id=topology.snapshot_id,
            job_id="job-1",
            session_id="session-1",
            source_buffer_id=src.buffer_id,
            destination_buffer_id=dst.buffer_id,
            direction=intent.direction,
            total_bytes=intent.total_bytes,
            ranges=intent.ranges,
            plan=decision.plan,
            lease_ids=("lease-1",),
            issued_at=2.0,
            expires_at=3.0,
        )
        receipt = TransferReceipt(
            receipt_id="receipt-1",
            ticket_id=ticket.ticket_id,
            intent_id=intent.intent_id,
            decision_id=decision.decision_id,
            topology_snapshot_id=topology.snapshot_id,
            job_id="job-1",
            session_id="session-1",
            state=TransferStatusState.COMPLETE,
            bytes_total=4096,
            bytes_completed=4096,
            started_at=2.1,
            completed_at=2.8,
            path_stats=({"kind": "relay", "bytes": 4096, "seconds": 0.7},),
        )

        payload = json.loads(
            json.dumps(
                {
                    "src": asdict(src),
                    "dst": asdict(dst),
                    "intent": asdict(intent),
                    "topology": asdict(topology),
                    "decision": asdict(decision),
                    "ticket": asdict(ticket),
                    "receipt": asdict(receipt),
                }
            )
        )

        self.assertEqual(payload["src"]["kind"], "cpu_pinned")
        self.assertEqual(payload["dst"]["kind"], "gpu")
        self.assertEqual(payload["intent"]["workload_kind"], "kv_cache")
        self.assertEqual(payload["intent"]["policy_hints"]["latency_sensitive"], True)
        self.assertEqual(payload["topology"]["snapshot_id"], "topology-1")
        self.assertEqual(payload["decision"]["state"], "planned")
        self.assertEqual(payload["ticket"]["lease_ids"], ["lease-1"])
        self.assertEqual(payload["receipt"]["state"], "complete")

    def test_transfer_intent_rejects_physical_path_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical paths"):
            TransferIntent(
                intent_id="intent-1",
                job_id="job-1",
                session_id="session-1",
                source_buffer_id="cpu-buffer",
                destination_buffer_id="gpu-buffer",
                direction="h2d",
                total_bytes=1024,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 1024},),
                policy_hints={"relay_gpu": 1},
            )

        with self.assertRaisesRegex(ValueError, "sum of range bytes"):
            TransferIntent(
                intent_id="intent-1",
                job_id="job-1",
                session_id="session-1",
                source_buffer_id="cpu-buffer",
                destination_buffer_id="gpu-buffer",
                direction="h2d",
                total_bytes=2048,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 1024},),
            )

        with self.assertRaisesRegex(ValueError, "direction"):
            TransferIntent(
                intent_id="intent-1",
                job_id="job-1",
                session_id="session-1",
                source_buffer_id="cpu-buffer",
                destination_buffer_id="gpu-buffer",
                direction="p2p",
                total_bytes=1024,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 1024},),
            )

    def test_scheduling_decision_validation_separates_plans_from_rejections(self) -> None:
        with self.assertRaisesRegex(ValueError, "require a plan"):
            SchedulingDecision(
                decision_id="decision-1",
                intent_id="intent-1",
                topology_snapshot_id="topology-1",
                job_id="job-1",
                session_id="session-1",
                state=SchedulingDecisionState.PLANNED,
            )

        with self.assertRaisesRegex(ValueError, "fallback_reason"):
            SchedulingDecision(
                decision_id="decision-1",
                intent_id="intent-1",
                topology_snapshot_id="topology-1",
                job_id="job-1",
                session_id="session-1",
                state=SchedulingDecisionState.FALLBACK,
                plan={"assignments": []},
            )

        rejected = SchedulingDecision(
            decision_id="decision-1",
            intent_id="intent-1",
            topology_snapshot_id="topology-1",
            job_id="job-1",
            session_id="session-1",
            state=SchedulingDecisionState.REJECTED,
            rejection_reason="no eligible relay or direct path",
        )

        self.assertEqual(rejected.state, SchedulingDecisionState.REJECTED)

    def test_execution_ticket_requires_daemon_decision_binding(self) -> None:
        base = {
            "ticket_id": "ticket-1",
            "decision_id": "decision-1",
            "intent_id": "intent-1",
            "topology_snapshot_id": "topology-1",
            "job_id": "job-1",
            "session_id": "session-1",
            "source_buffer_id": "cpu-buffer",
            "destination_buffer_id": "gpu-buffer",
            "direction": "h2d",
            "total_bytes": 1024,
            "ranges": ({"src_offset": 0, "dst_offset": 0, "bytes": 1024},),
            "plan": {"assignments": []},
            "issued_at": 1.0,
            "expires_at": 2.0,
        }

        self.assertEqual(ExecutionTicket(**base).decision_id, "decision-1")

        with self.assertRaisesRegex(ValueError, "decision_id"):
            ExecutionTicket(**{**base, "decision_id": ""})
        with self.assertRaisesRegex(ValueError, "daemon-issued plan"):
            ExecutionTicket(**{**base, "plan": {}})
        with self.assertRaisesRegex(ValueError, "later than issued_at"):
            ExecutionTicket(**{**base, "expires_at": 1.0})

    def test_transfer_receipt_validation_matches_completion_state(self) -> None:
        base = {
            "receipt_id": "receipt-1",
            "ticket_id": "ticket-1",
            "intent_id": "intent-1",
            "decision_id": "decision-1",
            "topology_snapshot_id": "topology-1",
            "job_id": "job-1",
            "session_id": "session-1",
            "bytes_total": 1024,
            "bytes_completed": 1024,
        }

        receipt = TransferReceipt(**base, state=TransferStatusState.COMPLETE)

        self.assertEqual(receipt.bytes_completed, 1024)

        with self.assertRaisesRegex(ValueError, "all bytes"):
            TransferReceipt(
                **{**base, "bytes_completed": 512},
                state=TransferStatusState.COMPLETE,
            )
        with self.assertRaisesRegex(ValueError, "requires error"):
            TransferReceipt(
                **{**base, "bytes_completed": 512},
                state=TransferStatusState.FAILED,
            )
        failed = TransferReceipt(
            **{**base, "bytes_completed": 512},
            state=TransferStatusState.FAILED,
            error="worker failed",
        )

        self.assertEqual(failed.error, "worker failed")

    def test_buffer_handle_and_topology_snapshot_reject_invalid_ownership_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_index"):
            BufferHandle(
                buffer_id="gpu-buffer",
                job_id="job-1",
                session_id="session-1",
                kind=BufferKind.GPU,
                size_bytes=1,
            )

        with self.assertRaisesRegex(ValueError, "pinned=True"):
            BufferHandle(
                buffer_id="cpu-buffer",
                job_id="job-1",
                session_id="session-1",
                kind=BufferKind.CPU_PINNED,
                size_bytes=1,
                pinned=False,
            )

        with self.assertRaisesRegex(ValueError, "discovered_at"):
            TopologySnapshot(
                snapshot_id="topology-1",
                source="daemon-nvml",
                discovered_at=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
