from __future__ import annotations

import unittest

from turbobus.schema import BufferRegistration, WorkerDataPlaneRequest, WorkerTransferAuthorization
from turbobus.worker import WorkerStagingPool, WorkerStagingPoolError, WorkerStagingSlot


def data_plane_request(
    *,
    transfer_id: str = "transfer-1",
    lease_id: str = "lease-1",
    relay_gpu: int = 1,
    bytes_count: int = 65,
    alignment_bytes: int = 64,
) -> WorkerDataPlaneRequest:
    authorization = WorkerTransferAuthorization(
        transfer_id=transfer_id,
        lease_id=lease_id,
        session_id="session-1",
        job_id="job-1",
        src_buffer=BufferRegistration(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=bytes_count,
            pinned=True,
        ),
        dst_buffer=BufferRegistration(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=bytes_count,
            device_index=0,
        ),
        direction="h2d",
        ranges=({"src_offset": 0, "dst_offset": 0, "bytes": bytes_count},),
        relay_gpu=relay_gpu,
    )
    return WorkerDataPlaneRequest.from_authorization(
        authorization,
        staging_alignment_bytes=alignment_bytes,
    )


class WorkerStagingPoolTest(unittest.TestCase):
    def test_allocate_describes_aligned_slot(self) -> None:
        pool = WorkerStagingPool()

        slot = pool.allocate(data_plane_request())

        self.assertIsInstance(slot, WorkerStagingSlot)
        self.assertEqual(slot.slot_id, "staging-1")
        self.assertEqual(slot.transfer_id, "transfer-1")
        self.assertEqual(slot.lease_id, "lease-1")
        self.assertEqual(slot.relay_gpu, 1)
        self.assertEqual(slot.requested_bytes, 65)
        self.assertEqual(slot.allocated_bytes, 128)
        self.assertEqual(slot.max_chunk_bytes, 65)
        self.assertEqual(slot.chunk_count, 1)
        self.assertEqual(slot.metadata["src_buffer_id"], "cpu-buffer")
        self.assertTrue(slot.active)
        described = pool.describe()
        self.assertIn("staging-1", described["active_slots"])
        self.assertEqual(described["active_slots"]["staging-1"]["allocated_bytes"], 128)

    def test_validate_accepts_matching_request(self) -> None:
        pool = WorkerStagingPool()
        request = data_plane_request()
        slot = pool.allocate(request)

        checked = pool.validate_slot_for_request(slot.slot_id, request)

        self.assertEqual(checked.slot_id, slot.slot_id)

    def test_release_removes_slot_and_returns_inactive_record(self) -> None:
        pool = WorkerStagingPool()
        request = data_plane_request()
        slot = pool.allocate(request)

        released = pool.release(slot.slot_id, request)

        self.assertFalse(released.active)
        self.assertEqual(released.slot_id, slot.slot_id)
        self.assertEqual(pool.describe(), {"active_slots": {}})

    def test_double_release_is_rejected(self) -> None:
        pool = WorkerStagingPool()
        request = data_plane_request()
        slot = pool.allocate(request)

        pool.release(slot.slot_id, request)

        with self.assertRaisesRegex(WorkerStagingPoolError, "unknown staging slot"):
            pool.release(slot.slot_id, request)

    def test_release_rejects_wrong_transfer_request(self) -> None:
        pool = WorkerStagingPool()
        slot = pool.allocate(data_plane_request(transfer_id="transfer-1"))

        with self.assertRaisesRegex(WorkerStagingPoolError, "transfer mismatch"):
            pool.release(
                slot.slot_id,
                data_plane_request(transfer_id="transfer-2"),
            )

        self.assertIn(slot.slot_id, pool.describe()["active_slots"])

    def test_validate_rejects_wrong_relay(self) -> None:
        pool = WorkerStagingPool()
        slot = pool.allocate(data_plane_request(relay_gpu=1))

        with self.assertRaisesRegex(WorkerStagingPoolError, "relay mismatch"):
            pool.validate_slot_for_request(
                slot.slot_id,
                data_plane_request(relay_gpu=2),
            )

    def test_validate_rejects_wrong_transfer(self) -> None:
        pool = WorkerStagingPool()
        slot = pool.allocate(data_plane_request(transfer_id="transfer-1"))

        with self.assertRaisesRegex(WorkerStagingPoolError, "transfer mismatch"):
            pool.validate_slot_for_request(
                slot.slot_id,
                data_plane_request(transfer_id="transfer-2"),
            )

    def test_validate_rejects_wrong_lease(self) -> None:
        pool = WorkerStagingPool()
        slot = pool.allocate(data_plane_request(lease_id="lease-1"))

        with self.assertRaisesRegex(WorkerStagingPoolError, "lease mismatch"):
            pool.validate_slot_for_request(
                slot.slot_id,
                data_plane_request(lease_id="lease-2"),
            )


if __name__ == "__main__":
    unittest.main()
