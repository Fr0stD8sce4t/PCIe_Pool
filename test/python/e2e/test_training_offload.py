from __future__ import annotations

import unittest

from turbobus.offload_store import (
    AdapterTransferContext,
    BlockState,
    OffloadBatch,
    TransferStats,
)
from turbobus.schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind
from turbobus.training_offload import TrainingOffloadManager


class FakeTensor:
    def __init__(self, bytes_: int) -> None:
        self._bytes = bytes_

    def numel(self) -> int:
        return self._bytes

    def element_size(self) -> int:
        return 1


class FakeClient:
    def __init__(self) -> None:
        self.submitted: list[TransferIntent] = []
        self.waited: list[tuple[str, float | None]] = []

    def submit_transfer_intent(self, intent: TransferIntent) -> TransferReceipt:
        self.submitted.append(intent)
        return make_receipt(intent, receipt_id=f"submitted-{intent.intent_id}")

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> TransferReceipt:
        self.waited.append((str(intent_id), timeout_seconds))
        intent = next(item for item in self.submitted if item.intent_id == intent_id)
        return make_receipt(intent, receipt_id=f"receipt-{intent_id}")


class TrainingOffloadManagerTest(unittest.TestCase):
    def test_add_bucket_tracks_training_metadata(self) -> None:
        manager = make_manager()
        cpu = FakeTensor(128)
        gpu = object()

        bucket = manager.add_bucket("adam.m0", cpu, gpu, bucket_id=("adam", 0))

        self.assertEqual(bucket.name, "adam.m0")
        self.assertEqual(bucket.block_id, ("adam", 0))
        self.assertEqual(bucket.cpu_slot, ("adam", 0))
        self.assertEqual(bucket.gpu_slot, ("adam", 0))
        self.assertEqual(manager.block_ids(), [("adam", 0)])
        self.assertEqual(manager.bucket_info("adam.m0").state, BlockState.CPU)

    def test_prefetch_and_offload_submit_intents_for_both_directions(self) -> None:
        client = FakeClient()
        manager = make_manager(client)
        cpu = FakeTensor(64)
        gpu = object()
        manager.add_bucket("param0", cpu, gpu)

        prefetch = manager.prefetch_bucket("param0")
        manager.wait("param0")
        offload = manager.offload_bucket("param0")
        manager.wait("param0")

        self.assertEqual([intent.direction for intent in client.submitted], ["h2d", "d2h"])
        self.assertEqual(client.submitted[0].source_buffer_id, "cpu-buffer")
        self.assertEqual(client.submitted[0].destination_buffer_id, "gpu-buffer")
        self.assertEqual(client.submitted[1].source_buffer_id, "gpu-buffer")
        self.assertEqual(client.submitted[1].destination_buffer_id, "cpu-buffer")
        self.assertEqual(client.submitted[0].workload_kind, WorkloadKind.TRAINING_STATE)
        self.assertEqual(client.submitted[1].workload_kind, WorkloadKind.TRAINING_STATE)
        self.assertEqual(prefetch.wait_calls, 1)
        self.assertEqual(offload.wait_calls, 1)
        self.assertEqual(manager.bucket("param0").state, BlockState.CPU)
        self.assertEqual(
            manager.transfer_stats("param0"),
            TransferStats(bytes=64, direct_chunks=1, relay_chunks=1),
        )

    def test_packed_buckets_use_range_intents_in_both_directions(self) -> None:
        client = FakeClient()
        manager = make_manager(client)
        cpu = FakeTensor(256)
        gpu = object()
        manager.add_packed_buckets(
            "bucket",
            cpu,
            gpu,
            bucket_bytes=32,
            bucket_count=2,
            start_offset=16,
        )

        prefetch_handles = manager.prefetch_all()
        manager.wait_all()
        offload_handles = manager.offload_all()
        manager.wait_all()

        self.assertEqual(prefetch_handles[0], prefetch_handles[1])
        self.assertEqual(offload_handles[0], offload_handles[1])
        self.assertEqual(
            client.submitted[0].ranges,
            (
                {"src_offset": 16, "dst_offset": 16, "bytes": 32},
                {"src_offset": 48, "dst_offset": 48, "bytes": 32},
            ),
        )
        self.assertEqual(
            client.submitted[1].ranges,
            (
                {"src_offset": 16, "dst_offset": 16, "bytes": 32},
                {"src_offset": 48, "dst_offset": 48, "bytes": 32},
            ),
        )
        self.assertEqual(offload_handles[0].wait_calls, 1)
        self.assertEqual(
            manager.transfer_stats_many(manager.names()),
            TransferStats(bytes=64, direct_chunks=1, relay_chunks=1),
        )

    def test_batch_methods_return_batch_objects(self) -> None:
        manager = make_manager()
        cpu = FakeTensor(256)
        gpu = object()
        manager.add_packed_buckets(
            "bucket",
            cpu,
            gpu,
            bucket_bytes=32,
            bucket_count=2,
            start_offset=16,
        )

        prefetch_batch = manager.prefetch_batch(manager.names())
        prefetch_batch.wait()
        self.assertEqual(prefetch_batch.transfer_stats(), TransferStats(bytes=64, direct_chunks=1, relay_chunks=1))
        offload_batch = manager.offload_batch(manager.names())
        offload_batch.wait()

        self.assertIsInstance(prefetch_batch, OffloadBatch)
        self.assertIsInstance(offload_batch, OffloadBatch)
        self.assertEqual(prefetch_batch.operation, "prefetch")
        self.assertEqual(offload_batch.operation, "evict")
        self.assertEqual(offload_batch.transfer_stats(), TransferStats(bytes=64, direct_chunks=1, relay_chunks=1))
        self.assertEqual(manager.bucket("bucket0").state, BlockState.CPU)
        self.assertEqual(manager.bucket("bucket1").state, BlockState.CPU)

    def test_mark_helpers_reset_state_without_copying(self) -> None:
        manager = make_manager()
        manager.add_bucket("param0", FakeTensor(1), object())

        manager.mark_on_gpu()
        self.assertEqual(manager.bucket("param0").state, BlockState.GPU)

        manager.mark_on_cpu()
        bucket = manager.bucket("param0")
        self.assertEqual(bucket.state, BlockState.CPU)
        self.assertIsNone(bucket.last_handle)
        self.assertIsNone(bucket.last_operation)


def make_manager(client: FakeClient | None = None) -> TrainingOffloadManager:
    return TrainingOffloadManager(
        client or FakeClient(),
        AdapterTransferContext(
            job_id="job-1",
            session_id="session-1",
            cpu_buffer_id="cpu-buffer",
            gpu_buffer_id="gpu-buffer",
            workload_kind=WorkloadKind.TRAINING_STATE,
            intent_prefix="training",
            wait_timeout_seconds=2.5,
        ),
    )


def make_receipt(intent: TransferIntent, *, receipt_id: str) -> TransferReceipt:
    direct_bytes = intent.total_bytes // 2
    relay_bytes = intent.total_bytes - direct_bytes
    return TransferReceipt(
        receipt_id=receipt_id,
        ticket_id=f"ticket-{intent.intent_id}",
        intent_id=intent.intent_id,
        decision_id=f"decision-{intent.intent_id}",
        topology_snapshot_id="topology-1",
        job_id=intent.job_id,
        session_id=intent.session_id,
        state=TransferStatusState.COMPLETE,
        bytes_total=intent.total_bytes,
        bytes_completed=intent.total_bytes,
        path_stats=(
            {"kind": "direct", "bytes": direct_bytes, "chunk_count": 1},
            {"kind": "relay", "bytes": relay_bytes, "chunk_count": 1},
        ),
    )


if __name__ == "__main__":
    unittest.main()
