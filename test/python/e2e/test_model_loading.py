from __future__ import annotations

import unittest

from turbobus.model_loading import ModelWeightLoader
from turbobus.offload_store import AdapterTransferContext, BlockState, OffloadBatch, TransferStats
from turbobus.schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind


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


class ModelWeightLoaderTest(unittest.TestCase):
    def test_add_bucket_tracks_model_weight_metadata(self) -> None:
        loader = make_loader()
        cpu = FakeTensor(128)
        gpu = object()

        bucket = loader.add_bucket("layer0.mlp", cpu, gpu, bucket_id=("layer0", 0))

        self.assertEqual(bucket.name, "layer0.mlp")
        self.assertEqual(bucket.block_id, ("layer0", 0))
        self.assertEqual(bucket.cpu_slot, ("layer0", 0))
        self.assertEqual(bucket.gpu_slot, ("layer0", 0))
        self.assertEqual(loader.block_ids(), [("layer0", 0)])
        self.assertEqual(loader.names(), ["layer0.mlp"])
        self.assertEqual(loader.bucket_info("layer0.mlp").state, BlockState.CPU)

    def test_load_bucket_submits_model_weight_intent_and_marks_loaded_after_wait(self) -> None:
        client = FakeClient()
        loader = make_loader(client)
        cpu = FakeTensor(64)
        gpu = object()
        loader.add_bucket("w0", cpu, gpu)

        handle = loader.load_bucket("w0")

        self.assertEqual(len(client.submitted), 1)
        intent = client.submitted[0]
        self.assertEqual(intent.workload_kind, WorkloadKind.MODEL_WEIGHTS)
        self.assertEqual(intent.direction, "h2d")
        self.assertEqual(intent.source_buffer_id, "cpu-buffer")
        self.assertEqual(intent.destination_buffer_id, "gpu-buffer")
        self.assertEqual(intent.total_bytes, 64)
        self.assertEqual(intent.ranges, ({"src_offset": 0, "dst_offset": 0, "bytes": 64},))
        self.assertEqual(intent.policy_hints, {})
        self.assertEqual(intent.metadata["operation"], "prefetch")
        self.assertEqual(loader.bucket("w0").state, BlockState.PREFETCHING)

        loader.wait("w0")

        self.assertEqual(client.waited, [(intent.intent_id, 2.5)])
        self.assertEqual(handle.wait_calls, 1)
        self.assertEqual(loader.bucket("w0").state, BlockState.GPU)
        self.assertEqual(
            loader.transfer_stats("w0"),
            TransferStats(bytes=64, direct_chunks=1, relay_chunks=1),
        )

    def test_packed_buckets_use_one_transfer_intent(self) -> None:
        client = FakeClient()
        loader = make_loader(client)
        cpu = FakeTensor(256)
        gpu = object()
        loader.add_packed_buckets(
            "bucket",
            cpu,
            gpu,
            bucket_bytes=32,
            bucket_count=3,
            start_offset=16,
        )

        handles = loader.load_all()

        self.assertEqual(handles[0], handles[1])
        self.assertEqual(handles[1], handles[2])
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(
            client.submitted[0].ranges,
            (
                {"src_offset": 16, "dst_offset": 16, "bytes": 32},
                {"src_offset": 48, "dst_offset": 48, "bytes": 32},
                {"src_offset": 80, "dst_offset": 80, "bytes": 32},
            ),
        )

        loader.wait_all()

        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(loader.transfer_stats_many(loader.names()), TransferStats(96, 1, 1))
        self.assertEqual(
            [info.state for info in loader.bucket_infos()],
            [BlockState.GPU, BlockState.GPU, BlockState.GPU],
        )

    def test_load_batch_returns_batch_object(self) -> None:
        loader = make_loader()
        cpu = FakeTensor(256)
        gpu = object()
        loader.add_packed_buckets(
            "bucket",
            cpu,
            gpu,
            bucket_bytes=32,
            bucket_count=2,
            start_offset=16,
        )

        batch = loader.load_batch(loader.names())
        batch.wait()

        self.assertIsInstance(batch, OffloadBatch)
        self.assertEqual(batch.operation, "prefetch")
        self.assertEqual(batch.names, ("bucket0", "bucket1"))
        self.assertEqual(batch.transfer_stats(), TransferStats(64, 1, 1))
        self.assertEqual(batch.as_dict()["transfer_stats"]["bytes"], 64)
        self.assertEqual(loader.bucket("bucket0").state, BlockState.GPU)
        self.assertEqual(loader.bucket("bucket1").state, BlockState.GPU)

    def test_mark_unloaded_resets_transfer_state_without_copying(self) -> None:
        loader = make_loader()
        loader.add_bucket("w0", FakeTensor(1), object())
        loader.load_bucket("w0")
        loader.wait("w0")

        loader.mark_unloaded()

        bucket = loader.bucket("w0")
        self.assertEqual(bucket.state, BlockState.CPU)
        self.assertIsNone(bucket.last_handle)
        self.assertIsNone(bucket.last_operation)


def make_loader(client: FakeClient | None = None) -> ModelWeightLoader:
    return ModelWeightLoader(
        client or FakeClient(),
        AdapterTransferContext(
            job_id="job-1",
            session_id="session-1",
            cpu_buffer_id="cpu-buffer",
            gpu_buffer_id="gpu-buffer",
            workload_kind=WorkloadKind.MODEL_WEIGHTS,
            intent_prefix="model",
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
