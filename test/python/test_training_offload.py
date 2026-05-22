from __future__ import annotations

import unittest

from turbobus import (
    BlockState,
    OffloadBatch,
    TrainingOffloadManager,
    TrainingOffloadStore,
    TransferStats,
)


class FakeTensor:
    def __init__(self, bytes_: int) -> None:
        self._bytes = bytes_

    def numel(self) -> int:
        return self._bytes

    def element_size(self) -> int:
        return 1


class FakeHandle:
    def __init__(self, stats) -> None:
        self.stats = stats
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1


class FakeRuntime:
    target_gpu = 6

    def __init__(self) -> None:
        self.calls = []

    def fetch_to_gpu(self, cpu_tensor, gpu_tensor):
        self.calls.append(("fetch", cpu_tensor, gpu_tensor))
        return FakeHandle({"bytes": cpu_tensor.numel(), "direct_chunks": 1})

    def offload_to_cpu(self, gpu_tensor, cpu_tensor):
        self.calls.append(("offload", gpu_tensor, cpu_tensor))
        return FakeHandle({"bytes": cpu_tensor.numel(), "relay_chunks": 1})

    def fetch_ranges_to_gpu(self, cpu_tensor, gpu_tensor, ranges):
        self.calls.append(("fetch_ranges", cpu_tensor, gpu_tensor, ranges))
        return FakeHandle(
            {"bytes": sum(item["bytes"] for item in ranges), "direct_chunks": len(ranges)}
        )

    def offload_ranges_to_cpu(self, gpu_tensor, cpu_tensor, ranges):
        self.calls.append(("offload_ranges", gpu_tensor, cpu_tensor, ranges))
        return FakeHandle(
            {"bytes": sum(item["bytes"] for item in ranges), "relay_chunks": len(ranges)}
        )


class TrainingOffloadManagerTest(unittest.TestCase):
    def test_add_bucket_tracks_training_metadata(self) -> None:
        manager = TrainingOffloadManager(FakeRuntime())
        cpu = FakeTensor(128)
        gpu = object()

        bucket = manager.add_bucket("adam.m0", cpu, gpu, bucket_id=("adam", 0))

        self.assertEqual(bucket.name, "adam.m0")
        self.assertEqual(bucket.block_id, ("adam", 0))
        self.assertEqual(bucket.cpu_slot, ("adam", 0))
        self.assertEqual(bucket.gpu_slot, ("adam", 0))
        self.assertEqual(manager.block_ids(), [("adam", 0)])
        self.assertEqual(manager.bucket_info("adam.m0").state, BlockState.CPU)

    def test_prefetch_and_offload_use_runtime_directions(self) -> None:
        runtime = FakeRuntime()
        manager = TrainingOffloadManager(runtime)
        cpu = FakeTensor(64)
        gpu = object()
        manager.add_bucket("param0", cpu, gpu)

        prefetch = manager.prefetch_bucket("param0")
        manager.wait("param0")
        offload = manager.offload_bucket("param0")
        manager.wait("param0")

        self.assertEqual(runtime.calls, [("fetch", cpu, gpu), ("offload", gpu, cpu)])
        self.assertEqual(prefetch.wait_calls, 1)
        self.assertEqual(offload.wait_calls, 1)
        self.assertEqual(manager.bucket("param0").state, BlockState.CPU)
        self.assertEqual(manager.transfer_stats("param0"), TransferStats(bytes=64, relay_chunks=1))

    def test_packed_buckets_use_range_transfers_in_both_directions(self) -> None:
        runtime = FakeRuntime()
        manager = TrainingOffloadManager(runtime)
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
        self.assertEqual(runtime.calls[0][0], "fetch_ranges")
        self.assertEqual(runtime.calls[1][0], "offload_ranges")
        self.assertEqual(
            runtime.calls[0][3],
            [
                {"src_offset": 16, "dst_offset": 16, "bytes": 32},
                {"src_offset": 48, "dst_offset": 48, "bytes": 32},
            ],
        )
        self.assertEqual(
            runtime.calls[1][3],
            [
                {"src_offset": 16, "dst_offset": 16, "bytes": 32},
                {"src_offset": 48, "dst_offset": 48, "bytes": 32},
            ],
        )
        self.assertEqual(offload_handles[0].wait_calls, 1)
        self.assertEqual(
            manager.transfer_stats_many(manager.names()),
            TransferStats(bytes=64, relay_chunks=2),
        )

    def test_batch_methods_return_batch_objects(self) -> None:
        runtime = FakeRuntime()
        manager = TrainingOffloadManager(runtime)
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
        self.assertEqual(prefetch_batch.transfer_stats(), TransferStats(bytes=64, direct_chunks=2))
        offload_batch = manager.offload_batch(manager.names())
        offload_batch.wait()

        self.assertIsInstance(prefetch_batch, OffloadBatch)
        self.assertIsInstance(offload_batch, OffloadBatch)
        self.assertEqual(prefetch_batch.operation, "prefetch")
        self.assertEqual(offload_batch.operation, "evict")
        self.assertEqual(offload_batch.transfer_stats(), TransferStats(bytes=64, relay_chunks=2))
        self.assertEqual(manager.bucket("bucket0").state, BlockState.CPU)
        self.assertEqual(manager.bucket("bucket1").state, BlockState.CPU)

    def test_mark_helpers_reset_state_without_copying(self) -> None:
        manager = TrainingOffloadManager(FakeRuntime())
        manager.add_bucket("param0", FakeTensor(1), object())

        manager.mark_on_gpu()
        self.assertEqual(manager.bucket("param0").state, BlockState.GPU)

        manager.mark_on_cpu()
        bucket = manager.bucket("param0")
        self.assertEqual(bucket.state, BlockState.CPU)
        self.assertIsNone(bucket.last_handle)
        self.assertIsNone(bucket.last_operation)

    def test_training_offload_store_alias_points_to_manager(self) -> None:
        self.assertIs(TrainingOffloadStore, TrainingOffloadManager)


if __name__ == "__main__":
    unittest.main()
