from __future__ import annotations

import unittest

from turbobus import BlockState, ModelLoader, ModelWeightLoader, TransferStats


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

    def fetch_ranges_to_gpu(self, cpu_tensor, gpu_tensor, ranges):
        self.calls.append(("fetch_ranges", cpu_tensor, gpu_tensor, ranges))
        return FakeHandle(
            {
                "bytes": sum(item["bytes"] for item in ranges),
                "direct_chunks": len(ranges),
                "relay_chunks": 1,
            }
        )


class ModelWeightLoaderTest(unittest.TestCase):
    def test_add_bucket_tracks_model_weight_metadata(self) -> None:
        loader = ModelWeightLoader(FakeRuntime())
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

    def test_load_bucket_uses_runtime_fetch_and_marks_loaded_after_wait(self) -> None:
        runtime = FakeRuntime()
        loader = ModelWeightLoader(runtime)
        cpu = FakeTensor(64)
        gpu = object()
        loader.add_bucket("w0", cpu, gpu)

        handle = loader.load_bucket("w0")

        self.assertEqual(runtime.calls, [("fetch", cpu, gpu)])
        self.assertEqual(loader.bucket("w0").state, BlockState.PREFETCHING)

        loader.wait("w0")

        self.assertEqual(handle.wait_calls, 1)
        self.assertEqual(loader.bucket("w0").state, BlockState.GPU)
        self.assertEqual(loader.transfer_stats("w0"), TransferStats(bytes=64, direct_chunks=1))

    def test_packed_buckets_use_one_range_transfer(self) -> None:
        runtime = FakeRuntime()
        loader = ModelWeightLoader(runtime)
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
        self.assertEqual(runtime.calls[0][0], "fetch_ranges")
        self.assertEqual(
            runtime.calls[0][3],
            [
                {"src_offset": 16, "dst_offset": 16, "bytes": 32},
                {"src_offset": 48, "dst_offset": 48, "bytes": 32},
                {"src_offset": 80, "dst_offset": 80, "bytes": 32},
            ],
        )

        loader.wait_all()

        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(loader.transfer_stats_many(loader.names()), TransferStats(96, 3, 1))
        self.assertEqual(
            [info.state for info in loader.bucket_infos()],
            [BlockState.GPU, BlockState.GPU, BlockState.GPU],
        )

    def test_mark_unloaded_resets_transfer_state_without_copying(self) -> None:
        loader = ModelWeightLoader(FakeRuntime())
        loader.add_bucket("w0", FakeTensor(1), object())
        loader.load_bucket("w0")
        loader.wait("w0")

        loader.mark_unloaded()

        bucket = loader.bucket("w0")
        self.assertEqual(bucket.state, BlockState.CPU)
        self.assertIsNone(bucket.last_handle)
        self.assertIsNone(bucket.last_operation)

    def test_model_loader_alias_points_to_weight_loader(self) -> None:
        self.assertIs(ModelLoader, ModelWeightLoader)


if __name__ == "__main__":
    unittest.main()
