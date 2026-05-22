from __future__ import annotations

import unittest

from types import SimpleNamespace

from turbobus import (
    BlockState,
    KVBlockStore,
    OffloadManager,
    OffloadStore,
    OffloadBlockInfo,
    TransferStats,
    summarize_transfer_handles,
)


class FakeTensor:
    def __init__(self, bytes_: int) -> None:
        self._bytes = bytes_

    def numel(self) -> int:
        return self._bytes

    def element_size(self) -> int:
        return 1


class FakeHandle:
    def __init__(self, label: str, stats=None) -> None:
        self.label = label
        self.wait_calls = 0
        self.stats = {"label": label} if stats is None else stats

    def wait(self) -> None:
        self.wait_calls += 1


class FakeRuntime:
    target_gpu = 6

    def __init__(self) -> None:
        self.calls = []

    def fetch_to_gpu(self, cpu_tensor, gpu_tensor):
        self.calls.append(("prefetch", cpu_tensor, gpu_tensor))
        return FakeHandle("prefetch")

    def offload_to_cpu(self, gpu_tensor, cpu_tensor):
        self.calls.append(("evict", gpu_tensor, cpu_tensor))
        return FakeHandle("evict")

    def fetch_ranges_to_gpu(self, cpu_tensor, gpu_tensor, ranges):
        self.calls.append(("prefetch_ranges", cpu_tensor, gpu_tensor, ranges))
        return FakeHandle(
            "prefetch_ranges",
            {"bytes": sum(item["bytes"] for item in ranges), "direct_chunks": len(ranges)},
        )

    def offload_ranges_to_cpu(self, gpu_tensor, cpu_tensor, ranges):
        self.calls.append(("evict_ranges", gpu_tensor, cpu_tensor, ranges))
        return FakeHandle(
            "evict_ranges",
            {"bytes": sum(item["bytes"] for item in ranges), "relay_chunks": len(ranges)},
        )


class OffloadStoreTest(unittest.TestCase):
    def test_add_tracks_named_block(self) -> None:
        runtime = FakeRuntime()
        store = OffloadStore(runtime)
        cpu = FakeTensor(128)
        gpu = object()

        block = store.add("kv0", cpu, gpu)

        self.assertIs(block.cpu_tensor, cpu)
        self.assertIs(block.gpu_tensor, gpu)
        self.assertEqual(block.block_id, "kv0")
        self.assertEqual(block.state, BlockState.CPU)
        self.assertEqual(block.bytes, 128)
        self.assertEqual(store.names(), ["kv0"])
        self.assertIs(store.block("kv0"), block)

    def test_add_accepts_connector_fields(self) -> None:
        store = OffloadStore(FakeRuntime())

        block = store.add(
            "kv0",
            FakeTensor(1),
            object(),
            block_id=("request0", 0),
            cpu_slot=3,
            gpu_slot=7,
        )

        self.assertEqual(block.block_id, ("request0", 0))
        self.assertEqual(block.cpu_slot, 3)
        self.assertEqual(block.gpu_slot, 7)

    def test_add_accepts_packed_offsets(self) -> None:
        store = OffloadStore(FakeRuntime())

        block = store.add(
            "kv0",
            FakeTensor(128),
            object(),
            cpu_offset=16,
            gpu_offset=32,
            byte_count=8,
        )

        self.assertEqual(block.cpu_offset, 16)
        self.assertEqual(block.gpu_offset, 32)
        self.assertEqual(block.bytes, 8)

    def test_block_info_returns_connector_snapshot(self) -> None:
        store = OffloadStore(FakeRuntime())
        store.add(
            "kv0",
            FakeTensor(128),
            object(),
            block_id=("req0", 1),
            cpu_slot=3,
            gpu_slot=7,
            cpu_offset=16,
            gpu_offset=32,
            byte_count=8,
        )

        info = store.block_info("kv0")

        self.assertEqual(
            info,
            OffloadBlockInfo(
                name="kv0",
                block_id=("req0", 1),
                cpu_slot=3,
                gpu_slot=7,
                cpu_offset=16,
                gpu_offset=32,
                bytes=8,
                state=BlockState.CPU,
                last_operation=None,
                transfer_stats=None,
            ),
        )
        self.assertEqual(
            info.as_dict(),
            {
                "name": "kv0",
                "block_id": ("req0", 1),
                "cpu_slot": 3,
                "gpu_slot": 7,
                "cpu_offset": 16,
                "gpu_offset": 32,
                "bytes": 8,
                "state": "cpu",
                "last_operation": None,
                "transfer_stats": None,
            },
        )

    def test_block_infos_accepts_optional_name_filter(self) -> None:
        store = OffloadStore(FakeRuntime())
        store.add("kv0", FakeTensor(1), object())
        store.add("kv1", FakeTensor(1), object())

        self.assertEqual(
            [info.name for info in store.block_infos(["kv1"])],
            ["kv1"],
        )
        self.assertEqual(
            [info.name for info in store.block_infos()],
            ["kv0", "kv1"],
        )

    def test_block_store_aliases_and_state_helpers(self) -> None:
        runtime = FakeRuntime()
        store = OffloadStore(runtime)
        cpu = FakeTensor(64)
        gpu = object()

        block = store.add_block("kv0", cpu, gpu)
        store.prefetch("kv0")
        store.wait("kv0")

        self.assertIs(store.get_block("kv0"), block)
        self.assertEqual(store.block_ids(), ["kv0"])

        store.set_block_state("kv0", BlockState.GPU, clear_transfer_state=True)
        self.assertEqual(store.get_block("kv0").state, BlockState.GPU)
        self.assertIsNone(store.get_block("kv0").last_handle)
        store.clear_block_transfer_state("kv0")
        self.assertIsNone(store.get_block("kv0").last_operation)

    def test_add_rejects_invalid_packed_offsets(self) -> None:
        store = OffloadStore(FakeRuntime())

        with self.assertRaises(ValueError):
            store.add("kv0", FakeTensor(128), object(), cpu_offset=-1)
        with self.assertRaises(ValueError):
            store.add("kv1", FakeTensor(128), object(), byte_count=0)

    def test_duplicate_name_is_rejected(self) -> None:
        store = OffloadStore(FakeRuntime())
        store.add("kv0", FakeTensor(1), object())

        with self.assertRaises(ValueError):
            store.add("kv0", FakeTensor(1), object())

    def test_prefetch_and_evict_use_runtime_and_record_last_stats(self) -> None:
        runtime = FakeRuntime()
        store = OffloadStore(runtime)
        cpu = FakeTensor(64)
        gpu = object()
        store.add("kv0", cpu, gpu)

        prefetch = store.prefetch("kv0")
        self.assertEqual(runtime.calls[-1], ("prefetch", cpu, gpu))
        self.assertEqual(store.block("kv0").last_operation, "prefetch")
        self.assertEqual(store.block("kv0").state, BlockState.PREFETCHING)
        self.assertEqual(store.stats("kv0"), {"label": "prefetch"})
        self.assertEqual(store.transfer_stats("kv0"), TransferStats())

        store.wait("kv0")
        self.assertEqual(prefetch.wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.GPU)

        evict = store.evict("kv0")
        self.assertEqual(runtime.calls[-1], ("evict", gpu, cpu))
        self.assertEqual(store.block("kv0").last_operation, "evict")
        self.assertEqual(store.block("kv0").state, BlockState.EVICTING)
        self.assertEqual(store.stats("kv0"), {"label": "evict"})
        self.assertEqual(store.transfer_stats("kv0"), TransferStats())

        store.wait("kv0")
        self.assertEqual(evict.wait_calls, 1)
        self.assertEqual(prefetch.wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.CPU)

    def test_many_methods_submit_and_wait_in_order(self) -> None:
        runtime = FakeRuntime()
        store = OffloadStore(runtime)
        cpu0 = FakeTensor(1)
        cpu1 = FakeTensor(1)
        gpu0 = object()
        gpu1 = object()
        store.add("kv0", cpu0, gpu0)
        store.add("kv1", cpu1, gpu1)

        handles = store.prefetch_many(["kv0", "kv1"])
        store.wait_many(["kv0", "kv1"])

        self.assertEqual(runtime.calls[0], ("prefetch", cpu0, gpu0))
        self.assertEqual(runtime.calls[1], ("prefetch", cpu1, gpu1))
        self.assertEqual([handle.wait_calls for handle in handles], [1, 1])
        self.assertEqual(store.block("kv0").state, BlockState.GPU)
        self.assertEqual(store.block("kv1").state, BlockState.GPU)

    def test_many_methods_use_range_batch_for_packed_blocks(self) -> None:
        runtime = FakeRuntime()
        store = OffloadStore(runtime)
        cpu = FakeTensor(128)
        gpu = object()
        store.add("kv0", cpu, gpu, cpu_offset=0, gpu_offset=16, byte_count=8)
        store.add("kv1", cpu, gpu, cpu_offset=32, gpu_offset=48, byte_count=8)

        handles = store.prefetch_many(["kv0", "kv1"])

        self.assertEqual(handles[0], handles[1])
        self.assertEqual(runtime.calls[0][0], "prefetch_ranges")
        self.assertIs(runtime.calls[0][1], cpu)
        self.assertIs(runtime.calls[0][2], gpu)
        self.assertEqual(
            runtime.calls[0][3],
            [
                {"src_offset": 0, "dst_offset": 16, "bytes": 8},
                {"src_offset": 32, "dst_offset": 48, "bytes": 8},
            ],
        )
        store.wait_many(["kv0", "kv1"])
        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.GPU)
        self.assertEqual(store.block("kv1").state, BlockState.GPU)
        self.assertEqual(
            store.transfer_stats_many(["kv0", "kv1"]),
            TransferStats(bytes=16, direct_chunks=2),
        )

    def test_evict_many_uses_reversed_ranges_for_packed_blocks(self) -> None:
        runtime = FakeRuntime()
        store = OffloadStore(runtime)
        cpu = FakeTensor(128)
        gpu = object()
        store.add("kv0", cpu, gpu, cpu_offset=0, gpu_offset=16, byte_count=8)
        store.add("kv1", cpu, gpu, cpu_offset=32, gpu_offset=48, byte_count=8)

        handles = store.evict_many(["kv0", "kv1"])

        self.assertEqual(handles[0], handles[1])
        self.assertEqual(runtime.calls[0][0], "evict_ranges")
        self.assertIs(runtime.calls[0][1], gpu)
        self.assertIs(runtime.calls[0][2], cpu)
        self.assertEqual(
            runtime.calls[0][3],
            [
                {"src_offset": 16, "dst_offset": 0, "bytes": 8},
                {"src_offset": 48, "dst_offset": 32, "bytes": 8},
            ],
        )
        store.wait_many(["kv0", "kv1"])
        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.CPU)
        self.assertEqual(store.block("kv1").state, BlockState.CPU)
        self.assertEqual(
            store.transfer_stats_many(["kv0", "kv1"]),
            TransferStats(bytes=16, relay_chunks=2),
        )

    def test_manager_aliases_point_to_store(self) -> None:
        self.assertIs(OffloadManager, OffloadStore)
        self.assertIs(KVBlockStore, OffloadStore)

    def test_wait_before_transfer_is_noop(self) -> None:
        store = OffloadStore(FakeRuntime())
        store.add("kv0", FakeTensor(1), object())

        store.wait("kv0")

    def test_summarize_transfer_handles_deduplicates_handles(self) -> None:
        handle = FakeHandle(
            "transfer",
            SimpleNamespace(bytes=128, direct_chunks=2, relay_chunks=1),
        )

        stats = summarize_transfer_handles([handle, handle])

        self.assertEqual(stats, TransferStats(bytes=128, direct_chunks=2, relay_chunks=1))
        self.assertEqual(
            stats.as_dict(),
            {"bytes": 128, "direct_chunks": 2, "relay_chunks": 1},
        )

    def test_summarize_transfer_handles_accepts_dict_stats(self) -> None:
        handles = [
            FakeHandle("direct", {"bytes": 64, "direct_chunks": 1}),
            FakeHandle("relay", {"bytes": 32, "relay_chunks": 1}),
            FakeHandle("missing", None),
        ]

        stats = summarize_transfer_handles(handles)

        self.assertEqual(stats, TransferStats(bytes=96, direct_chunks=1, relay_chunks=1))


if __name__ == "__main__":
    unittest.main()
