from __future__ import annotations

import unittest

from turbobus import BlockState, KVBlockStore, OffloadManager, OffloadStore


class FakeTensor:
    def __init__(self, bytes_: int) -> None:
        self._bytes = bytes_

    def numel(self) -> int:
        return self._bytes

    def element_size(self) -> int:
        return 1


class FakeHandle:
    def __init__(self, label: str) -> None:
        self.label = label
        self.wait_calls = 0
        self.stats = {"label": label}

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

        store.wait("kv0")
        self.assertEqual(prefetch.wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.GPU)

        evict = store.evict("kv0")
        self.assertEqual(runtime.calls[-1], ("evict", gpu, cpu))
        self.assertEqual(store.block("kv0").last_operation, "evict")
        self.assertEqual(store.block("kv0").state, BlockState.EVICTING)
        self.assertEqual(store.stats("kv0"), {"label": "evict"})

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

    def test_manager_aliases_point_to_store(self) -> None:
        self.assertIs(OffloadManager, OffloadStore)
        self.assertIs(KVBlockStore, OffloadStore)

    def test_wait_before_transfer_is_noop(self) -> None:
        store = OffloadStore(FakeRuntime())
        store.add("kv0", FakeTensor(1), object())

        store.wait("kv0")


if __name__ == "__main__":
    unittest.main()
