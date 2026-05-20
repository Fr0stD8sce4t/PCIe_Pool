from __future__ import annotations

import unittest

from turbobus import OffloadStore


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
        self.assertEqual(block.bytes, 128)
        self.assertEqual(store.names(), ["kv0"])
        self.assertIs(store.block("kv0"), block)

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
        self.assertEqual(store.stats("kv0"), {"label": "prefetch"})

        evict = store.evict("kv0")
        self.assertEqual(runtime.calls[-1], ("evict", gpu, cpu))
        self.assertEqual(store.block("kv0").last_operation, "evict")
        self.assertEqual(store.stats("kv0"), {"label": "evict"})

        store.wait("kv0")
        self.assertEqual(evict.wait_calls, 1)
        self.assertEqual(prefetch.wait_calls, 0)

    def test_wait_before_transfer_is_noop(self) -> None:
        store = OffloadStore(FakeRuntime())
        store.add("kv0", FakeTensor(1), object())

        store.wait("kv0")


if __name__ == "__main__":
    unittest.main()
