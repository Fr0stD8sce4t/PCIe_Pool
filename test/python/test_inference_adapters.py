from __future__ import annotations

import unittest

from turbobus.inference import InferenceKVSlotAdapter, make_contiguous_kv_slots
from turbobus.vllm import (
    VllmKVGroup,
    VllmKVSlotAdapter,
    block_bytes_from_vllm_kv_tensor,
    make_vllm_layer_block_refs_from_ids,
    make_vllm_layer_range_refs_from_ids,
)


class FakeTensor:
    def __init__(
        self,
        bytes_: int = 0,
        *,
        shape=None,
        stride=None,
        element_size: int = 1,
    ) -> None:
        self._bytes = bytes_
        self.shape = shape or (bytes_,)
        self._stride = stride
        self._element_size = element_size

    def numel(self) -> int:
        return self._bytes

    def element_size(self) -> int:
        return self._element_size

    def stride(self, dim: int) -> int:
        if self._stride is None:
            raise ValueError("stride is not set")
        return self._stride[dim]


class FakeHandle:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1


class FakeRuntime:
    target_gpu = 6

    def __init__(self) -> None:
        self.calls = []

    def fetch_ranges_to_gpu(self, cpu_tensor, gpu_tensor, ranges):
        handle = FakeHandle()
        self.calls.append(("prefetch_ranges", cpu_tensor, gpu_tensor, ranges, handle))
        return handle

    def offload_ranges_to_cpu(self, gpu_tensor, cpu_tensor, ranges):
        handle = FakeHandle()
        self.calls.append(("evict_ranges", gpu_tensor, cpu_tensor, ranges, handle))
        return handle


class InferenceKVSlotAdapterTest(unittest.TestCase):
    def test_restore_and_save_use_registered_ranges(self) -> None:
        runtime = FakeRuntime()
        cpu = FakeTensor(128)
        gpu = object()
        adapter = InferenceKVSlotAdapter(runtime, cpu, gpu)
        slots = make_contiguous_kv_slots("prefix", 2, 32)

        adapter.register_slots(slots)
        restore_handles = adapter.restore_prefix(["prefix0", "prefix1"])
        save_handles = adapter.save_prefix(["prefix0", "prefix1"])

        self.assertEqual(runtime.calls[0][0], "prefetch_ranges")
        self.assertEqual(
            runtime.calls[0][3],
            [
                {"src_offset": 0, "dst_offset": 0, "bytes": 32},
                {"src_offset": 32, "dst_offset": 32, "bytes": 32},
            ],
        )
        self.assertEqual(runtime.calls[1][0], "evict_ranges")
        self.assertEqual(
            runtime.calls[1][3],
            [
                {"src_offset": 0, "dst_offset": 0, "bytes": 32},
                {"src_offset": 32, "dst_offset": 32, "bytes": 32},
            ],
        )
        self.assertEqual(runtime.calls[0][4].wait_calls, 1)
        self.assertEqual(runtime.calls[1][4].wait_calls, 1)
        self.assertEqual(len(restore_handles), 2)
        self.assertEqual(len(save_handles), 2)


class VllmKVSlotAdapterTest(unittest.TestCase):
    def test_block_bytes_uses_vllm_block_stride(self) -> None:
        tensor = FakeTensor(shape=(2, 9944, 16, 8, 128), stride=(162922496, 32768, 2048, 256, 1), element_size=2)

        self.assertEqual(block_bytes_from_vllm_kv_tensor(tensor), 65536)

    def test_layer_refs_expand_block_ids_across_layers(self) -> None:
        refs = make_vllm_layer_block_refs_from_ids("req0", [1, 3], layer_count=2)

        self.assertEqual(
            [(ref.group_id, ref.block_id, ref.cpu_slot, ref.gpu_slot) for ref in refs],
            [(0, 1, 0, 1), (0, 3, 1, 3), (1, 1, 0, 1), (1, 3, 1, 3)],
        )

    def test_layer_range_refs_expand_kv_lanes(self) -> None:
        tensor = FakeTensor(
            shape=(2, 8, 4),
            stride=(32, 4, 1),
            element_size=2,
        )

        refs = make_vllm_layer_range_refs_from_ids("req0", [1], [tensor])

        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0].lane_id, 0)
        self.assertEqual(refs[0].cpu_slot, 0)
        self.assertEqual(refs[0].gpu_offset, 8)
        self.assertEqual(refs[1].lane_id, 1)
        self.assertEqual(refs[1].cpu_slot, 1)
        self.assertEqual(refs[1].gpu_offset, 72)

    def test_restore_groups_refs_by_layer(self) -> None:
        runtime = FakeRuntime()
        group0 = VllmKVGroup(0, FakeTensor(128), object(), block_bytes=32)
        group1 = VllmKVGroup(1, FakeTensor(128), object(), block_bytes=32)
        adapter = VllmKVSlotAdapter(runtime, [group0, group1])
        refs = make_vllm_layer_block_refs_from_ids("req0", [1], layer_count=2)

        adapter.restore_prefix(refs)
        adapter.save_prefix(refs)
        adapter.restore_prefix(refs)

        self.assertEqual(len(runtime.calls), 6)
        self.assertEqual(runtime.calls[0][0], "prefetch_ranges")
        self.assertEqual(runtime.calls[0][3], [{"src_offset": 0, "dst_offset": 32, "bytes": 32}])
        self.assertEqual(runtime.calls[1][3], [{"src_offset": 0, "dst_offset": 32, "bytes": 32}])
        self.assertEqual(runtime.calls[2][0], "evict_ranges")
        self.assertEqual(runtime.calls[4][0], "prefetch_ranges")


if __name__ == "__main__":
    unittest.main()
