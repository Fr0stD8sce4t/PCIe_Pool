from __future__ import annotations

import unittest
from types import SimpleNamespace

from turbobus.vllm_integration import VllmTurboBusIntegration, extract_vllm_block_ids


class FakeTensor:
    def __init__(self, *, stride, element_size: int = 1) -> None:
        self.shape = (2, 8, 4)
        self._stride = stride
        self._element_size = element_size

    def stride(self, dim: int) -> int:
        return self._stride[dim]

    def element_size(self) -> int:
        return self._element_size


class FakeHandle:
    def wait(self) -> None:
        pass


class FakeRuntime:
    target_gpu = 6

    def __init__(self) -> None:
        self.calls = []

    def fetch_ranges_to_gpu(self, cpu_tensor, gpu_tensor, ranges):
        self.calls.append(("restore", cpu_tensor, gpu_tensor, ranges))
        return FakeHandle()

    def offload_ranges_to_cpu(self, gpu_tensor, cpu_tensor, ranges):
        self.calls.append(("save", gpu_tensor, cpu_tensor, ranges))
        return FakeHandle()


class FakeRequest:
    request_id = "req0"


class FakeBlocks:
    def __init__(self, ids) -> None:
        self.ids = ids

    def get_block_ids(self, allow_none: bool = False):
        return self.ids


class FakeRunner:
    def __init__(self) -> None:
        self.kv_caches = [
            FakeTensor(stride=(64, 16, 1), element_size=2),
            FakeTensor(stride=(64, 16, 1), element_size=2),
        ]

    def initialize_kv_cache(self, config):
        return "initialized"


class FakeManager:
    def __init__(self) -> None:
        self.calls = 0

    def allocate_slots(self, request):
        self.calls += 1
        if self.calls == 1:
            return FakeBlocks(([1, 3],))
        return FakeBlocks(([3, 5],))


class VllmTurboBusIntegrationTest(unittest.TestCase):
    def test_extract_block_ids(self) -> None:
        self.assertEqual(extract_vllm_block_ids(FakeBlocks(([1, None, 3], []))), ((1, 3), ()))
        self.assertEqual(extract_vllm_block_ids(None), tuple())
        self.assertEqual(extract_vllm_block_ids([1, None, 3]), ((1, 3),))
        self.assertEqual(extract_vllm_block_ids(((1, 2), (3, None))), ((1, 2), (3,)))
        self.assertEqual(extract_vllm_block_ids(SimpleNamespace(block_ids=[4, 5])), ((4, 5),))

    def test_hooks_capture_real_runner_cache_and_allocated_blocks(self) -> None:
        runtime = FakeRuntime()
        integration = VllmTurboBusIntegration(runtime, cpu_backings=[object(), object()])
        callbacks = []
        integration.set_allocation_callback(
            lambda integration, request, blocks, event: callbacks.append(event)
        )
        integration.install_on_classes(FakeRunner, FakeManager)

        runner = FakeRunner()
        self.assertEqual(runner.initialize_kv_cache("config"), "initialized")

        manager = FakeManager()
        manager.allocate_slots(FakeRequest())
        manager.allocate_slots(FakeRequest())

        self.assertEqual(integration.state.kv_cache_config, "config")
        self.assertEqual(integration.block_ids_for_request("req0"), (1, 3, 5))
        self.assertEqual(integration.state.allocations["req0"].event_count, 2)
        self.assertEqual([event.block_ids for event in callbacks], [(1, 3), (1, 3, 5)])

        integration.restore_request_prefix("req0")

        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.calls[0][0], "restore")
        self.assertEqual(
            runtime.calls[0][3],
            [
                {"src_offset": 0, "dst_offset": 32, "bytes": 32},
                {"src_offset": 32, "dst_offset": 96, "bytes": 32},
                {"src_offset": 64, "dst_offset": 160, "bytes": 32},
            ],
        )
        self.assertEqual(runtime.calls[1][3], runtime.calls[0][3])


if __name__ == "__main__":
    unittest.main()
