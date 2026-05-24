from __future__ import annotations

import unittest
from types import SimpleNamespace

from turbobus.offload_store import AdapterTransferContext
from turbobus.schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind
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

    def test_hooks_capture_runner_cache_and_submit_restore_intents(self) -> None:
        client = FakeClient()
        integration = VllmTurboBusIntegration(
            client,
            AdapterTransferContext(
                job_id="job-1",
                session_id="session-1",
                cpu_buffer_id="cpu-buffer",
                gpu_buffer_id="gpu-buffer",
                workload_kind=WorkloadKind.KV_CACHE,
                intent_prefix="vllm",
                wait_timeout_seconds=2.5,
            ),
            cpu_backings=[object(), object()],
        )
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

        self.assertEqual(len(client.submitted), 2)
        self.assertTrue(all(intent.direction == "h2d" for intent in client.submitted))
        self.assertEqual(
            client.submitted[0].ranges,
            (
                {"src_offset": 0, "dst_offset": 32, "bytes": 32},
                {"src_offset": 32, "dst_offset": 96, "bytes": 32},
                {"src_offset": 64, "dst_offset": 160, "bytes": 32},
            ),
        )
        self.assertEqual(client.submitted[1].ranges, client.submitted[0].ranges)
        self.assertEqual(
            client.waited,
            [
                (client.submitted[0].intent_id, 2.5),
                (client.submitted[1].intent_id, 2.5),
            ],
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
