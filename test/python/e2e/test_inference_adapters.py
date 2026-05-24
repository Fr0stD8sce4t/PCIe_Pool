from __future__ import annotations

import unittest

from turbobus.inference import InferenceKVSlotAdapter, make_contiguous_kv_slots
from turbobus.offload_store import AdapterTransferContext, OffloadBatch, TransferStats
from turbobus.schema import TransferIntent, TransferReceipt, TransferStatusState, WorkloadKind
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


class InferenceKVSlotAdapterTest(unittest.TestCase):
    def test_restore_and_save_use_registered_ranges(self) -> None:
        client = FakeClient()
        cpu = FakeTensor(128)
        gpu = object()
        adapter = InferenceKVSlotAdapter(client, make_context(), cpu, gpu)
        slots = make_contiguous_kv_slots("prefix", 2, 32)

        adapter.register_slots(slots)
        restore_handles = adapter.restore_prefix(["prefix0", "prefix1"])
        save_handles = adapter.save_prefix(["prefix0", "prefix1"])

        self.assertEqual([intent.direction for intent in client.submitted], ["h2d", "d2h"])
        self.assertEqual(
            client.submitted[0].ranges,
            (
                {"src_offset": 0, "dst_offset": 0, "bytes": 32},
                {"src_offset": 32, "dst_offset": 32, "bytes": 32},
            ),
        )
        self.assertEqual(
            client.submitted[1].ranges,
            (
                {"src_offset": 0, "dst_offset": 0, "bytes": 32},
                {"src_offset": 32, "dst_offset": 32, "bytes": 32},
            ),
        )
        self.assertEqual([handle.wait_calls for handle in restore_handles], [1, 1])
        self.assertEqual([handle.wait_calls for handle in save_handles], [1, 1])
        self.assertEqual(adapter.block_ids(), [0, 1])

    def test_submit_and_wait_can_be_called_separately(self) -> None:
        client = FakeClient()
        adapter = InferenceKVSlotAdapter(client, make_context(), FakeTensor(128), object())
        adapter.register_slots(make_contiguous_kv_slots("prefix", 2, 32))

        names, handles = adapter.submit_restore_prefix(["prefix0", "prefix1"])

        self.assertEqual(names, ["prefix0", "prefix1"])
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(client.waited, [])
        adapter.wait_prefix(names)
        self.assertEqual(client.waited, [(client.submitted[0].intent_id, 2.5)])
        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(len(handles), 2)

    def test_transfer_stats_reports_last_prefix_transfer(self) -> None:
        adapter = InferenceKVSlotAdapter(FakeClient(), make_context(), FakeTensor(128), object())
        adapter.register_slots(make_contiguous_kv_slots("prefix", 2, 32))

        adapter.restore_prefix(["prefix0", "prefix1"])

        self.assertEqual(adapter.transfer_stats(["prefix0", "prefix1"]).bytes, 64)
        self.assertEqual(adapter.transfer_stats(["prefix0", "prefix1"]).direct_chunks, 1)
        self.assertEqual(adapter.transfer_stats(["prefix0", "prefix1"]).relay_chunks, 1)

    def test_restore_and_save_batches_return_batch_objects(self) -> None:
        adapter = InferenceKVSlotAdapter(FakeClient(), make_context(), FakeTensor(128), object())
        adapter.register_slots(make_contiguous_kv_slots("prefix", 2, 32))

        restore_batch = adapter.restore_batch(["prefix0", "prefix1"])
        restore_batch.wait()
        self.assertEqual(restore_batch.transfer_stats(), TransferStats(bytes=64, direct_chunks=1, relay_chunks=1))
        save_batch = adapter.save_batch(["prefix0", "prefix1"])
        save_batch.wait()

        self.assertIsInstance(restore_batch, OffloadBatch)
        self.assertIsInstance(save_batch, OffloadBatch)
        self.assertEqual(restore_batch.operation, "restore")
        self.assertEqual(save_batch.operation, "save")
        self.assertEqual(save_batch.transfer_stats(), TransferStats(bytes=64, direct_chunks=1, relay_chunks=1))


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

    def test_layer_range_refs_merge_contiguous_blocks_by_lane(self) -> None:
        tensor = FakeTensor(
            shape=(2, 8, 4),
            stride=(32, 4, 1),
            element_size=2,
        )

        refs = make_vllm_layer_range_refs_from_ids("req0", [1, 2, 3], [tensor])

        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0].lane_id, 0)
        self.assertEqual(refs[0].cpu_slot, 0)
        self.assertEqual(refs[0].cpu_offset, 0)
        self.assertEqual(refs[0].gpu_offset, 8)
        self.assertEqual(refs[0].byte_count, 24)
        self.assertEqual(refs[1].lane_id, 1)
        self.assertEqual(refs[1].cpu_slot, 3)
        self.assertEqual(refs[1].cpu_offset, 24)
        self.assertEqual(refs[1].gpu_offset, 72)
        self.assertEqual(refs[1].byte_count, 24)

    def test_layer_range_refs_keep_noncontiguous_runs_separate(self) -> None:
        tensor = FakeTensor(
            shape=(1, 8, 4),
            stride=(32, 4, 1),
            element_size=2,
        )

        refs = make_vllm_layer_range_refs_from_ids("req0", [1, 3], [tensor])

        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0].block_id, 1)
        self.assertEqual(refs[0].byte_count, 8)
        self.assertEqual(refs[1].block_id, 3)
        self.assertEqual(refs[1].byte_count, 8)

    def test_restore_groups_refs_by_layer(self) -> None:
        client = FakeClient()
        group0 = VllmKVGroup(0, FakeTensor(128), object(), block_bytes=32)
        group1 = VllmKVGroup(1, FakeTensor(128), object(), block_bytes=32)
        adapter = VllmKVSlotAdapter(client, make_context(), [group0, group1])
        refs = make_vllm_layer_block_refs_from_ids("req0", [1], layer_count=2)

        adapter.restore_prefix(refs)
        adapter.save_prefix(refs)
        adapter.restore_prefix(refs)

        self.assertEqual(len(client.submitted), 6)
        self.assertEqual([intent.direction for intent in client.submitted], ["h2d", "h2d", "d2h", "d2h", "h2d", "h2d"])
        self.assertEqual(client.submitted[0].ranges, ({"src_offset": 0, "dst_offset": 32, "bytes": 32},))
        self.assertEqual(client.submitted[1].ranges, ({"src_offset": 0, "dst_offset": 32, "bytes": 32},))
        self.assertEqual(client.submitted[0].metadata["group_id"], 0)
        self.assertEqual(client.submitted[1].metadata["group_id"], 1)

    def test_restore_uses_total_chunk_metadata_across_layers(self) -> None:
        client = FakeClient()
        groups = [
            VllmKVGroup(index, FakeTensor(1024), object(), block_bytes=128)
            for index in range(4)
        ]
        adapter = VllmKVSlotAdapter(client, make_context(metadata={"chunk_bytes": 128}), groups)
        refs = make_vllm_layer_block_refs_from_ids("req0", [1, 2], layer_count=4)

        adapter.restore_prefix(refs)

        self.assertEqual(len(client.submitted), 4)
        self.assertTrue(all(intent.direction == "h2d" for intent in client.submitted))
        self.assertTrue(all(intent.metadata["chunk_bytes"] == 128 for intent in client.submitted))

    def test_restore_submits_all_layers_before_waiting(self) -> None:
        client = FakeClient()
        group0 = VllmKVGroup(0, FakeTensor(128), object(), block_bytes=32)
        group1 = VllmKVGroup(1, FakeTensor(128), object(), block_bytes=32)
        adapter = VllmKVSlotAdapter(client, make_context(), [group0, group1])
        refs = make_vllm_layer_block_refs_from_ids("req0", [1], layer_count=2)

        adapter.restore_prefix(refs)

        self.assertEqual(len(client.submitted), 2)
        self.assertEqual(
            client.waited,
            [
                (client.submitted[0].intent_id, 2.5),
                (client.submitted[1].intent_id, 2.5),
            ],
        )

    def test_transfer_stats_sums_groups(self) -> None:
        group0 = VllmKVGroup(0, FakeTensor(128), object(), block_bytes=32)
        group1 = VllmKVGroup(1, FakeTensor(128), object(), block_bytes=32)
        adapter = VllmKVSlotAdapter(FakeClient(), make_context(), [group0, group1])
        refs = make_vllm_layer_block_refs_from_ids("req0", [1], layer_count=2)

        adapter.restore_prefix(refs)
        stats = adapter.transfer_stats(refs)

        self.assertEqual(stats.bytes, 64)
        self.assertEqual(stats.direct_chunks, 2)
        self.assertEqual(stats.relay_chunks, 2)


def make_context(**overrides) -> AdapterTransferContext:
    values = {
        "job_id": "job-1",
        "session_id": "session-1",
        "cpu_buffer_id": "cpu-buffer",
        "gpu_buffer_id": "gpu-buffer",
        "workload_kind": WorkloadKind.KV_CACHE,
        "metadata": {"chunk_bytes": 32},
        "intent_prefix": "kv",
        "wait_timeout_seconds": 2.5,
    }
    values.update(overrides)
    return AdapterTransferContext(**values)


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
