from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
import unittest

from turbobus.offload_store import (
    AdapterTransferContext,
    BlockState,
    OffloadBatch,
    OffloadBlockInfo,
    OffloadStore,
    ReceiptTransferHandle,
    TransferStats,
    summarize_transfer_handles,
    transfer_stats_from_receipt,
)
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


class FakeHandle:
    def __init__(self, stats=None) -> None:
        self.wait_calls = 0
        self.stats = stats

    def wait(self) -> None:
        self.wait_calls += 1


class OffloadStoreTest(unittest.TestCase):
    def test_add_tracks_named_block(self) -> None:
        store = make_store()
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
        store = make_store()

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
        store = make_store()

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
        store = make_store()
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
        self.assertEqual(info.as_dict()["state"], "cpu")

    def test_block_infos_accepts_optional_name_filter(self) -> None:
        store = make_store()
        store.add("kv0", FakeTensor(1), object())
        store.add("kv1", FakeTensor(1), object())

        self.assertEqual([info.name for info in store.block_infos(["kv1"])], ["kv1"])
        self.assertEqual([info.name for info in store.block_infos()], ["kv0", "kv1"])

    def test_block_store_aliases_and_state_helpers(self) -> None:
        store = make_store()

        block = store.add_block("kv0", FakeTensor(64), object())
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
        store = make_store()

        with self.assertRaises(ValueError):
            store.add("kv0", FakeTensor(128), object(), cpu_offset=-1)
        with self.assertRaises(ValueError):
            store.add("kv1", FakeTensor(128), object(), byte_count=0)

    def test_duplicate_name_is_rejected(self) -> None:
        store = make_store()
        store.add("kv0", FakeTensor(1), object())

        with self.assertRaises(ValueError):
            store.add("kv0", FakeTensor(1), object())

    def test_prefetch_and_evict_submit_transfer_intent_and_record_receipts(self) -> None:
        client = FakeClient()
        store = make_store(client)
        store.add("kv0", FakeTensor(64), object())

        prefetch = store.prefetch("kv0")
        self.assertEqual(store.block("kv0").last_operation, "prefetch")
        self.assertEqual(store.block("kv0").state, BlockState.PREFETCHING)
        self.assertEqual(
            store.stats("kv0"),
            TransferStats(bytes=64, direct_chunks=1, relay_chunks=1),
        )

        store.wait("kv0")
        self.assertEqual(prefetch.wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.GPU)

        evict = store.evict("kv0")
        store.wait("kv0")

        self.assertEqual([intent.direction for intent in client.submitted], ["h2d", "d2h"])
        self.assertEqual(client.submitted[0].source_buffer_id, "cpu-buffer")
        self.assertEqual(client.submitted[0].destination_buffer_id, "gpu-buffer")
        self.assertEqual(client.submitted[1].source_buffer_id, "gpu-buffer")
        self.assertEqual(client.submitted[1].destination_buffer_id, "cpu-buffer")
        self.assertEqual(evict.wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.CPU)

    def test_many_methods_submit_and_wait_in_order(self) -> None:
        client = FakeClient()
        store = make_store(client)
        cpu0 = FakeTensor(1)
        cpu1 = FakeTensor(1)
        store.add("kv0", cpu0, object())
        store.add("kv1", cpu1, object())

        handles = store.prefetch_many(["kv0", "kv1"])
        store.wait_many(["kv0", "kv1"])

        self.assertEqual(len(client.submitted), 2)
        self.assertEqual([intent.ranges for intent in client.submitted], [
            ({"src_offset": 0, "dst_offset": 0, "bytes": 1},),
            ({"src_offset": 0, "dst_offset": 0, "bytes": 1},),
        ])
        self.assertEqual([handle.wait_calls for handle in handles], [1, 1])
        self.assertEqual(store.block("kv0").state, BlockState.GPU)
        self.assertEqual(store.block("kv1").state, BlockState.GPU)

    def test_many_methods_use_one_intent_for_packed_blocks(self) -> None:
        client = FakeClient()
        store = make_store(client)
        cpu = FakeTensor(128)
        gpu = object()
        store.add("kv0", cpu, gpu, cpu_offset=0, gpu_offset=16, byte_count=8)
        store.add("kv1", cpu, gpu, cpu_offset=32, gpu_offset=48, byte_count=8)

        handles = store.prefetch_many(["kv0", "kv1"])

        self.assertEqual(handles[0], handles[1])
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(
            client.submitted[0].ranges,
            (
                {"src_offset": 0, "dst_offset": 16, "bytes": 8},
                {"src_offset": 32, "dst_offset": 48, "bytes": 8},
            ),
        )
        self.assertEqual(client.submitted[0].workload_kind, WorkloadKind.KV_CACHE)
        self.assertEqual(client.submitted[0].policy_hints, {})

        store.wait_many(["kv0", "kv1"])

        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.GPU)
        self.assertEqual(store.block("kv1").state, BlockState.GPU)
        self.assertEqual(
            store.transfer_stats_many(["kv0", "kv1"]),
            TransferStats(bytes=16, direct_chunks=1, relay_chunks=1),
        )

    def test_submit_prefetch_many_returns_batch_object(self) -> None:
        store = make_store()
        cpu = FakeTensor(128)
        gpu = object()
        store.add("kv0", cpu, gpu, cpu_offset=0, gpu_offset=16, byte_count=8)
        store.add("kv1", cpu, gpu, cpu_offset=32, gpu_offset=48, byte_count=8)

        batch = store.submit_prefetch_many(["kv0", "kv1"])

        self.assertIsInstance(batch, OffloadBatch)
        self.assertEqual(batch.operation, "prefetch")
        self.assertEqual(batch.names, ("kv0", "kv1"))
        self.assertEqual(batch.handles[0], batch.handles[1])

        batch.wait()

        self.assertEqual(batch.handles[0].wait_calls, 1)
        self.assertEqual(batch.transfer_stats(), TransferStats(bytes=16, direct_chunks=1, relay_chunks=1))
        self.assertEqual(
            [info.state for info in batch.block_infos()],
            [BlockState.GPU, BlockState.GPU],
        )
        self.assertEqual(batch.as_dict()["transfer_stats"]["bytes"], 16)

    def test_empty_batch_object_is_noop(self) -> None:
        store = make_store()

        batch = store.submit_evict_many([])
        batch.wait()

        self.assertEqual(batch.operation, "evict")
        self.assertEqual(batch.names, ())
        self.assertEqual(batch.handles, ())
        self.assertEqual(batch.transfer_stats(), TransferStats())

    def test_evict_many_uses_reversed_ranges_for_packed_blocks(self) -> None:
        client = FakeClient()
        store = make_store(client)
        cpu = FakeTensor(128)
        gpu = object()
        store.add("kv0", cpu, gpu, cpu_offset=0, gpu_offset=16, byte_count=8)
        store.add("kv1", cpu, gpu, cpu_offset=32, gpu_offset=48, byte_count=8)

        handles = store.evict_many(["kv0", "kv1"])

        self.assertEqual(handles[0], handles[1])
        self.assertEqual(client.submitted[0].direction, "d2h")
        self.assertEqual(
            client.submitted[0].ranges,
            (
                {"src_offset": 16, "dst_offset": 0, "bytes": 8},
                {"src_offset": 48, "dst_offset": 32, "bytes": 8},
            ),
        )
        store.wait_many(["kv0", "kv1"])
        self.assertEqual(handles[0].wait_calls, 1)
        self.assertEqual(store.block("kv0").state, BlockState.CPU)
        self.assertEqual(store.block("kv1").state, BlockState.CPU)

    def test_wait_before_transfer_is_noop(self) -> None:
        store = make_store()
        store.add("kv0", FakeTensor(1), object())

        store.wait("kv0")

    def test_adapter_context_rejects_physical_policy_hints(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical paths"):
            make_context(policy_hints={"relay_gpu": 1})

    def test_receipt_handle_rejects_wait_receipt_mismatch(self) -> None:
        client = MismatchedWaitClient()
        intent = make_intent("intent-1")
        handle = ReceiptTransferHandle(
            client=client,
            intent=intent,
            receipt=make_receipt(intent, receipt_id="submitted"),
        )

        with self.assertRaisesRegex(ValueError, "intent_id"):
            handle.wait()

    def test_transfer_stats_from_receipt_counts_path_split(self) -> None:
        intent = make_intent("intent-1", total_bytes=96)
        receipt = TransferReceipt(
            receipt_id="receipt-1",
            ticket_id="ticket-1",
            intent_id=intent.intent_id,
            decision_id="decision-1",
            topology_snapshot_id="topology-1",
            job_id=intent.job_id,
            session_id=intent.session_id,
            state=TransferStatusState.COMPLETE,
            bytes_total=96,
            bytes_completed=96,
            path_stats=(
                {"kind": "direct", "bytes": 32, "chunk_count": 2},
                {"kind": "relay", "bytes": 64, "chunks": 4},
            ),
        )

        self.assertEqual(
            transfer_stats_from_receipt(receipt),
            TransferStats(bytes=96, direct_chunks=2, relay_chunks=4),
        )

    def test_summarize_transfer_handles_deduplicates_handles(self) -> None:
        handle = FakeHandle(SimpleNamespace(bytes=128, direct_chunks=2, relay_chunks=1))

        stats = summarize_transfer_handles([handle, handle])

        self.assertEqual(stats, TransferStats(bytes=128, direct_chunks=2, relay_chunks=1))
        self.assertEqual(
            stats.as_dict(),
            {"bytes": 128, "direct_chunks": 2, "relay_chunks": 1},
        )

    def test_summarize_transfer_handles_accepts_dict_stats(self) -> None:
        handles = [
            FakeHandle({"bytes": 64, "direct_chunks": 1}),
            FakeHandle({"bytes": 32, "relay_chunks": 1}),
            FakeHandle(None),
        ]

        stats = summarize_transfer_handles(handles)

        self.assertEqual(stats, TransferStats(bytes=96, direct_chunks=1, relay_chunks=1))


class MismatchedWaitClient(FakeClient):
    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> TransferReceipt:
        return make_receipt(make_intent("other-intent"), receipt_id="receipt-other")


def make_context(**overrides) -> AdapterTransferContext:
    values = {
        "job_id": "job-1",
        "session_id": "session-1",
        "cpu_buffer_id": "cpu-buffer",
        "gpu_buffer_id": "gpu-buffer",
        "workload_kind": WorkloadKind.KV_CACHE,
        "metadata": {"chunk_bytes": 32},
        "intent_prefix": "test-intent",
        "wait_timeout_seconds": 2.5,
    }
    values.update(overrides)
    return AdapterTransferContext(**values)


def make_store(client: FakeClient | None = None) -> OffloadStore:
    return OffloadStore(client or FakeClient(), make_context())


def make_intent(intent_id: str, *, total_bytes: int = 16) -> TransferIntent:
    return TransferIntent(
        intent_id=intent_id,
        job_id="job-1",
        session_id="session-1",
        source_buffer_id="cpu-buffer",
        destination_buffer_id="gpu-buffer",
        direction="h2d",
        total_bytes=total_bytes,
        ranges=({"src_offset": 0, "dst_offset": 0, "bytes": total_bytes},),
        workload_kind=WorkloadKind.KV_CACHE,
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
        metadata={"payload": asdict(intent)},
    )


if __name__ == "__main__":
    unittest.main()
