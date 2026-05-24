from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
import unittest
from unittest import mock

from turbobus.schema import TransferReceipt, TransferStatusState, WorkloadKind
from turbobus.adapters import vllm_kv_connector
from turbobus.vllm_kv_connector import (
    TurboBusCPUBackingPool,
    TurboBusConnector,
    TurboBusConnectorConfig,
    TurboBusConnectorMetadata,
    TurboBusPrefixStore,
    TurboBusRequestMetadata,
    TurboBusSavedPrefix,
    clear_connector_events,
    clear_saved_prefixes,
    get_connector_events,
    get_saved_prefix,
)


class FakeCacheConfig:
    block_size = 16


class FakeTransferConfig:
    def __init__(self, extra=None):
        self.extra = extra or {}
        self.engine_id = self.extra.get("engine_id")

    def get_from_extra_config(self, key, default):
        return self.extra.get(key, default)


class FakeVllmConfig:
    def __init__(self, extra=None):
        self.cache_config = FakeCacheConfig()
        self.kv_transfer_config = FakeTransferConfig(extra)


class FakeRequest:
    request_id = "req0"
    num_tokens = 128

    def __init__(self, params=None):
        self.kv_transfer_params = params or {}


class FakeBlocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self, allow_none=False):
        return self.ids


class FakeTensor:
    shape = (1, 8, 4)

    def stride(self, dim):
        return (32, 4, 1)[dim]

    def element_size(self):
        return 2


class FakeClient:
    def __init__(self, path_stats=None, fallback_reasons=None):
        self.path_stats = list(path_stats or [])
        self.fallback_reasons = list(fallback_reasons or [])
        self.submitted = []
        self.receipts = {}

    def submit_transfer_intent(self, intent):
        self.submitted.append(intent)
        index = len(self.submitted)
        path_stats = (
            self.path_stats.pop(0)
            if self.path_stats
            else ({"kind": "direct", "bytes": intent.total_bytes, "chunk_count": 1},)
        )
        fallback_reason = self.fallback_reasons.pop(0) if self.fallback_reasons else ""
        receipt = TransferReceipt(
            receipt_id=f"receipt-{index}",
            ticket_id=f"ticket-{index}",
            intent_id=intent.intent_id,
            decision_id=f"decision-{index}",
            topology_snapshot_id=f"topology-{index}",
            job_id=intent.job_id,
            session_id=intent.session_id,
            state=TransferStatusState.COMPLETE,
            bytes_total=intent.total_bytes,
            bytes_completed=intent.total_bytes,
            path_stats=tuple(path_stats),
            metadata={"fallback_reason": fallback_reason} if fallback_reason else {},
        )
        self.receipts[intent.intent_id] = receipt
        return receipt

    def wait_transfer_receipt(self, intent_id, timeout_seconds=None):
        return self.receipts[str(intent_id)]


class TurboBusConnectorTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_connector_events()
        clear_saved_prefixes()

    def make_connector(self, extra=None, client=None):
        defaults = {
            "turbobus.job_id": "job-a",
            "turbobus.session_id": "session-a",
            "turbobus.cpu_buffer_id": "cpu-buffer",
            "turbobus.gpu_buffer_id": "gpu-buffer",
            "turbobus.daemon_socket_path": "/tmp/turbobusd.sock",
        }
        defaults.update(extra or {})
        connector = TurboBusConnector(
            FakeVllmConfig(defaults),
            role="scheduler",
            kv_cache_config=object(),
        )
        if client is not None:
            connector.client = client
        return connector

    def test_prefix_store_replaces_saved_prefix_by_key(self) -> None:
        store = TurboBusPrefixStore()
        first = TurboBusSavedPrefix("key", [object()], block_count=1, matched_tokens=16)
        self.assertEqual(store.put(first), [])
        previous = store.put(TurboBusSavedPrefix("key", [object()], block_count=2, matched_tokens=32))

        self.assertEqual(previous, [first])
        self.assertEqual(store.get("key").block_count, 2)

    def test_prefix_store_isolates_same_key_by_session(self) -> None:
        store = TurboBusPrefixStore()
        first = TurboBusSavedPrefix("key", [object()], block_count=1, matched_tokens=16, session_id="a")
        second = TurboBusSavedPrefix("key", [object()], block_count=2, matched_tokens=32, session_id="b")

        self.assertEqual(store.put(first), [])
        self.assertEqual(store.put(second), [])

        self.assertIs(store.get("key", "a"), first)
        self.assertIs(store.get("key", "b"), second)
        self.assertIsNone(store.get("key"))

    def test_clear_saved_prefixes_can_clear_one_session(self) -> None:
        first = TurboBusSavedPrefix("key", [object()], block_count=1, matched_tokens=16, session_id="a")
        second = TurboBusSavedPrefix("key", [object()], block_count=2, matched_tokens=32, session_id="b")
        self.assertEqual(TurboBusPrefixStore().put(first), [])
        self.assertEqual(vllm_kv_connector._store_saved_prefix(first), [])
        self.assertEqual(vllm_kv_connector._store_saved_prefix(second), [])

        clear_saved_prefixes("a")

        self.assertIsNone(get_saved_prefix("key", "a"))
        self.assertEqual(get_saved_prefix("key", "b").block_count, 2)

    def test_connector_events_are_readable_and_clearable(self) -> None:
        self.make_connector()

        events = get_connector_events()
        self.assertEqual(events[-1]["event"], "init")
        self.assertEqual(events[-1]["job_id"], "job-a")

        clear_connector_events()
        self.assertEqual(get_connector_events(), [])

    def test_cpu_backing_pool_reuses_released_backings(self) -> None:
        pool = TurboBusCPUBackingPool()
        first = [object(), object()]
        kv_caches = [mock.Mock(), mock.Mock()]

        with (
            mock.patch(
                "turbobus.vllm_kv_connector._backing_signature",
                return_value=((2, 64), (2, 64)),
            ),
            mock.patch.object(TurboBusCPUBackingPool, "_allocate", return_value=first) as allocate,
        ):
            acquired, reused = pool.acquire(2, kv_caches)
            pool.release(2, kv_caches, acquired)
            second, second_reused = pool.acquire(2, kv_caches)

        allocate.assert_called_once()
        self.assertEqual(second, first)
        self.assertIs(second[0], first[0])
        self.assertFalse(reused)
        self.assertTrue(second_reused)

    def test_connector_config_reads_daemon_identity_and_ignores_physical_route_keys(self) -> None:
        extra = {
            "turbobus.job_id": "job-extra",
            "turbobus.session_id": "session-extra",
            "turbobus.cpu_buffer_id": "cpu-extra",
            "turbobus.gpu_buffer_id": "gpu-extra",
            "turbobus.chunk_bytes": "4194304",
            "turbobus.daemon_socket_path": "/tmp/turbobusd.sock",
            "turbobus.wait_timeout_seconds": "2.5",
            "turbobus.restore_block_limit": "8",
            "turbobus.restore_enabled": "true",
            "turbobus.max_saved_prefixes": "2",
        }

        config = TurboBusConnectorConfig.from_vllm_config(FakeVllmConfig(extra))

        self.assertEqual(config.job_id, "job-extra")
        self.assertEqual(config.session_id, "session-extra")
        self.assertEqual(config.cpu_buffer_id, "cpu-extra")
        self.assertEqual(config.gpu_buffer_id, "gpu-extra")
        self.assertEqual(config.chunk_bytes, 4194304)
        self.assertEqual(config.daemon_socket_path, "/tmp/turbobusd.sock")
        self.assertEqual(config.wait_timeout_seconds, 2.5)
        self.assertEqual(config.restore_block_limit, 8)
        self.assertTrue(config.restore_enabled)
        self.assertEqual(config.max_saved_prefixes, 2)
        self.assertEqual(
            {field.name for field in fields(TurboBusConnectorConfig)},
            {
                "job_id",
                "session_id",
                "cpu_buffer_id",
                "gpu_buffer_id",
                "chunk_bytes",
                "daemon_socket_path",
                "wait_timeout_seconds",
                "restore_block_limit",
                "restore_enabled",
                "max_saved_prefixes",
            },
        )

    def test_connector_requires_piecewise_cudagraph_for_layer_lifecycle(self) -> None:
        self.assertTrue(TurboBusConnector.requires_piecewise_for_cudagraph({}))

    def test_reports_explicit_external_match(self) -> None:
        connector = self.make_connector({"turbobus.restore_enabled": True})
        vllm_kv_connector._store_saved_prefix(
            TurboBusSavedPrefix("default", [object()], block_count=8, matched_tokens=96, session_id="session-a")
        )
        request = FakeRequest(
            {
                "turbobus.do_restore": True,
                "turbobus.matched_tokens": 96,
            }
        )

        self.assertEqual(connector.get_num_new_matched_tokens(request, 32), (64, True))
        self.assertEqual(connector.state.events[-1]["event"], "match")
        self.assertEqual(connector.state.events[-1]["available_tokens"], 64)

    def test_does_not_report_external_match_until_restore_enabled(self) -> None:
        connector = self.make_connector()
        vllm_kv_connector._store_saved_prefix(
            TurboBusSavedPrefix("default", [object()], block_count=8, matched_tokens=96, session_id="session-a")
        )
        request = FakeRequest(
            {
                "turbobus.do_restore": True,
                "turbobus.matched_tokens": 96,
            }
        )

        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertEqual(connector.state.events[-1]["event"], "match_skipped")

    def test_records_allocated_blocks_for_connector_metadata(self) -> None:
        connector = self.make_connector({"turbobus.restore_block_limit": 4})
        vllm_kv_connector._store_saved_prefix(
            TurboBusSavedPrefix("default", [object()], block_count=4, matched_tokens=128, session_id="session-a")
        )
        request = FakeRequest({"turbobus.do_restore": True, "turbobus.matched_tokens": 128})
        blocks = FakeBlocks(([1, 2, 3, 4, 5, 6],))

        connector.update_state_after_alloc(request, blocks, num_external_tokens=128)
        metadata = connector.build_connector_meta(scheduler_output=object())

        self.assertIsInstance(metadata, TurboBusConnectorMetadata)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata.requests[0].request_id, "req0")
        self.assertEqual(metadata.requests[0].prefix_key, "default")
        self.assertEqual(metadata.requests[0].block_ids, (1, 2, 3, 4))
        self.assertEqual(metadata.requests[0].block_count, 4)
        self.assertEqual(connector.state.pending_loads, {})

    def test_build_connector_meta_can_collect_save_from_scheduler_output(self) -> None:
        connector = self.make_connector()
        request = SimpleNamespace(
            req_id="req1",
            new_block_ids=[4, 5, 6],
            kv_transfer_params={
                "turbobus.do_save": True,
                "turbobus.prefix_key": "scheduled",
                "turbobus.save_blocks": 2,
                "turbobus.matched_tokens": 32,
            },
        )
        scheduler_output = SimpleNamespace(scheduled_new_reqs=[request])

        metadata = connector.build_connector_meta(scheduler_output)

        self.assertEqual(len(metadata.save_requests), 1)
        self.assertEqual(metadata.save_requests[0].request_id, "req1")
        self.assertEqual(metadata.save_requests[0].prefix_key, "scheduled")
        self.assertEqual(metadata.save_requests[0].block_ids, (4, 5))

    def test_request_finished_delays_free_for_saved_request(self) -> None:
        connector = self.make_connector()
        request = FakeRequest(
            {
                "turbobus.do_save": True,
                "turbobus.prefix_key": "saved",
                "turbobus.save_blocks": 3,
            }
        )
        connector.update_state_after_alloc(request, FakeBlocks(([1, 2, 3],)), 0)

        should_delay, params = connector.request_finished(request, [1, 2, 3])

        self.assertTrue(should_delay)
        self.assertEqual(params["turbobus.prefix_key"], "saved")
        self.assertEqual(params["turbobus.matched_tokens"], 48)

    def test_start_load_kv_is_safe_until_restore_enabled(self) -> None:
        connector = self.make_connector({"turbobus.restore_block_limit": 4})
        vllm_kv_connector._store_saved_prefix(
            TurboBusSavedPrefix("default", [object()], block_count=4, matched_tokens=64, session_id="session-a")
        )
        request = FakeRequest({"turbobus.do_restore": True, "turbobus.matched_tokens": 64})
        connector.update_state_after_alloc(request, FakeBlocks(([1, 2, 3, 4],)), 64)
        connector.bind_connector_metadata(connector.build_connector_meta(object()))

        connector.start_load_kv(forward_context=object())

        self.assertEqual(connector.state.events[-1]["event"], "load_ready")
        self.assertFalse(connector.state.events[-1]["restore_enabled"])
        self.assertEqual(connector.get_finished(set()), (None, {"req0"}))

    def test_wait_for_save_rejects_missing_layer_save(self) -> None:
        client = FakeClient(
            path_stats=[
                ({"kind": "direct", "bytes": 16, "chunk_count": 1},),
                ({"kind": "relay", "bytes": 16, "chunk_count": 2},),
            ],
            fallback_reasons=["", "direct_saturated"],
        )
        connector = self.make_connector(client=client)
        connector.state.kv_caches = {"0": FakeTensor(), "1": FakeTensor()}
        request = TurboBusRequestMetadata("req0", "saved", (1, 2), 32, 2)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector.bind_connector_metadata(metadata)

        with mock.patch.object(
            connector._backing_pool,
            "acquire",
            return_value=([object(), object()], False),
        ):
            with self.assertRaisesRegex(RuntimeError, "save_kv_layer"):
                connector.wait_for_save()

        self.assertEqual(client.submitted, [])

    def test_kv_connector_lifecycle_saves_and_restores_through_metadata(self) -> None:
        client = FakeClient(
            path_stats=[
                ({"kind": "direct", "bytes": 8, "chunk_count": 1},),
                ({"kind": "relay", "bytes": 8, "chunk_count": 1},),
                (
                    {"kind": "direct", "bytes": 4, "chunk_count": 1},
                    {"kind": "relay", "bytes": 4, "chunk_count": 1},
                ),
                ({"kind": "direct", "bytes": 8, "chunk_count": 1},),
            ],
            fallback_reasons=["", "direct_saturated", "relay_quota_exhausted", ""],
        )
        connector = self.make_connector({"turbobus.restore_enabled": True}, client=client)
        layer0 = FakeTensor()
        layer1 = FakeTensor()
        connector.register_kv_caches({"layer0": layer0, "layer1": layer1})
        save_request = FakeRequest(
            {
                "turbobus.do_save": True,
                "turbobus.prefix_key": "saved",
                "turbobus.save_blocks": 1,
                "turbobus.matched_tokens": 16,
            }
        )
        connector.update_state_after_alloc(save_request, FakeBlocks(([1],)), 0)
        connector.bind_connector_metadata(connector.build_connector_meta(SimpleNamespace()))

        with mock.patch.object(
            connector._backing_pool,
            "acquire",
            return_value=([object(), object()], False),
        ):
            connector.save_kv_layer("layer0", layer0, attn_metadata=object())
            connector.save_kv_layer("layer1", layer1, attn_metadata=object())
            connector.wait_for_save()

        saved = get_saved_prefix("saved", "session-a")
        self.assertIsNotNone(saved)
        self.assertEqual(saved.source_request_id, "req0")
        self.assertEqual(saved.receipt_ids, "receipt-1,receipt-2")
        self.assertEqual(saved.decision_ids, "decision-1,decision-2")
        self.assertEqual(saved.topology_snapshot_ids, "topology-1,topology-2")
        self.assertEqual(saved.ticket_ids, "ticket-1,ticket-2")
        self.assertEqual(saved.fallback_reason, "direct_saturated")
        self.assertEqual(saved.save_layer_count, 2)
        self.assertEqual(connector.get_finished({"req0"}), ({"req0"}, None))

        restore_request = FakeRequest(
            {
                "turbobus.do_restore": True,
                "turbobus.prefix_key": "saved",
                "turbobus.matched_tokens": 16,
            }
        )
        self.assertEqual(connector.get_num_new_matched_tokens(restore_request, 0), (16, True))
        connector.update_state_after_alloc(restore_request, FakeBlocks(([4],)), 16)
        connector.bind_connector_metadata(connector.build_connector_meta(SimpleNamespace()))
        connector.start_load_kv(forward_context=object())

        self.assertEqual(len(client.submitted), 4)
        save_intent = client.submitted[0]
        self.assertEqual(save_intent.job_id, "job-a")
        self.assertEqual(save_intent.session_id, "session-a")
        self.assertEqual(save_intent.source_buffer_id, "gpu-buffer")
        self.assertEqual(save_intent.destination_buffer_id, "cpu-buffer")
        self.assertEqual(save_intent.direction, "d2h")
        self.assertEqual(save_intent.workload_kind, WorkloadKind.KV_CACHE)
        self.assertEqual(save_intent.metadata["connector"], "vllm_kv")
        self.assertEqual(save_intent.metadata["operation"], "evict")
        self.assertEqual(save_intent.metadata["vllm_operation"], "save")
        self.assertEqual(save_intent.metadata["vllm_lifecycle"], "save_kv_layer")
        self.assertEqual(save_intent.metadata["prefix_key"], "saved")
        self.assertEqual(save_intent.metadata["block_ids"], [1])

        restore_intents = client.submitted[2:]
        self.assertTrue(all(intent.source_buffer_id == "cpu-buffer" for intent in restore_intents))
        self.assertTrue(all(intent.destination_buffer_id == "gpu-buffer" for intent in restore_intents))
        self.assertTrue(all(intent.direction == "h2d" for intent in restore_intents))
        self.assertTrue(all(intent.metadata["operation"] == "prefetch" for intent in restore_intents))
        self.assertTrue(all(intent.metadata["vllm_operation"] == "restore" for intent in restore_intents))
        self.assertTrue(all(intent.metadata["vllm_lifecycle"] == "start_load_kv" for intent in restore_intents))
        self.assertTrue(all(intent.metadata["prefix_key"] == "saved" for intent in restore_intents))
        self.assertTrue(all(intent.metadata["source_request_id"] == "req0" for intent in restore_intents))
        self.assertTrue(all(intent.metadata["block_ids"] == [4] for intent in restore_intents))

        event = connector.state.events[-1]
        self.assertEqual(event["event"], "restore")
        self.assertEqual(event["source_request_id"], "req0")
        self.assertEqual(event["bytes"], 16)
        self.assertEqual(event["direct_chunks"], 2)
        self.assertEqual(event["relay_chunks"], 1)
        self.assertEqual(event["direct_bytes"], 12)
        self.assertEqual(event["relay_bytes"], 4)
        self.assertEqual(event["receipt_ids"], "receipt-3,receipt-4")
        self.assertEqual(event["decision_ids"], "decision-3,decision-4")
        self.assertEqual(event["topology_snapshot_ids"], "topology-3,topology-4")
        self.assertEqual(event["ticket_ids"], "ticket-3,ticket-4")
        self.assertEqual(event["fallback_reason"], "relay_quota_exhausted")

    def test_save_kv_layer_aggregates_layer_receipts_before_registering_prefix(self) -> None:
        client = FakeClient(
            path_stats=[
                ({"kind": "direct", "bytes": 8, "chunk_count": 1},),
                ({"kind": "relay", "bytes": 8, "chunk_count": 1},),
            ]
        )
        connector = self.make_connector(client=client)
        layer0 = FakeTensor()
        layer1 = FakeTensor()
        connector.state.kv_caches = {"layer0": layer0, "layer1": layer1}
        request = TurboBusRequestMetadata("req0", "saved", (1,), 16, 1)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector.bind_connector_metadata(metadata)

        with mock.patch.object(
            connector._backing_pool,
            "acquire",
            return_value=([object(), object()], False),
        ):
            connector.save_kv_layer("layer0", layer0, attn_metadata=object())
            connector.save_kv_layer("layer1", layer1, attn_metadata=object())
            connector.wait_for_save()

        saved = get_saved_prefix("saved", "session-a")
        self.assertIsNotNone(saved)
        self.assertEqual(saved.source_request_id, "req0")
        self.assertEqual(saved.bytes, 16)
        self.assertEqual(saved.direct_chunks, 1)
        self.assertEqual(saved.relay_chunks, 1)
        self.assertEqual(saved.direct_bytes, 8)
        self.assertEqual(saved.relay_bytes, 8)
        self.assertEqual(saved.receipt_ids, "receipt-1,receipt-2")
        self.assertEqual(saved.save_layer_count, 2)
        self.assertEqual(saved.save_layer_ranges, 2)
        save_event = next(
            event for event in reversed(connector.state.events) if event["event"] == "save"
        )
        self.assertEqual(save_event["layers"], 2)
        self.assertEqual(save_event["ranges"], 2)
        self.assertEqual(connector.state.events[-1]["event"], "wait_for_save_done")

    def test_wait_for_save_rejects_incomplete_layer_save(self) -> None:
        connector = self.make_connector(client=FakeClient())
        layer0 = FakeTensor()
        layer1 = FakeTensor()
        connector.state.kv_caches = {"layer0": layer0, "layer1": layer1}
        request = TurboBusRequestMetadata("req0", "saved", (1,), 16, 1)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector.bind_connector_metadata(metadata)

        with mock.patch.object(
            connector._backing_pool,
            "acquire",
            return_value=([object(), object()], False),
        ):
            connector.save_kv_layer("layer0", layer0, attn_metadata=object())
            with self.assertRaises(RuntimeError):
                connector.wait_for_save()


if __name__ == "__main__":
    unittest.main()
