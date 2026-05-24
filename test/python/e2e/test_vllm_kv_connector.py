from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from turbobus.vllm_kv_connector import (
    TurboBusCPUBackingPool,
    TurboBusConnectorConfig,
    TurboBusConnector,
    TurboBusConnectorMetadata,
    TurboBusPrefixStore,
    TurboBusRequestMetadata,
    TurboBusSavedPrefix,
    clear_connector_events,
    clear_saved_prefixes,
    get_connector_events,
    get_saved_prefix,
    _make_runtime_from_config,
    register_saved_prefix,
)


class FakeCacheConfig:
    block_size = 16


class FakeTransferConfig:
    def __init__(self, extra=None):
        self.extra = extra or {}

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


class FakeStats:
    bytes = 64
    direct_chunks = 1
    relay_chunks = 0


class FakeHandle:
    stats = FakeStats()


class FakeTensor:
    shape = (1, 8, 4)

    def stride(self, dim):
        return (32, 4, 1)[dim]

    def element_size(self):
        return 2


class FakeAdapter:
    def restore_prefix(self, refs):
        return [FakeHandle()]

    def save_prefix(self, refs):
        return [FakeHandle()]


class TurboBusConnectorTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_connector_events()
        clear_saved_prefixes()

    def make_connector(self, extra=None):
        with mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()):
            return TurboBusConnector(
                FakeVllmConfig(extra),
                role="scheduler",
                kv_cache_config=object(),
            )

    def test_prefix_store_replaces_saved_prefix_by_key(self) -> None:
        store = TurboBusPrefixStore()
        first = TurboBusSavedPrefix("key", [object()], block_count=1, matched_tokens=16)
        self.assertEqual(store.put(first), [])
        previous = store.put(TurboBusSavedPrefix("key", [object()], block_count=2, matched_tokens=32))

        self.assertEqual(previous, [first])
        self.assertEqual(store.get("key").block_count, 2)

    def test_prefix_store_evicts_oldest_when_capacity_is_exceeded(self) -> None:
        store = TurboBusPrefixStore(max_prefixes=1)
        first = TurboBusSavedPrefix("a", [object()], block_count=1, matched_tokens=16)
        second = TurboBusSavedPrefix("b", [object()], block_count=1, matched_tokens=16)

        self.assertEqual(store.put(first), [])
        self.assertEqual(store.put(second), [first])
        self.assertIsNone(store.get("a"))
        self.assertIs(store.get("b"), second)

    def test_prefix_store_isolates_same_key_by_session(self) -> None:
        store = TurboBusPrefixStore()
        first = TurboBusSavedPrefix("key", [object()], block_count=1, matched_tokens=16, session_id="a")
        second = TurboBusSavedPrefix("key", [object()], block_count=2, matched_tokens=32, session_id="b")

        self.assertEqual(store.put(first), [])
        self.assertEqual(store.put(second), [])

        self.assertIs(store.get("key", "a"), first)
        self.assertIs(store.get("key", "b"), second)
        self.assertIsNone(store.get("key"))

    def test_prefix_store_can_clear_one_session(self) -> None:
        store = TurboBusPrefixStore()
        first = TurboBusSavedPrefix("key", [object()], block_count=1, matched_tokens=16, session_id="a")
        second = TurboBusSavedPrefix("key", [object()], block_count=2, matched_tokens=32, session_id="b")
        store.put(first)
        store.put(second)

        store.clear("a")

        self.assertIsNone(store.get("key", "a"))
        self.assertIs(store.get("key", "b"), second)

    def test_clear_saved_prefixes_can_clear_one_session(self) -> None:
        register_saved_prefix("key", [object()], block_count=1, matched_tokens=16, session_id="a")
        register_saved_prefix("key", [object()], block_count=2, matched_tokens=32, session_id="b")

        clear_saved_prefixes("a")

        self.assertIsNone(get_saved_prefix("key", "a"))
        self.assertEqual(get_saved_prefix("key", "b").block_count, 2)

    def test_connector_events_are_readable_and_clearable(self) -> None:
        register_saved_prefix("key", [object()], block_count=1, matched_tokens=16)

        events = get_connector_events()
        self.assertEqual(events[-1]["event"], "register_saved_prefix")
        self.assertEqual(events[-1]["prefix_key"], "key")

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

    def test_connector_config_reads_extra_config_before_environment(self) -> None:
        extra = {
            "turbobus.target_gpu": "6",
            "turbobus.relay_gpus": "5,7",
            "turbobus.chunk_bytes": "4194304",
            "turbobus.profile_bytes": "16777216",
            "turbobus.mode": "auto",
            "turbobus.min_pool_bytes": "12582912",
            "turbobus.daemon_socket_path": "/tmp/turbobusd.sock",
            "turbobus.daemon_max_inflight_chunks": "6",
            "turbobus.daemon_profile_max_age_seconds": "45",
            "turbobus.restore_block_limit": "8",
            "turbobus.restore_enabled": "true",
            "turbobus.session_id": "session-a",
            "turbobus.max_saved_prefixes": "2",
        }

        with mock.patch.dict(
            "os.environ",
            {
                "TURBOBUS_TARGET_GPU": "0",
                "TURBOBUS_RELAY_GPUS": "1",
                "TURBOBUS_MODE": "direct",
            },
        ):
            config = TurboBusConnectorConfig.from_vllm_config(FakeVllmConfig(extra))

        self.assertEqual(config.target_gpu, 6)
        self.assertEqual(config.relay_gpus, (5, 7))
        self.assertEqual(config.chunk_bytes, 4194304)
        self.assertEqual(config.profile_bytes, 16777216)
        self.assertEqual(config.mode, "auto")
        self.assertEqual(config.min_pool_bytes, 12582912)
        self.assertEqual(config.daemon_socket_path, "/tmp/turbobusd.sock")
        self.assertEqual(config.daemon_max_inflight_chunks, 6)
        self.assertEqual(config.daemon_profile_max_age_seconds, 45.0)
        self.assertEqual(config.restore_block_limit, 8)
        self.assertTrue(config.restore_enabled)
        self.assertEqual(config.session_id, "session-a")
        self.assertEqual(config.max_saved_prefixes, 2)

    def test_make_runtime_from_config_uses_connector_config(self) -> None:
        extra = {
            "turbobus.target_gpu": 6,
            "turbobus.relay_gpus": [5],
            "turbobus.chunk_bytes": 4194304,
            "turbobus.profile_bytes": 16777216,
            "turbobus.mode": "auto",
            "turbobus.min_pool_bytes": 12582912,
            "turbobus.daemon_socket_path": "/tmp/turbobusd.sock",
            "turbobus.daemon_max_inflight_chunks": 6,
            "turbobus.daemon_profile_max_age_seconds": 45.0,
        }

        with mock.patch("turbobus.vllm_kv_connector.Runtime") as runtime_class:
            _make_runtime_from_config(FakeVllmConfig(extra))

        _, kwargs = runtime_class.call_args
        self.assertEqual(kwargs["target_gpu"], 6)
        self.assertEqual(kwargs["relay_gpus"], [5])
        self.assertEqual(kwargs["options"].chunk_bytes, 4194304)
        self.assertEqual(kwargs["options"].profile_bytes, 16777216)
        self.assertEqual(kwargs["options"].transfer_mode, "auto")
        self.assertEqual(kwargs["options"].min_pool_bytes, 12582912)
        self.assertEqual(kwargs["options"].daemon_socket_path, "/tmp/turbobusd.sock")
        self.assertEqual(kwargs["options"].daemon_max_inflight_chunks, 6)
        self.assertEqual(kwargs["options"].daemon_profile_max_age_seconds, 45.0)

    def test_reports_explicit_external_match(self) -> None:
        register_saved_prefix("default", [object()], block_count=8, matched_tokens=96)
        connector = self.make_connector({"turbobus.restore_enabled": True})
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
        register_saved_prefix("default", [object()], block_count=8, matched_tokens=96)
        connector = self.make_connector()
        request = FakeRequest(
            {
                "turbobus.do_restore": True,
                "turbobus.matched_tokens": 96,
            }
        )

        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertEqual(connector.state.events[-1]["event"], "match_skipped")

    def test_ignores_requests_without_turbobus_restore(self) -> None:
        connector = self.make_connector()
        self.assertEqual(connector.get_num_new_matched_tokens(FakeRequest(), 0), (0, False))

    def test_restore_enabled_requires_registered_prefix(self) -> None:
        connector = self.make_connector({"turbobus.restore_enabled": True})
        request = FakeRequest(
            {
                "turbobus.do_restore": True,
                "turbobus.prefix_key": "missing",
                "turbobus.matched_tokens": 96,
            }
        )

        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertEqual(connector.state.events[-1]["event"], "match_miss")

    def test_records_allocated_blocks_for_connector_metadata(self) -> None:
        register_saved_prefix("default", [object()], block_count=4, matched_tokens=128)
        connector = self.make_connector({"turbobus.restore_block_limit": 4})
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

    def test_records_save_blocks_for_connector_metadata(self) -> None:
        connector = self.make_connector()
        request = FakeRequest(
            {
                "turbobus.do_save": True,
                "turbobus.prefix_key": "saved",
                "turbobus.save_blocks": 3,
                "turbobus.matched_tokens": 48,
            }
        )
        blocks = FakeBlocks(([1, 2, 3, 4],))

        connector.update_state_after_alloc(request, blocks, num_external_tokens=0)
        metadata = connector.build_connector_meta(scheduler_output=object())

        self.assertEqual(len(metadata), 1)
        self.assertEqual(len(metadata.requests), 0)
        self.assertEqual(len(metadata.save_requests), 1)
        self.assertEqual(metadata.save_requests[0].request_id, "req0")
        self.assertEqual(metadata.save_requests[0].prefix_key, "saved")
        self.assertEqual(metadata.save_requests[0].block_ids, (1, 2, 3))
        self.assertEqual(metadata.save_requests[0].matched_tokens, 48)
        self.assertEqual(connector.state.pending_saves, {})

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

    def test_build_connector_meta_remembers_save_params_until_blocks_arrive(self) -> None:
        connector = self.make_connector()
        first_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req1",
                    new_block_ids=[4],
                    kv_transfer_params={
                        "turbobus.do_save": True,
                        "turbobus.prefix_key": "scheduled",
                        "turbobus.save_blocks": 2,
                        "turbobus.matched_tokens": 32,
                    },
                )
            ]
        )

        first_metadata = connector.build_connector_meta(first_output)

        self.assertEqual(len(first_metadata.save_requests), 0)
        self.assertIn("req1", connector.state.save_params_by_request_id)

        second_output = SimpleNamespace(
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["req1"],
                new_block_ids=[[4, 5]],
                requests={},
            )
        )
        second_metadata = connector.build_connector_meta(second_output)

        self.assertEqual(len(second_metadata.save_requests), 1)
        self.assertEqual(second_metadata.save_requests[0].request_id, "req1")
        self.assertEqual(second_metadata.save_requests[0].prefix_key, "scheduled")
        self.assertEqual(second_metadata.save_requests[0].block_ids, (4, 5))
        self.assertNotIn("req1", connector.state.save_params_by_request_id)

    def test_update_state_after_alloc_remembers_save_params_until_blocks_arrive(self) -> None:
        connector = self.make_connector()
        request = FakeRequest(
            {
                "turbobus.do_save": True,
                "turbobus.prefix_key": "saved",
                "turbobus.save_blocks": 2,
                "turbobus.matched_tokens": 32,
            }
        )

        connector.update_state_after_alloc(request, FakeBlocks(([1],)), 0)
        metadata = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=["req0"],
                    new_block_ids=[[1, 2]],
                    requests={},
                )
            )
        )

        self.assertEqual(len(metadata.save_requests), 1)
        self.assertEqual(metadata.save_requests[0].prefix_key, "saved")
        self.assertEqual(metadata.save_requests[0].block_ids, (1, 2))

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

    def test_request_finished_does_not_delay_until_save_is_queued(self) -> None:
        connector = self.make_connector()
        request = FakeRequest(
            {
                "turbobus.do_save": True,
                "turbobus.prefix_key": "saved",
                "turbobus.save_blocks": 3,
            }
        )

        self.assertEqual(connector.request_finished(request, [1, 2]), (False, None))

    def test_request_finished_all_groups_delegates_to_save_finish(self) -> None:
        connector = self.make_connector()
        request = FakeRequest(
            {
                "turbobus.do_save": True,
                "turbobus.prefix_key": "saved",
                "turbobus.save_blocks": 3,
            }
        )
        connector.update_state_after_alloc(request, FakeBlocks(([1, 2, 3],)), 0)

        should_delay, params = connector.request_finished_all_groups(
            request,
            ([1], [2, 3]),
        )

        self.assertTrue(should_delay)
        self.assertEqual(params["turbobus.prefix_key"], "saved")
        self.assertEqual(params["turbobus.matched_tokens"], 48)

    def test_request_finished_all_groups_does_not_delay_until_save_is_queued(self) -> None:
        connector = self.make_connector()
        request = FakeRequest({"turbobus.do_save": True})

        self.assertEqual(
            connector.request_finished_all_groups(request, ([1], [2])),
            (False, None),
        )

    def test_empty_connector_metadata_does_not_add_events(self) -> None:
        connector = self.make_connector()
        metadata = connector.build_connector_meta(object())

        self.assertEqual(len(metadata), 0)
        self.assertEqual(connector.state.events, [])

    def test_start_load_kv_is_safe_until_restore_enabled(self) -> None:
        register_saved_prefix("default", [object()], block_count=4, matched_tokens=64)
        connector = self.make_connector({"turbobus.restore_block_limit": 4})
        request = FakeRequest({"turbobus.do_restore": True, "turbobus.matched_tokens": 64})
        connector.update_state_after_alloc(request, FakeBlocks(([1, 2, 3, 4],)), 64)
        connector._connector_metadata = connector.build_connector_meta(object())

        connector.start_load_kv(forward_context=object())

        self.assertEqual(connector.state.events[-1]["event"], "load_ready")
        self.assertFalse(connector.state.events[-1]["restore_enabled"])
        self.assertEqual(connector.get_finished(set()), (None, {"req0"}))
        self.assertEqual(connector.get_finished(set()), (None, None))

    def test_start_load_kv_reports_finished_after_restore(self) -> None:
        register_saved_prefix("default", [object()], block_count=4, matched_tokens=64)
        connector = self.make_connector(
            {
                "turbobus.restore_block_limit": 4,
                "turbobus.restore_enabled": True,
            }
        )
        request = FakeRequest({"turbobus.do_restore": True, "turbobus.matched_tokens": 64})
        connector.update_state_after_alloc(request, FakeBlocks(([1, 2, 3, 4],)), 64)
        connector._connector_metadata = connector.build_connector_meta(object())

        with mock.patch.object(connector, "_restore_request") as restore:
            connector.start_load_kv(forward_context=object())

        restore.assert_called_once()
        self.assertEqual(connector.get_finished(set()), (None, {"req0"}))
        self.assertEqual(connector.get_finished(set()), (None, None))

    def test_wait_for_save_registers_saved_prefix_and_reports_finished(self) -> None:
        connector = self.make_connector()
        connector.state.kv_caches = {
            "0": mock.Mock(),
            "1": mock.Mock(),
        }
        request = mock.Mock(
            request_id="req0",
            prefix_key="saved",
            block_ids=(1, 2),
            matched_tokens=32,
            block_count=2,
            cpu_slot_start=0,
        )
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector._connector_metadata = metadata

        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()),
            mock.patch.object(connector._backing_pool, "acquire", return_value=([object(), object()], False)),
            mock.patch("turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids", return_value=[object()]),
            mock.patch("turbobus.vllm.make_vllm_layer_groups_from_kv_caches", return_value=[object()]),
        ):
            with mock.patch("turbobus.vllm.VllmKVSlotAdapter", return_value=FakeAdapter()):
                connector.wait_for_save()

        saved = get_saved_prefix("saved")
        self.assertIsNotNone(saved)
        self.assertEqual(saved.source_request_id, "req0")
        self.assertEqual(saved.block_count, 2)
        self.assertEqual(saved.matched_tokens, 32)
        self.assertFalse(saved.reused_backing)
        self.assertEqual(connector.get_finished(set()), (None, None))
        self.assertEqual(connector.get_finished({"req0"}), ({"req0"}, None))
        self.assertEqual(connector.get_finished(set()), (None, None))

    def test_save_event_reports_auto_and_daemon_runtime_context(self) -> None:
        connector = self.make_connector()
        connector.state.kv_caches = {"0": mock.Mock()}
        request = TurboBusRequestMetadata("req0", "saved", (1,), 16, 1)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector._connector_metadata = metadata
        runtime = mock.Mock(
            last_auto_decision_dict=mock.Mock(
                return_value={
                    "auto_resolved_mode": "pool",
                    "auto_reason": "pool_speedup_1.500",
                }
            ),
            last_daemon_reservation_dict=mock.Mock(
                return_value={
                    "daemon_session_id": "daemon-session",
                    "daemon_reservation_status": "granted",
                    "daemon_reserved_relays": "5",
                }
            ),
        )

        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=runtime),
            mock.patch.object(connector._backing_pool, "acquire", return_value=([object()], False)),
            mock.patch("turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids", return_value=[object()]),
            mock.patch("turbobus.vllm.make_vllm_layer_groups_from_kv_caches", return_value=[object()]),
            mock.patch("turbobus.vllm.VllmKVSlotAdapter", return_value=FakeAdapter()),
        ):
            connector.wait_for_save()

        event = connector.state.events[-1]
        emitted = get_connector_events()[-1]
        self.assertEqual(event["event"], "save")
        self.assertEqual(event["auto_resolved_mode"], "pool")
        self.assertEqual(event["auto_reason"], "pool_speedup_1.500")
        self.assertEqual(event["daemon_session_id"], "daemon-session")
        self.assertEqual(event["daemon_reservation_status"], "granted")
        self.assertEqual(emitted["daemon_reserved_relays"], "5")

    def test_save_kv_layer_copies_layers_before_wait_for_save_registers_prefix(self) -> None:
        connector = self.make_connector()
        layer0 = FakeTensor()
        layer1 = FakeTensor()
        connector.state.kv_caches = {
            "layer0": layer0,
            "layer1": layer1,
        }
        request = TurboBusRequestMetadata("req0", "saved", (1, 2), 32, 2)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector._connector_metadata = metadata

        adapters = [FakeAdapter(), FakeAdapter()]
        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()),
            mock.patch.object(connector._backing_pool, "acquire", return_value=([object(), object()], False)),
            mock.patch("turbobus.vllm.VllmKVSlotAdapter", side_effect=adapters),
        ):
            connector.save_kv_layer("layer0", layer0, attn_metadata=object())
            connector.save_kv_layer("layer1", layer1, attn_metadata=object())
            connector.wait_for_save()

        saved = get_saved_prefix("saved")
        self.assertIsNotNone(saved)
        self.assertEqual(saved.source_request_id, "req0")
        self.assertEqual(saved.bytes, 128)
        self.assertEqual(saved.direct_chunks, 2)
        self.assertEqual(saved.relay_chunks, 0)
        self.assertEqual(saved.save_layer_count, 2)
        self.assertEqual(saved.save_layer_ranges, 2)
        self.assertEqual(connector.state.events[-1]["event"], "save")
        self.assertEqual(connector.state.events[-1]["layers"], 2)
        self.assertEqual(connector.state.events[-1]["ranges"], 2)
        self.assertEqual(connector.get_finished({"req0"}), ({"req0"}, None))

    def test_wait_for_save_rejects_incomplete_layer_save(self) -> None:
        connector = self.make_connector()
        layer0 = FakeTensor()
        layer1 = FakeTensor()
        connector.state.kv_caches = {
            "layer0": layer0,
            "layer1": layer1,
        }
        request = TurboBusRequestMetadata("req0", "saved", (1,), 16, 1)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector._connector_metadata = metadata

        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()),
            mock.patch.object(connector._backing_pool, "acquire", return_value=([object(), object()], False)),
            mock.patch("turbobus.vllm.VllmKVSlotAdapter", return_value=FakeAdapter()),
        ):
            connector.save_kv_layer("layer0", layer0, attn_metadata=object())
            with self.assertRaises(RuntimeError):
                connector.wait_for_save()

    def test_saved_prefix_is_registered_under_connector_session(self) -> None:
        connector = self.make_connector({"turbobus.session_id": "session-a"})
        connector.state.kv_caches = {
            "0": mock.Mock(),
            "1": mock.Mock(),
        }
        request = TurboBusRequestMetadata("req0", "saved", (1, 2), 32, 2)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(request)
        connector._connector_metadata = metadata

        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()),
            mock.patch.object(connector._backing_pool, "acquire", return_value=([object(), object()], False)),
            mock.patch("turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids", return_value=[object()]),
            mock.patch("turbobus.vllm.make_vllm_layer_groups_from_kv_caches", return_value=[object()]),
            mock.patch("turbobus.vllm.VllmKVSlotAdapter", return_value=FakeAdapter()),
        ):
            connector.wait_for_save()

        self.assertIsNone(get_saved_prefix("saved"))
        self.assertEqual(get_saved_prefix("saved", "session-a").source_request_id, "req0")

    def test_replacing_saved_prefix_releases_previous_backing(self) -> None:
        connector = self.make_connector()
        connector.state.kv_caches = {
            "0": mock.Mock(),
            "1": mock.Mock(),
        }
        first = TurboBusRequestMetadata("req0", "saved", (1, 2), 32, 2)
        second = TurboBusRequestMetadata("req1", "saved", (3, 4), 32, 2)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(first)
        metadata.add_save_request(second)
        connector._connector_metadata = metadata

        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()),
            mock.patch.object(
                connector._backing_pool,
                "acquire",
                side_effect=[
                    ([object(), object()], False),
                    ([object(), object()], True),
                ],
            ) as acquire,
            mock.patch.object(connector._backing_pool, "release_prefix") as release,
            mock.patch("turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids", return_value=[object()]),
            mock.patch("turbobus.vllm.make_vllm_layer_groups_from_kv_caches", return_value=[object()]),
            mock.patch("turbobus.vllm.VllmKVSlotAdapter", return_value=FakeAdapter()),
        ):
            connector.wait_for_save()

        self.assertEqual(acquire.call_count, 2)
        release.assert_called_once()
        saved = get_saved_prefix("saved")
        self.assertEqual(saved.source_request_id, "req1")
        self.assertTrue(saved.reused_backing)

    def test_prefix_capacity_evicts_old_prefix_and_releases_backing(self) -> None:
        connector = self.make_connector({"turbobus.max_saved_prefixes": 1})
        connector.state.kv_caches = {
            "0": mock.Mock(),
            "1": mock.Mock(),
        }
        first = TurboBusRequestMetadata("req0", "first", (1, 2), 32, 2)
        second = TurboBusRequestMetadata("req1", "second", (3, 4), 32, 2)
        metadata = TurboBusConnectorMetadata()
        metadata.add_save_request(first)
        metadata.add_save_request(second)
        connector._connector_metadata = metadata

        with (
            mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()),
            mock.patch.object(
                connector._backing_pool,
                "acquire",
                side_effect=[
                    ([object(), object()], False),
                    ([object(), object()], False),
                ],
            ),
            mock.patch.object(connector._backing_pool, "release_prefix") as release,
            mock.patch("turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids", return_value=[object()]),
            mock.patch("turbobus.vllm.make_vllm_layer_groups_from_kv_caches", return_value=[object()]),
            mock.patch("turbobus.vllm.VllmKVSlotAdapter", return_value=FakeAdapter()),
        ):
            connector.wait_for_save()

        release.assert_called_once()
        self.assertIsNone(get_saved_prefix("first"))
        self.assertEqual(get_saved_prefix("second").source_request_id, "req1")
        self.assertEqual(connector.state.events[-2]["event"], "evict_prefix")

    def test_restore_event_reports_timing_and_shape(self) -> None:
        connector = self.make_connector({"turbobus.restore_enabled": True})
        connector.state.kv_caches = {
            "0": mock.Mock(),
            "1": mock.Mock(),
        }
        request = mock.Mock(
            request_id="req0",
            prefix_key="default",
            block_ids=(1, 2),
            matched_tokens=32,
            cpu_slot_start=0,
        )
        register_saved_prefix("default", [object(), object()], block_count=2, matched_tokens=32)

        with (
            mock.patch.object(connector, "_adapter_for_saved_prefix", return_value=FakeAdapter()),
            mock.patch(
                "turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids",
                return_value=[object(), object(), object(), object()],
            ),
        ):
            connector._restore_request(request)

        event = connector.state.events[-1]
        self.assertEqual(event["event"], "restore")
        self.assertEqual(event["layers"], 2)
        self.assertEqual(event["ranges"], 4)
        self.assertIn("prepare_ms", event)
        self.assertIn("transfer_ms", event)
        self.assertIn("total_ms", event)

    def test_restore_event_prefers_adapter_transfer_stats(self) -> None:
        class StatsAdapter(FakeAdapter):
            def transfer_stats(self, refs):
                self.stats_refs = list(refs)
                return SimpleNamespace(bytes=96, direct_chunks=3, relay_chunks=2)

        adapter = StatsAdapter()
        connector = self.make_connector({"turbobus.restore_enabled": True})
        connector.runtime = mock.Mock(
            last_auto_decision_dict=mock.Mock(return_value={}),
            last_daemon_reservation_dict=mock.Mock(
                return_value={
                    "daemon_session_id": "daemon-session",
                    "daemon_reservation_status": "granted",
                }
            ),
        )
        connector.state.kv_caches = {"0": mock.Mock()}
        request = mock.Mock(
            request_id="req0",
            prefix_key="default",
            block_ids=(1,),
            matched_tokens=16,
            cpu_slot_start=0,
        )
        register_saved_prefix("default", [object()], block_count=1, matched_tokens=16)

        with (
            mock.patch.object(connector, "_adapter_for_saved_prefix", return_value=adapter),
            mock.patch(
                "turbobus.vllm_kv_connector.make_vllm_layer_range_refs_from_ids",
                return_value=["ref0"],
            ),
        ):
            connector._restore_request(request)

        self.assertEqual(adapter.stats_refs, ["ref0"])
        self.assertEqual(connector.state.events[-1]["bytes"], 96)
        self.assertEqual(connector.state.events[-1]["direct_chunks"], 3)
        self.assertEqual(connector.state.events[-1]["relay_chunks"], 2)
        self.assertEqual(connector.state.events[-1]["daemon_session_id"], "daemon-session")
        self.assertEqual(connector.state.events[-1]["daemon_reservation_status"], "granted")


if __name__ == "__main__":
    unittest.main()
