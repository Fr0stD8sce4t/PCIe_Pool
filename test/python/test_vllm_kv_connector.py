from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from turbobus.vllm_kv_connector import (
    TurboBusConnector,
    TurboBusConnectorMetadata,
    clear_saved_prefixes,
    get_saved_prefix,
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


class FakeAdapter:
    def restore_prefix(self, refs):
        return [FakeHandle()]

    def save_prefix(self, refs):
        return [FakeHandle()]


class TurboBusConnectorTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_saved_prefixes()

    def make_connector(self, extra=None):
        with mock.patch("turbobus.vllm_kv_connector._make_runtime_from_config", return_value=object()):
            return TurboBusConnector(FakeVllmConfig(extra), role="scheduler")

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

    def test_request_finished_all_groups_flattens_groups(self) -> None:
        connector = self.make_connector()

        self.assertEqual(
            connector.request_finished_all_groups(FakeRequest(), ([1, 2], [3])),
            (False, None),
        )

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
            mock.patch.object(connector, "_allocate_cpu_backings", return_value=[object(), object()]),
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
        self.assertEqual(connector.get_finished(set()), ({"req0"}, None))
        self.assertEqual(connector.get_finished(set()), (None, None))

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


if __name__ == "__main__":
    unittest.main()
