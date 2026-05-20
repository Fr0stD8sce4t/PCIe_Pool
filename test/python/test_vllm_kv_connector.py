from __future__ import annotations

import unittest
from unittest import mock

from turbobus.vllm_kv_connector import (
    TurboBusConnector,
    TurboBusConnectorMetadata,
    clear_saved_prefixes,
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


if __name__ == "__main__":
    unittest.main()
