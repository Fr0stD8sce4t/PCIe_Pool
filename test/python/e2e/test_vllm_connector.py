from __future__ import annotations

from types import SimpleNamespace
import unittest

from turbobus.vllm_connector import VllmTurboBusConnector


class FakeTensor:
    shape = (1, 8, 4)

    def stride(self, dim: int) -> int:
        return (32, 4, 1)[dim]

    def element_size(self) -> int:
        return 2


class FakeHandle:
    def __init__(self, stats) -> None:
        self.stats = stats


class FakeAdapter:
    def __init__(self) -> None:
        self.handle = FakeHandle(
            SimpleNamespace(bytes=128, direct_chunks=2, relay_chunks=1)
        )

    def save_prefix(self, refs):
        self.refs = list(refs)
        return [self.handle, self.handle, FakeHandle(None)]


class FakeIntegration:
    def __init__(self) -> None:
        self.adapter = FakeAdapter()
        self.state = SimpleNamespace(kv_caches=[FakeTensor()])

    def block_ids_for_request(self, request_id: str):
        return (1, 2)

    def require_adapter(self):
        return self.adapter


class VllmTurboBusConnectorTest(unittest.TestCase):
    def test_save_event_deduplicates_transfer_handle_stats(self) -> None:
        connector = VllmTurboBusConnector(FakeIntegration())

        event = connector.save_request("req0", 2)

        self.assertEqual(event.request_id, "req0")
        self.assertEqual(event.operation, "save")
        self.assertEqual(event.block_count, 2)
        self.assertEqual(event.bytes, 128)
        self.assertEqual(event.direct_chunks, 2)
        self.assertEqual(event.relay_chunks, 1)


if __name__ == "__main__":
    unittest.main()
