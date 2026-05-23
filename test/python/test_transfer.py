from __future__ import annotations

import json
import unittest

from turbobus import TransferDirection, TransferRange, TransferRequest
from turbobus.schema import TransferMode


class TransferRequestTest(unittest.TestCase):
    def test_contiguous_request_normalizes_values(self) -> None:
        request = TransferRequest(
            total_bytes=33,
            chunk_bytes=16,
            direction="h2d",
            mode="pool",
        )

        self.assertEqual(request.direction, TransferDirection.H2D)
        self.assertEqual(request.mode, TransferMode.POOL)
        self.assertEqual(request.request_chunks, 3)
        self.assertEqual(
            request.daemon_payload(),
            {
                "total_bytes": 33,
                "chunk_bytes": 16,
                "mode": "pool",
                "direction": "h2d",
                "request_chunks": 3,
            },
        )

    def test_range_request_computes_total_bytes_and_chunks(self) -> None:
        request = TransferRequest.from_ranges(
            [
                {"src_offset": 0, "dst_offset": 16, "bytes": 8},
                (8, 24, 24),
            ],
            chunk_bytes=16,
            direction=TransferDirection.D2H,
            mode=TransferMode.RELAY,
            job_id="job-1",
        )

        self.assertEqual(request.total_bytes, 32)
        self.assertEqual(request.request_chunks, 3)
        self.assertEqual(
            [item.as_dict() for item in request.ranges],
            [
                {"src_offset": 0, "dst_offset": 16, "bytes": 8},
                {"src_offset": 8, "dst_offset": 24, "bytes": 24},
            ],
        )
        payload = request.daemon_payload()
        self.assertEqual(payload["direction"], "d2h")
        self.assertEqual(payload["mode"], "relay")
        self.assertEqual(payload["job_id"], "job-1")
        self.assertEqual(len(payload["ranges"]), 2)
        json.dumps(payload)

    def test_invalid_request_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TransferRequest(total_bytes=-1, chunk_bytes=16, direction="h2d")
        with self.assertRaises(ValueError):
            TransferRequest(total_bytes=1, chunk_bytes=0, direction="h2d")
        with self.assertRaises(ValueError):
            TransferRequest(total_bytes=1, chunk_bytes=16, direction="sideways")
        with self.assertRaises(ValueError):
            TransferRange(src_offset=0, dst_offset=0, bytes=0)
        with self.assertRaises(ValueError):
            TransferRequest(
                total_bytes=32,
                chunk_bytes=16,
                direction="h2d",
                ranges=(TransferRange(src_offset=0, dst_offset=0, bytes=16),),
            )
        with self.assertRaises(ValueError):
            TransferRequest(
                total_bytes=32,
                chunk_bytes=16,
                direction="h2d",
                request_chunks=1,
                ranges=(TransferRange(src_offset=0, dst_offset=0, bytes=32),),
            )

    def test_with_mode_preserves_shape(self) -> None:
        request = TransferRequest.from_ranges(
            [TransferRange(src_offset=0, dst_offset=0, bytes=16)],
            chunk_bytes=8,
            direction="h2d",
            mode="auto",
            metadata={"adapter": "vllm"},
        )

        resolved = request.with_mode("direct")

        self.assertEqual(resolved.mode, TransferMode.DIRECT)
        self.assertEqual(resolved.total_bytes, request.total_bytes)
        self.assertEqual(resolved.request_chunks, request.request_chunks)
        self.assertEqual(resolved.ranges, request.ranges)
        self.assertEqual(resolved.as_dict()["metadata"], {"adapter": "vllm"})


if __name__ == "__main__":
    unittest.main()
