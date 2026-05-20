from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from turbobus.runtime import RuntimeOptions, TransferHandle, TransferMode


class NativeHandle:
    def __init__(self, handle_id: int = 1) -> None:
        self.id = handle_id


class SuccessfulRuntime:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self, handle: TransferHandle) -> None:
        self.wait_calls += 1


class FailingRuntime:
    def wait(self, handle: TransferHandle) -> None:
        raise RuntimeError("simulated wait failure")


class TransferHandleTest(unittest.TestCase):
    def test_wait_marks_complete(self) -> None:
        runtime = SuccessfulRuntime()
        handle = TransferHandle(runtime, NativeHandle())

        self.assertEqual(handle.status, "submitted")
        self.assertFalse(handle.done)

        handle.wait()

        self.assertEqual(handle.status, "complete")
        self.assertTrue(handle.done)
        self.assertEqual(runtime.wait_calls, 1)

        handle.wait()
        self.assertEqual(runtime.wait_calls, 1)

    def test_wait_failure_marks_failed(self) -> None:
        handle = TransferHandle(FailingRuntime(), NativeHandle())

        with self.assertRaises(RuntimeError):
            handle.wait()

        self.assertEqual(handle.status, "failed")
        self.assertEqual(handle.error, "simulated wait failure")


class RuntimeOptionsTest(unittest.TestCase):
    def test_transfer_mode_accepts_string_values(self) -> None:
        options = RuntimeOptions(
            transfer_mode="direct",
            min_chunks_for_relay=3,
            relay_min_effective_bw_gbps=6.5,
            relay_min_direct_ratio=0.8,
        )

        self.assertEqual(options.transfer_mode, "direct")
        self.assertEqual(TransferMode(options.transfer_mode), TransferMode.DIRECT)
        self.assertEqual(options.min_chunks_for_relay, 3)
        self.assertEqual(options.relay_min_effective_bw_gbps, 6.5)
        self.assertEqual(options.relay_min_direct_ratio, 0.8)

    def test_from_tuning_json_reads_best_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tune.json"
            path.write_text(
                '{"best": {"chunk_bytes": 4194304, "staging_slots": 3}}',
                encoding="utf-8",
            )

            options = RuntimeOptions.from_tuning_json(path)

        self.assertEqual(options.chunk_bytes, 4194304)
        self.assertEqual(options.staging_slots, 3)

    def test_from_profile_json_reads_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            path.write_text(
                '{"config": {"chunk_bytes": 8388608, "profile_bytes": 16777216}}',
                encoding="utf-8",
            )

            options = RuntimeOptions.from_profile_json(path)

        self.assertEqual(options.chunk_bytes, 8388608)
        self.assertEqual(options.profile_bytes, 16777216)
        self.assertEqual(options.staging_slots, 2)


if __name__ == "__main__":
    unittest.main()
