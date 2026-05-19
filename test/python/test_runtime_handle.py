from __future__ import annotations

import unittest

from turbobus.runtime import TransferHandle


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


if __name__ == "__main__":
    unittest.main()

