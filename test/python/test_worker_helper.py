from __future__ import annotations

import unittest

from turbobus.schema import BufferRegistration, WorkerTransferAuthorization
from turbobus.worker import (
    UnsupportedWorkerExecution,
    WorkerTransferRequest,
    WorkerTransferState,
    WorkerTransferUnsupportedExecutor,
)


def authorization_payload() -> dict:
    authorization = WorkerTransferAuthorization(
        transfer_id="transfer-1",
        lease_id="lease-1",
        session_id="session-1",
        job_id="job-1",
        src_buffer=BufferRegistration(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=64,
            pinned=True,
        ),
        dst_buffer=BufferRegistration(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=64,
            device_index=0,
        ),
        direction="h2d",
        ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
        relay_gpu=1,
    )
    return WorkerTransferRequest(authorization=authorization).as_dict()


class WorkerHelperTest(unittest.TestCase):
    def test_worker_request_parses_daemon_authorization_payload(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())

        self.assertEqual(request.transfer_id, "transfer-1")
        self.assertEqual(request.authorization.src_buffer.buffer_id, "cpu-buffer")
        self.assertEqual(request.authorization.dst_buffer.buffer_id, "gpu-buffer")
        self.assertEqual(request.authorization.ranges[0]["bytes"], 16)

    def test_unsupported_executor_reports_no_data_movement(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        executor = WorkerTransferUnsupportedExecutor()

        result = executor.execute(request)

        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(result.bytes_completed, 0)
        self.assertIn("not implemented", result.error)
        self.assertEqual(result.metadata["relay_gpu"], 1)
        self.assertEqual(result.metadata["src_buffer_id"], "cpu-buffer")

    def test_unsupported_executor_can_raise_explicit_error(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        executor = WorkerTransferUnsupportedExecutor()

        with self.assertRaises(UnsupportedWorkerExecution):
            executor.execute_or_raise(request)


if __name__ == "__main__":
    unittest.main()
