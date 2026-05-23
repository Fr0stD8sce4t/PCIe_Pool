from __future__ import annotations

import unittest

from turbobus.schema import (
    BufferRegistration,
    DaemonResponse,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)
from turbobus.worker import (
    UnsupportedWorkerExecution,
    WorkerAuthorizationError,
    WorkerStatusReportError,
    WorkerTransferAuthorizer,
    WorkerTransferClient,
    WorkerTransferRequest,
    WorkerTransferResult,
    WorkerTransferState,
    WorkerTransferStatusReporter,
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


def authorization_request() -> WorkerTransferAuthorizationRequest:
    return WorkerTransferAuthorizationRequest(
        transfer_id="transfer-1",
        lease_id="lease-1",
        token="lease-token",
        session_id="session-1",
        job_id="job-1",
        src_buffer_id="cpu-buffer",
        dst_buffer_id="gpu-buffer",
        direction="h2d",
        ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
        relay_gpu=1,
    )


class FakeDaemonClient:
    def __init__(
        self,
        response: DaemonResponse,
        status_response: DaemonResponse | None = None,
    ) -> None:
        self.response = response
        self.status_response = status_response or DaemonResponse(ok=True)
        self.requests: list[WorkerTransferAuthorizationRequest] = []
        self.status_updates: list[dict[str, object]] = []

    def authorize_worker_transfer(
        self,
        request: WorkerTransferAuthorizationRequest,
    ) -> DaemonResponse:
        self.requests.append(request)
        return self.response

    def transfer_status(
        self,
        transfer_id: str,
        state: str | None = None,
        bytes_completed: int | None = None,
        error: str | None = None,
    ) -> DaemonResponse:
        self.status_updates.append(
            {
                "transfer_id": transfer_id,
                "state": state,
                "bytes_completed": bytes_completed,
                "error": error,
            }
        )
        return self.status_response


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

    def test_authorizer_builds_worker_request_from_daemon_response(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        authorizer = WorkerTransferAuthorizer(daemon_client)

        request = authorizer.authorize(authorization_request())

        self.assertEqual(request.transfer_id, "transfer-1")
        self.assertEqual(request.authorization.relay_gpu, 1)
        self.assertEqual(len(daemon_client.requests), 1)
        self.assertEqual(daemon_client.requests[0].lease_id, "lease-1")

    def test_authorizer_raises_on_daemon_denial(self) -> None:
        daemon_client = FakeDaemonClient(DaemonResponse(ok=False, error="denied"))
        authorizer = WorkerTransferAuthorizer(daemon_client)

        with self.assertRaisesRegex(WorkerAuthorizationError, "denied"):
            authorizer.authorize(authorization_request())

    def test_worker_client_submit_keeps_execution_unsupported(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        client = WorkerTransferClient(daemon_client)

        result = client.submit(authorization_request())

        self.assertEqual(result.transfer_id, "transfer-1")
        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(result.bytes_completed, 0)
        self.assertIn("not implemented", result.error)

    def test_status_reporter_maps_unsupported_to_daemon_failed_status(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        reporter = WorkerTransferStatusReporter(daemon_client)

        response = reporter.report(
            WorkerTransferResult(
                transfer_id="transfer-1",
                state=WorkerTransferState.UNSUPPORTED,
                error="worker execution is not implemented yet",
                bytes_completed=0,
            )
        )

        self.assertTrue(response.ok)
        self.assertEqual(len(daemon_client.status_updates), 1)
        self.assertEqual(daemon_client.status_updates[0]["transfer_id"], "transfer-1")
        self.assertEqual(daemon_client.status_updates[0]["state"], "failed")
        self.assertEqual(daemon_client.status_updates[0]["bytes_completed"], 0)
        self.assertIn("not implemented", daemon_client.status_updates[0]["error"])

    def test_status_reporter_maps_complete_to_daemon_complete_status(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        reporter = WorkerTransferStatusReporter(daemon_client)

        reporter.report(
            WorkerTransferResult(
                transfer_id="transfer-1",
                state=WorkerTransferState.COMPLETE,
                bytes_completed=64,
            )
        )

        self.assertEqual(daemon_client.status_updates[0]["state"], "complete")
        self.assertEqual(daemon_client.status_updates[0]["bytes_completed"], 64)
        self.assertIsNone(daemon_client.status_updates[0]["error"])

    def test_status_reporter_raises_on_daemon_rejection(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            status_response=DaemonResponse(ok=False, error="unknown transfer"),
        )
        reporter = WorkerTransferStatusReporter(daemon_client)

        with self.assertRaisesRegex(WorkerStatusReportError, "unknown transfer"):
            reporter.report(
                WorkerTransferResult(
                    transfer_id="transfer-1",
                    state=WorkerTransferState.FAILED,
                    error="copy failed",
                )
            )

    def test_worker_client_submit_and_report_updates_daemon_status(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        client = WorkerTransferClient(daemon_client)

        result = client.submit_and_report(authorization_request())

        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(len(daemon_client.status_updates), 1)
        self.assertEqual(daemon_client.status_updates[0]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
