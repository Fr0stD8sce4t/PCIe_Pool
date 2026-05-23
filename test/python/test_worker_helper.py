from __future__ import annotations

import unittest

from turbobus.schema import (
    BufferRegistration,
    DaemonResponse,
    WorkerDataPlaneRequest,
    WorkerTransferAuthorization,
    WorkerTransferAuthorizationRequest,
)
from turbobus.daemon.server import TurboBusDaemon
from turbobus.daemon.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
    StaticTopologyProvider,
)
from turbobus.worker import (
    UnsupportedWorkerExecution,
    WorkerAuthorizationError,
    WorkerCleanupError,
    WorkerDataPlaneCompletionEnvelope,
    WorkerServiceRequestEnvelope,
    WorkerServiceResponseEnvelope,
    WorkerStatusReportError,
    WorkerStagingPool,
    WorkerStagingSlot,
    WorkerTransferAuthorizer,
    WorkerTransferClient,
    WorkerTransferCleanupCoordinator,
    WorkerTransferLifecycleRecord,
    WorkerTransferRequest,
    WorkerTransferResult,
    WorkerTransferService,
    WorkerTransferState,
    WorkerTransferStatusReporter,
    WorkerTransferUnsupportedExecutor,
    parse_worker_authorization_request_payload,
    run_worker_service_control_plane_smoke,
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


def authorization_request_payload() -> dict:
    return {
        "transfer_id": "transfer-1",
        "lease_id": "lease-1",
        "token": "lease-token",
        "session_id": "session-1",
        "job_id": "job-1",
        "src_buffer_id": "cpu-buffer",
        "dst_buffer_id": "gpu-buffer",
        "direction": "h2d",
        "ranges": [{"src_offset": 0, "dst_offset": 0, "bytes": 16}],
        "relay_gpu": 1,
    }


def daemon_with_relay_transfer_path() -> tuple[TurboBusDaemon, str]:
    daemon = TurboBusDaemon(
        relay_gpus=[1],
        max_sessions_per_relay=1,
        max_inflight_chunks_per_relay=8,
        topology_provider=StaticTopologyProvider(
            DaemonResourceInventory(
                gpus=(
                    GpuInventoryRecord(device_id=0, role="target"),
                    GpuInventoryRecord(device_id=1, role="relay"),
                ),
                pcie_paths=(PciePathRecord(device_id=1),),
                fabric_links=(
                    FabricLinkRecord(
                        src_device_id=1,
                        dst_device_id=0,
                        fabric="nvlink",
                        enabled=True,
                    ),
                ),
                source="test",
            )
        ),
    )
    registered = daemon.register_session(
        target_gpu=0,
        requested_relays=[1],
        max_inflight_chunks=8,
    )
    session_id = registered.payload["session"]["session_id"]
    daemon.register_job(job_id="job-1", session_id=session_id)
    daemon.register_buffer(
        buffer_id="cpu-buffer",
        job_id="job-1",
        kind="cpu_pinned",
        size_bytes=64,
        pinned=True,
    )
    daemon.register_buffer(
        buffer_id="gpu-buffer",
        job_id="job-1",
        kind="gpu",
        size_bytes=64,
        device_index=0,
    )
    daemon.put_profile(
        target_gpu=0,
        relay_gpus=[1],
        profile={
            "target_device": 0,
            "direct_h2d_bw_gbps": 7.5,
            "direct_d2h_bw_gbps": 6.5,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 7.5,
                    "d2h_bw_gbps": 6.5,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 7.5,
                    "effective_d2h_bw_gbps": 6.5,
                    "p2p_enabled": True,
                }
            ],
        },
    )
    return daemon, session_id


class FakeDaemonClient:
    def __init__(
        self,
        response: DaemonResponse,
        status_response: DaemonResponse | None = None,
        cleanup_response: DaemonResponse | None = None,
    ) -> None:
        self.response = response
        self.status_response = status_response or DaemonResponse(ok=True)
        self.cleanup_response = cleanup_response or DaemonResponse(ok=True)
        self.requests: list[WorkerTransferAuthorizationRequest] = []
        self.status_updates: list[dict[str, object]] = []
        self.cleanup_requests: list[dict[str, object]] = []

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

    def cleanup(
        self,
        target_kind: str,
        target_id: str,
        reason: str = "manual",
        force: bool = False,
    ) -> DaemonResponse:
        self.cleanup_requests.append(
            {
                "target_kind": target_kind,
                "target_id": target_id,
                "reason": reason,
                "force": force,
            }
        )
        return self.cleanup_response


class WorkerHelperTest(unittest.TestCase):
    def test_worker_request_parses_daemon_authorization_payload(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())

        self.assertEqual(request.transfer_id, "transfer-1")
        self.assertEqual(request.authorization.src_buffer.buffer_id, "cpu-buffer")
        self.assertEqual(request.authorization.dst_buffer.buffer_id, "gpu-buffer")
        self.assertEqual(request.authorization.ranges[0]["bytes"], 16)

    def test_worker_request_builds_data_plane_request_from_authorization(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())

        self.assertIsInstance(request.data_plane, WorkerDataPlaneRequest)
        self.assertEqual(request.data_plane.transfer_id, "transfer-1")
        self.assertEqual(request.data_plane.lease_id, "lease-1")
        self.assertEqual(request.data_plane.relay_gpu, 1)
        self.assertEqual(request.data_plane.direction, "h2d")
        self.assertEqual(request.data_plane.src_handle.buffer_id, "cpu-buffer")
        self.assertEqual(request.data_plane.src_handle.access, "read")
        self.assertEqual(request.data_plane.dst_handle.buffer_id, "gpu-buffer")
        self.assertEqual(request.data_plane.dst_handle.access, "write")
        self.assertEqual(request.data_plane.staging.relay_gpu, 1)
        self.assertEqual(request.data_plane.staging.total_bytes, 16)
        self.assertEqual(request.as_dict()["data_plane"]["staging"]["chunk_count"], 1)

    def test_worker_request_rejects_mismatched_data_plane_authority(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        bad_data_plane = WorkerDataPlaneRequest(
            transfer_id="other-transfer",
            lease_id=request.data_plane.lease_id,
            session_id=request.data_plane.session_id,
            job_id=request.data_plane.job_id,
            relay_gpu=request.data_plane.relay_gpu,
            direction=request.data_plane.direction,
            src_handle=request.data_plane.src_handle,
            dst_handle=request.data_plane.dst_handle,
            staging=request.data_plane.staging,
            ranges=request.data_plane.ranges,
        )

        with self.assertRaisesRegex(ValueError, "transfer id"):
            WorkerTransferRequest(
                authorization=request.authorization,
                data_plane=bad_data_plane,
            )

    def test_worker_request_rejects_mismatched_data_plane_handles(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        bad_data_plane = WorkerDataPlaneRequest(
            transfer_id=request.data_plane.transfer_id,
            lease_id=request.data_plane.lease_id,
            session_id=request.data_plane.session_id,
            job_id=request.data_plane.job_id,
            relay_gpu=request.data_plane.relay_gpu,
            direction=request.data_plane.direction,
            src_handle=request.data_plane.dst_handle,
            dst_handle=request.data_plane.dst_handle,
            staging=request.data_plane.staging,
            ranges=request.data_plane.ranges,
        )

        with self.assertRaisesRegex(ValueError, "src handle"):
            WorkerTransferRequest(
                authorization=request.authorization,
                data_plane=bad_data_plane,
            )

    def test_unsupported_executor_reports_no_data_movement(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        executor = WorkerTransferUnsupportedExecutor()
        staging_pool = WorkerStagingPool(slot_id_factory=lambda: "staging-1")
        staging_slot = staging_pool.allocate(request.data_plane)

        result = executor.execute(request, staging_slot)

        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(result.bytes_completed, 0)
        self.assertIn("not implemented", result.error)
        self.assertEqual(result.metadata["relay_gpu"], 1)
        self.assertEqual(result.metadata["src_buffer_id"], "cpu-buffer")
        self.assertEqual(result.metadata["staging_slot_id"], "staging-1")
        self.assertEqual(result.metadata["staging_allocated_bytes"], 256)

    def test_worker_result_builds_data_plane_completion_report(self) -> None:
        result = WorkerTransferResult(
            transfer_id="transfer-1",
            state=WorkerTransferState.UNSUPPORTED,
            error="worker execution is not implemented yet",
            bytes_completed=0,
        )

        completion = result.data_plane_completion("lease-1")

        self.assertEqual(completion.transfer_id, "transfer-1")
        self.assertEqual(completion.lease_id, "lease-1")
        self.assertEqual(completion.state.value, "failed")
        self.assertEqual(completion.bytes_completed, 0)
        self.assertIn("not implemented", completion.error)

    def test_unsupported_executor_can_raise_explicit_error(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        executor = WorkerTransferUnsupportedExecutor()
        staging_slot = WorkerStagingPool().allocate(request.data_plane)

        with self.assertRaises(UnsupportedWorkerExecution):
            executor.execute_or_raise(request, staging_slot)

    def test_unsupported_executor_rejects_mismatched_staging_slot(self) -> None:
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())
        other_request = WorkerTransferRequest.from_authorization_payload(
            {
                "authorization": {
                    **authorization_payload()["authorization"],
                    "transfer_id": "transfer-2",
                }
            }
        )
        staging_slot = WorkerStagingPool().allocate(other_request.data_plane)
        executor = WorkerTransferUnsupportedExecutor()

        with self.assertRaisesRegex(ValueError, "transfer"):
            executor.execute(request, staging_slot)

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
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        result = client.submit(authorization_request())

        self.assertEqual(result.transfer_id, "transfer-1")
        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(result.bytes_completed, 0)
        self.assertIn("not implemented", result.error)
        self.assertEqual(result.metadata["staging_slot_id"], "staging-1")
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

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

    def test_cleanup_coordinator_cleans_authorization_failure_reservation(self) -> None:
        daemon_client = FakeDaemonClient(DaemonResponse(ok=False, error="denied"))
        coordinator = WorkerTransferCleanupCoordinator(daemon_client)

        response = coordinator.cleanup_authorization_failure(authorization_request())

        self.assertTrue(response.ok)
        self.assertEqual(
            daemon_client.cleanup_requests,
            [
                {
                    "target_kind": "reservation",
                    "target_id": "lease-1",
                    "reason": "worker_authorization_failed",
                    "force": True,
                }
            ],
        )

    def test_cleanup_coordinator_cleans_failed_worker_session(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        coordinator = WorkerTransferCleanupCoordinator(daemon_client)
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())

        coordinator.cleanup_execution_failure(
            request,
            WorkerTransferResult(
                transfer_id="transfer-1",
                state=WorkerTransferState.FAILED,
                error="copy failed",
            ),
            target_kind="session",
        )

        self.assertEqual(
            daemon_client.cleanup_requests,
            [
                {
                    "target_kind": "session",
                    "target_id": "session-1",
                    "reason": "worker_failed",
                    "force": True,
                }
            ],
        )

    def test_cleanup_coordinator_skips_complete_transfer(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        coordinator = WorkerTransferCleanupCoordinator(daemon_client)
        request = WorkerTransferRequest.from_authorization_payload(authorization_payload())

        response = coordinator.cleanup_execution_failure(
            request,
            WorkerTransferResult(
                transfer_id="transfer-1",
                state=WorkerTransferState.COMPLETE,
                bytes_completed=64,
            ),
        )

        self.assertTrue(response.ok)
        self.assertTrue(response.payload["cleanup_skipped"])
        self.assertEqual(daemon_client.cleanup_requests, [])

    def test_cleanup_coordinator_raises_on_daemon_rejection(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            cleanup_response=DaemonResponse(ok=False, error="unknown reservation"),
        )
        coordinator = WorkerTransferCleanupCoordinator(daemon_client)

        with self.assertRaisesRegex(WorkerCleanupError, "unknown reservation"):
            coordinator.cleanup_authorization_failure(authorization_request())

    def test_worker_client_submit_report_and_cleanup_failed_result(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        client = WorkerTransferClient(daemon_client)

        result = client.submit_report_and_cleanup(authorization_request())

        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(daemon_client.status_updates[0]["state"], "failed")
        self.assertEqual(
            daemon_client.cleanup_requests,
            [
                {
                    "target_kind": "reservation",
                    "target_id": "lease-1",
                    "reason": "worker_unsupported",
                    "force": True,
                }
            ],
        )

    def test_worker_client_cleans_reservation_after_authorization_failure(self) -> None:
        daemon_client = FakeDaemonClient(DaemonResponse(ok=False, error="denied"))
        client = WorkerTransferClient(daemon_client)

        with self.assertRaisesRegex(WorkerAuthorizationError, "denied"):
            client.submit_report_and_cleanup(authorization_request())

        self.assertEqual(daemon_client.status_updates, [])
        self.assertEqual(
            daemon_client.cleanup_requests,
            [
                {
                    "target_kind": "reservation",
                    "target_id": "lease-1",
                    "reason": "worker_authorization_failed",
                    "force": True,
                }
            ],
        )

    def test_worker_client_cleanup_releases_daemon_reservation(self) -> None:
        daemon, session_id = daemon_with_relay_transfer_path()
        planned = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=64,
            chunk_bytes=16,
            mode="pool",
            direction="h2d",
            job_id="job-1",
            buffer_ids=["cpu-buffer", "gpu-buffer"],
        )
        transfer_id = planned.payload["transfer_id"]
        lease_token = planned.payload["lease_tokens"][0]
        client = WorkerTransferClient(daemon)

        result = client.submit_report_and_cleanup(
            WorkerTransferAuthorizationRequest(
                transfer_id=transfer_id,
                lease_id=lease_token["lease_id"],
                token=lease_token["token"],
                session_id=session_id,
                job_id="job-1",
                src_buffer_id="cpu-buffer",
                dst_buffer_id="gpu-buffer",
                direction="h2d",
                relay_gpu=1,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
            )
        )

        self.assertEqual(result.state, WorkerTransferState.UNSUPPORTED)
        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertIn(
            {
                "target_kind": "reservation",
                "target_id": lease_token["lease_id"],
                "reason": "worker_unsupported",
                "force": True,
            },
            profile["cleanup_events"],
        )
        status = daemon.transfer_status(transfer_id)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "failed")

    def test_worker_service_smoke_reclaims_daemon_reservation(self) -> None:
        daemon, session_id = daemon_with_relay_transfer_path()

        smoke = run_worker_service_control_plane_smoke(
            daemon,
            session_id=session_id,
            job_id="job-1",
            src_buffer_id="cpu-buffer",
            dst_buffer_id="gpu-buffer",
            total_bytes=64,
            chunk_bytes=16,
            direction="h2d",
            mode="pool",
            relay_gpu=1,
        )

        service_response = smoke["service_response"]
        self.assertTrue(service_response["ok"])
        self.assertEqual(service_response["final_state"], "unsupported")
        self.assertEqual(
            service_response["lifecycle"]["cleanup_target"]["target_id"],
            smoke["lease_id"],
        )
        describe = smoke["daemon_describe"]["payload"]
        self.assertEqual(describe["reservations"], {})
        self.assertIn(
            {
                "target_kind": "reservation",
                "target_id": smoke["lease_id"],
                "reason": "worker_unsupported",
                "force": True,
            },
            describe["cleanup_events"],
        )

    def test_worker_service_smoke_reports_unsupported_execution_failed(self) -> None:
        daemon, session_id = daemon_with_relay_transfer_path()

        smoke = run_worker_service_control_plane_smoke(
            daemon,
            session_id=session_id,
            job_id="job-1",
            src_buffer_id="cpu-buffer",
            dst_buffer_id="gpu-buffer",
            total_bytes=64,
            chunk_bytes=16,
            direction="h2d",
            mode="pool",
            relay_gpu=1,
            ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
        )

        lifecycle = smoke["service_response"]["lifecycle"]
        self.assertEqual(lifecycle["result"]["state"], "unsupported")
        self.assertEqual(lifecycle["status_update"]["state"], "failed")
        self.assertIn("not implemented", lifecycle["status_update"]["error"])
        self.assertEqual(smoke["daemon_status"]["payload"]["status"]["state"], "failed")
        self.assertEqual(
            smoke["daemon_status"]["payload"]["status"]["bytes_completed"],
            0,
        )

    def test_lifecycle_record_serializes_control_plane_state(self) -> None:
        request = authorization_request()
        worker_request = WorkerTransferRequest.from_authorization_payload(
            authorization_payload()
        )
        result = WorkerTransferResult(
            transfer_id="transfer-1",
            state=WorkerTransferState.UNSUPPORTED,
            error="worker execution is not implemented yet",
        )
        staging_slot = WorkerStagingPool(
            slot_id_factory=lambda: "staging-1",
        ).allocate(worker_request.data_plane)

        record = WorkerTransferLifecycleRecord(
            authorization_request=request,
            worker_request=worker_request,
            staging_slot=staging_slot,
            result=result,
            status_update={
                "transfer_id": "transfer-1",
                "state": "failed",
                "bytes_completed": 0,
                "error": result.error,
            },
            status_response=DaemonResponse(ok=True, payload={"status": {"state": "failed"}}),
            cleanup_target_kind="reservation",
            cleanup_target_id="lease-1",
            cleanup_response=DaemonResponse(ok=True, payload={"removed": {"reservations": 1}}),
            final_state="unsupported",
            error=result.error,
        )
        payload = record.as_dict()

        self.assertEqual(payload["authorization_request"]["lease_id"], "lease-1")
        self.assertEqual(
            payload["worker_request"]["authorization"]["src_buffer"]["buffer_id"],
            "cpu-buffer",
        )
        self.assertEqual(payload["staging_slot"]["transfer_id"], "transfer-1")
        self.assertIsNone(payload["staging_release"])
        self.assertEqual(payload["result"]["state"], "unsupported")
        self.assertEqual(payload["status_update"]["state"], "failed")
        self.assertEqual(payload["status_response"]["payload"]["status"]["state"], "failed")
        self.assertEqual(payload["cleanup_target"]["target_id"], "lease-1")
        self.assertEqual(payload["cleanup_response"]["payload"]["removed"]["reservations"], 1)
        self.assertEqual(payload["final_state"], "unsupported")

    def test_worker_client_lifecycle_records_status_and_cleanup(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "unsupported")
        self.assertEqual(lifecycle.result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(lifecycle.status_response, daemon_client.status_response)
        self.assertEqual(lifecycle.cleanup_response, daemon_client.cleanup_response)
        self.assertEqual(lifecycle.cleanup_target_kind, "reservation")
        self.assertEqual(lifecycle.cleanup_target_id, "lease-1")
        self.assertEqual(lifecycle.status_update["state"], "failed")
        self.assertIn("not implemented", lifecycle.status_update["error"])
        self.assertEqual(lifecycle.staging_slot.transfer_id, "transfer-1")
        self.assertTrue(lifecycle.staging_slot.active)
        self.assertEqual(lifecycle.staging_release.slot_id, lifecycle.staging_slot.slot_id)
        self.assertFalse(lifecycle.staging_release.active)
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})
        self.assertEqual(daemon_client.status_updates[0]["state"], "failed")
        self.assertEqual(daemon_client.cleanup_requests[0]["target_id"], "lease-1")

    def test_worker_data_plane_completion_envelope_serializes_success_lifecycle(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())
        envelope = WorkerDataPlaneCompletionEnvelope.from_lifecycle(lifecycle)
        payload = envelope.as_dict()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["transfer_id"], "transfer-1")
        self.assertEqual(payload["lease_id"], "lease-1")
        self.assertEqual(payload["final_state"], "unsupported")
        self.assertEqual(payload["staging_slot"]["slot_id"], "staging-1")
        self.assertTrue(payload["staging_slot"]["active"])
        self.assertEqual(payload["worker_result"]["state"], "unsupported")
        self.assertEqual(payload["daemon_status_update"]["state"], "failed")
        self.assertTrue(payload["daemon_status_response"]["ok"])
        self.assertTrue(payload["daemon_cleanup_response"]["ok"])
        self.assertEqual(payload["staging_release"]["slot_id"], "staging-1")
        self.assertFalse(payload["staging_release"]["active"])
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_client_lifecycle_passes_staging_slot_to_executor(self) -> None:
        class RecordingExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[WorkerTransferRequest, WorkerStagingSlot]] = []

            def execute(
                self,
                request: WorkerTransferRequest,
                staging_slot: WorkerStagingSlot,
            ) -> WorkerTransferResult:
                self.calls.append((request, staging_slot))
                return WorkerTransferResult(
                    transfer_id=request.transfer_id,
                    state=WorkerTransferState.UNSUPPORTED,
                    error="recorded unsupported",
                    metadata={"staging_slot_id": staging_slot.slot_id},
                )

        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        staging_pool = WorkerStagingPool()
        executor = RecordingExecutor()
        client = WorkerTransferClient(
            daemon_client,
            executor=executor,
            staging_pool=staging_pool,
        )

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(len(executor.calls), 1)
        recorded_request, recorded_slot = executor.calls[0]
        self.assertEqual(recorded_request.transfer_id, "transfer-1")
        self.assertEqual(recorded_slot.slot_id, lifecycle.staging_slot.slot_id)
        self.assertEqual(lifecycle.result.metadata["staging_slot_id"], recorded_slot.slot_id)
        self.assertFalse(lifecycle.staging_release.active)
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_client_lifecycle_authorization_failure_does_not_allocate_staging(self) -> None:
        daemon_client = FakeDaemonClient(DaemonResponse(ok=False, error="denied"))
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "authorization_failed")
        self.assertIsNone(lifecycle.worker_request)
        self.assertIsNone(lifecycle.staging_slot)
        self.assertIsNone(lifecycle.staging_release)
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_client_lifecycle_status_failure_releases_staging(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            status_response=DaemonResponse(ok=False, error="unknown transfer"),
        )
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "status_failed")
        self.assertEqual(lifecycle.staging_slot.transfer_id, "transfer-1")
        self.assertFalse(lifecycle.staging_release.active)
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_data_plane_completion_envelope_preserves_status_failure_release(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            status_response=DaemonResponse(ok=False, error="unknown transfer"),
        )
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())
        payload = lifecycle.completion_envelope().as_dict()

        self.assertEqual(payload["final_state"], "status_failed")
        self.assertIn("unknown transfer", payload["error"])
        self.assertEqual(payload["worker_result"]["state"], "unsupported")
        self.assertEqual(payload["daemon_status_update"]["state"], "failed")
        self.assertIsNone(payload["daemon_status_response"])
        self.assertIsNone(payload["daemon_cleanup_response"])
        self.assertEqual(payload["staging_release"]["slot_id"], "staging-1")
        self.assertFalse(payload["staging_release"]["active"])
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_client_lifecycle_cleanup_failure_releases_staging(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            cleanup_response=DaemonResponse(ok=False, error="unknown reservation"),
        )
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "cleanup_failed")
        self.assertEqual(lifecycle.staging_slot.transfer_id, "transfer-1")
        self.assertFalse(lifecycle.staging_release.active)
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_data_plane_completion_envelope_preserves_cleanup_failure_release(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            cleanup_response=DaemonResponse(ok=False, error="unknown reservation"),
        )
        staging_pool = WorkerStagingPool()
        client = WorkerTransferClient(daemon_client, staging_pool=staging_pool)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())
        payload = WorkerDataPlaneCompletionEnvelope.from_lifecycle(lifecycle).as_dict()

        self.assertEqual(payload["final_state"], "cleanup_failed")
        self.assertIn("unknown reservation", payload["error"])
        self.assertEqual(payload["worker_result"]["state"], "unsupported")
        self.assertEqual(payload["daemon_status_update"]["state"], "failed")
        self.assertTrue(payload["daemon_status_response"]["ok"])
        self.assertIsNone(payload["daemon_cleanup_response"])
        self.assertEqual(payload["staging_release"]["slot_id"], "staging-1")
        self.assertFalse(payload["staging_release"]["active"])
        self.assertEqual(staging_pool.describe(), {"active_slots": {}})

    def test_worker_client_lifecycle_records_status_and_cleanup_without_custom_pool(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        client = WorkerTransferClient(daemon_client)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "unsupported")
        self.assertEqual(lifecycle.result.state, WorkerTransferState.UNSUPPORTED)
        self.assertEqual(lifecycle.status_response, daemon_client.status_response)
        self.assertEqual(lifecycle.cleanup_response, daemon_client.cleanup_response)
        self.assertEqual(lifecycle.cleanup_target_kind, "reservation")
        self.assertEqual(lifecycle.cleanup_target_id, "lease-1")
        self.assertEqual(lifecycle.status_update["state"], "failed")
        self.assertIn("not implemented", lifecycle.status_update["error"])
        self.assertFalse(lifecycle.staging_release.active)
        self.assertEqual(daemon_client.status_updates[0]["state"], "failed")
        self.assertEqual(daemon_client.cleanup_requests[0]["target_id"], "lease-1")

    def test_worker_client_lifecycle_skips_cleanup_for_complete_result(self) -> None:
        class CompleteExecutor:
            def execute(
                self,
                request: WorkerTransferRequest,
                staging_slot: WorkerStagingSlot,
            ) -> WorkerTransferResult:
                return WorkerTransferResult(
                    transfer_id=request.transfer_id,
                    state=WorkerTransferState.COMPLETE,
                    bytes_completed=64,
                )

        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        client = WorkerTransferClient(daemon_client, executor=CompleteExecutor())

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "complete")
        self.assertEqual(lifecycle.cleanup_target_kind, "reservation")
        self.assertIsNone(lifecycle.cleanup_target_id)
        self.assertTrue(lifecycle.cleanup_response.payload["cleanup_skipped"])
        self.assertEqual(daemon_client.cleanup_requests, [])
        self.assertEqual(lifecycle.status_update["state"], "complete")
        self.assertEqual(lifecycle.status_update["bytes_completed"], 64)
        self.assertEqual(daemon_client.status_updates[0]["state"], "complete")

    def test_worker_client_lifecycle_records_authorization_failure_cleanup(self) -> None:
        daemon_client = FakeDaemonClient(DaemonResponse(ok=False, error="denied"))
        client = WorkerTransferClient(daemon_client)

        lifecycle = client.submit_report_cleanup_lifecycle(authorization_request())

        self.assertEqual(lifecycle.final_state, "authorization_failed")
        self.assertIn("denied", lifecycle.error)
        self.assertIsNone(lifecycle.worker_request)
        self.assertIsNone(lifecycle.result)
        self.assertEqual(lifecycle.cleanup_target_kind, "reservation")
        self.assertEqual(lifecycle.cleanup_target_id, "lease-1")
        self.assertEqual(daemon_client.cleanup_requests[0]["reason"], "worker_authorization_failed")

    def test_worker_service_returns_unsupported_lifecycle_payload(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        service = WorkerTransferService(daemon_client)

        payload = service.handle(authorization_request())

        self.assertEqual(payload["final_state"], "unsupported")
        self.assertEqual(payload["result"]["state"], "unsupported")
        self.assertEqual(payload["status_update"]["state"], "failed")
        self.assertEqual(payload["cleanup_target"]["target_id"], "lease-1")
        self.assertEqual(daemon_client.status_updates[0]["state"], "failed")
        self.assertEqual(daemon_client.cleanup_requests[0]["target_kind"], "reservation")

    def test_worker_service_returns_authorization_denial_lifecycle_payload(self) -> None:
        daemon_client = FakeDaemonClient(DaemonResponse(ok=False, error="denied"))
        service = WorkerTransferService(daemon_client)

        payload = service.handle(authorization_request())

        self.assertEqual(payload["final_state"], "authorization_failed")
        self.assertIn("denied", payload["error"])
        self.assertIsNone(payload["worker_request"])
        self.assertEqual(payload["cleanup_target"]["target_id"], "lease-1")
        self.assertEqual(daemon_client.cleanup_requests[0]["reason"], "worker_authorization_failed")

    def test_worker_service_returns_status_failure_lifecycle_payload(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            status_response=DaemonResponse(ok=False, error="unknown transfer"),
        )
        service = WorkerTransferService(daemon_client)

        payload = service.handle(authorization_request())

        self.assertEqual(payload["final_state"], "status_failed")
        self.assertIn("unknown transfer", payload["error"])
        self.assertEqual(payload["status_update"]["state"], "failed")
        self.assertIsNone(payload["cleanup_target"])
        self.assertEqual(daemon_client.cleanup_requests, [])

    def test_worker_service_returns_cleanup_failure_lifecycle_payload(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            cleanup_response=DaemonResponse(ok=False, error="unknown reservation"),
        )
        service = WorkerTransferService(daemon_client)

        payload = service.handle(authorization_request())

        self.assertEqual(payload["final_state"], "cleanup_failed")
        self.assertIn("unknown reservation", payload["error"])
        self.assertEqual(payload["status_update"]["state"], "failed")
        self.assertEqual(payload["cleanup_target"]["target_id"], "lease-1")
        self.assertEqual(daemon_client.cleanup_requests[0]["target_id"], "lease-1")

    def test_worker_service_handle_lifecycle_returns_record(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        service = WorkerTransferService(daemon_client)

        lifecycle = service.handle_lifecycle(authorization_request())

        self.assertIsInstance(lifecycle, WorkerTransferLifecycleRecord)
        self.assertEqual(lifecycle.final_state, "unsupported")

    def test_worker_authorization_payload_parser_accepts_plain_dict(self) -> None:
        request = parse_worker_authorization_request_payload(
            authorization_request_payload()
        )

        self.assertEqual(request.transfer_id, "transfer-1")
        self.assertEqual(request.lease_id, "lease-1")
        self.assertEqual(request.direction, "h2d")
        self.assertEqual(request.ranges[0]["bytes"], 16)
        self.assertEqual(request.relay_gpu, 1)

    def test_worker_authorization_payload_parser_accepts_nested_dict(self) -> None:
        request = parse_worker_authorization_request_payload(
            {"authorization_request": authorization_request_payload()}
        )

        self.assertEqual(request.session_id, "session-1")
        self.assertEqual(request.src_buffer_id, "cpu-buffer")

    def test_worker_authorization_payload_parser_rejects_missing_required_field(self) -> None:
        payload = authorization_request_payload()
        payload.pop("token")

        with self.assertRaisesRegex(ValueError, "missing worker authorization field: token"):
            parse_worker_authorization_request_payload(payload)

    def test_worker_authorization_payload_parser_rejects_invalid_direction(self) -> None:
        payload = authorization_request_payload()
        payload["direction"] = "sideways"

        with self.assertRaisesRegex(ValueError, "direction must be h2d or d2h"):
            parse_worker_authorization_request_payload(payload)

    def test_worker_authorization_payload_parser_rejects_invalid_range(self) -> None:
        payload = authorization_request_payload()
        payload["ranges"] = [{"src_offset": 0, "dst_offset": 0, "bytes": 0}]

        with self.assertRaisesRegex(ValueError, "range bytes must be positive"):
            parse_worker_authorization_request_payload(payload)

    def test_worker_service_handle_payload_preserves_lifecycle_output(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        service = WorkerTransferService(daemon_client)

        payload = service.handle_payload(authorization_request_payload())

        self.assertEqual(payload["authorization_request"]["transfer_id"], "transfer-1")
        self.assertEqual(payload["final_state"], "unsupported")
        self.assertEqual(payload["status_update"]["state"], "failed")
        self.assertEqual(payload["cleanup_target"]["target_id"], "lease-1")

    def test_worker_service_request_envelope_serializes_payload(self) -> None:
        envelope = WorkerServiceRequestEnvelope(
            payload=authorization_request_payload(),
            cleanup_target_kind="session",
        )

        payload = envelope.as_dict()

        self.assertEqual(payload["cleanup_target_kind"], "session")
        self.assertEqual(payload["payload"]["transfer_id"], "transfer-1")

    def test_worker_service_request_envelope_rejects_invalid_cleanup_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "cleanup_target_kind"):
            WorkerServiceRequestEnvelope(
                payload=authorization_request_payload(),
                cleanup_target_kind="job",
            )

    def test_worker_service_response_envelope_serializes_error(self) -> None:
        response = WorkerServiceResponseEnvelope.from_error("bad payload")

        payload = response.as_dict()

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "bad payload")
        self.assertEqual(payload["final_state"], "parse_failed")
        self.assertIsNone(payload["lifecycle"])

    def test_worker_service_returns_success_envelope_payload(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        service = WorkerTransferService(daemon_client)

        response = service.handle_envelope_payload(
            WorkerServiceRequestEnvelope(payload=authorization_request_payload())
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["final_state"], "unsupported")
        self.assertEqual(response["lifecycle"]["result"]["state"], "unsupported")
        self.assertEqual(response["lifecycle"]["cleanup_target"]["target_id"], "lease-1")
        self.assertEqual(response["completion"]["worker_result"]["state"], "unsupported")
        self.assertEqual(response["completion"]["daemon_status_update"]["state"], "failed")
        self.assertTrue(response["completion"]["daemon_cleanup_response"]["ok"])
        self.assertEqual(response["completion"]["staging_release"]["slot_id"], "staging-1")
        self.assertFalse(response["completion"]["staging_release"]["active"])

    def test_worker_service_returns_malformed_payload_envelope(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload())
        )
        service = WorkerTransferService(daemon_client)
        payload = authorization_request_payload()
        payload.pop("token")

        response = service.handle_envelope_payload(payload)

        self.assertFalse(response["ok"])
        self.assertEqual(response["final_state"], "parse_failed")
        self.assertIn("missing worker authorization field: token", response["error"])
        self.assertIsNone(response["lifecycle"])
        self.assertIsNone(response["completion"])
        self.assertEqual(daemon_client.requests, [])

    def test_worker_service_returns_status_failure_envelope(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            status_response=DaemonResponse(ok=False, error="unknown transfer"),
        )
        service = WorkerTransferService(daemon_client)

        response = service.handle_envelope_payload(authorization_request_payload())

        self.assertTrue(response["ok"])
        self.assertEqual(response["final_state"], "status_failed")
        self.assertIn("unknown transfer", response["error"])
        self.assertEqual(response["lifecycle"]["status_update"]["state"], "failed")
        self.assertEqual(response["completion"]["daemon_status_update"]["state"], "failed")
        self.assertIsNone(response["completion"]["daemon_status_response"])
        self.assertIsNone(response["completion"]["daemon_cleanup_response"])
        self.assertFalse(response["completion"]["staging_release"]["active"])

    def test_worker_service_returns_cleanup_failure_envelope(self) -> None:
        daemon_client = FakeDaemonClient(
            DaemonResponse(ok=True, payload=authorization_payload()),
            cleanup_response=DaemonResponse(ok=False, error="unknown reservation"),
        )
        service = WorkerTransferService(daemon_client)

        response = service.handle_envelope_payload(authorization_request_payload())

        self.assertTrue(response["ok"])
        self.assertEqual(response["final_state"], "cleanup_failed")
        self.assertIn("unknown reservation", response["error"])
        self.assertEqual(response["lifecycle"]["cleanup_target"]["target_id"], "lease-1")
        self.assertEqual(response["completion"]["daemon_status_update"]["state"], "failed")
        self.assertTrue(response["completion"]["daemon_status_response"]["ok"])
        self.assertIsNone(response["completion"]["daemon_cleanup_response"])
        self.assertFalse(response["completion"]["staging_release"]["active"])


if __name__ == "__main__":
    unittest.main()
