from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import unittest

from turbobus.client import CudaIpcDeviceBuffer, SharedPinnedCpuBufferAllocator
from turbobus.client_transfer import make_worker_managed_transfer_client
from turbobus.daemon.server import TurboBusDaemon
from turbobus.daemon.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
    StaticTopologyProvider,
)
from turbobus.schema import DaemonResponse
from turbobus.worker import (
    CudaWorkerExecutor,
    WorkerDataPlaneCompletionEnvelope,
    WorkerDataPlaneResourceBinder,
    WorkerServiceRequestEnvelope,
    WorkerServiceEndpoint,
    WorkerServiceSocketClient,
    WorkerServiceUnixSocketTransport,
    WorkerStagingSlot,
    WorkerTransferClient,
    WorkerTransferRequest,
    WorkerTransferResult,
    WorkerTransferService,
    WorkerTransferState,
)


class CompleteExecutor:
    def __init__(self) -> None:
        self.requests: list[WorkerTransferRequest] = []

    def execute(
        self,
        request: WorkerTransferRequest,
        staging_slot: WorkerStagingSlot,
    ) -> WorkerTransferResult:
        self.requests.append(request)
        return WorkerTransferResult(
            transfer_id=request.transfer_id,
            state=WorkerTransferState.COMPLETE,
            bytes_completed=sum(item["bytes"] for item in request.data_plane.ranges),
            metadata={"staging_slot_id": staging_slot.slot_id},
        )


class FakeCudaBackend:
    def export_device_ipc_handle(self, device_ptr: int) -> bytes:
        return b"g" * 64


class EnvelopeWorkerClient:
    def __init__(self, service: WorkerTransferService) -> None:
        self.service = service
        self.envelopes: list[WorkerServiceRequestEnvelope] = []

    def submit_envelope(
        self,
        envelope: WorkerServiceRequestEnvelope,
    ) -> WorkerDataPlaneCompletionEnvelope:
        self.envelopes.append(envelope)
        response = self.service.handle_envelope(envelope)
        if response.completion is None:
            return WorkerDataPlaneCompletionEnvelope(
                ok=response.ok,
                final_state=response.final_state,
                error=response.error,
            )
        completion = dict(response.completion)
        completion["ok"] = bool(response.ok and completion.get("ok", True))
        return WorkerDataPlaneCompletionEnvelope(**completion)


class WorkerManagedTransferClientTest(unittest.TestCase):
    def test_fetch_shared_cpu_to_cuda_ipc_runs_daemon_worker_completion(self) -> None:
        daemon = daemon_with_relay_path()
        executor = CompleteExecutor()
        worker_client = WorkerTransferClient(daemon, executor=executor)
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=worker_client,
            max_inflight_chunks=8,
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as source:
            source.write(b"TurboBus")
            target = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                device_index=0,
                size_bytes=64,
                device_ptr=4096,
                backend=FakeCudaBackend(),
            )

            result = transfer_client.fetch_shared_cpu_to_cuda_ipc(
                source,
                target,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                chunk_bytes=16,
                mode="relay",
            )

        self.assertEqual(result.state, "complete")
        self.assertEqual(result.bytes_completed, 16)
        self.assertEqual(result.source_buffer_id, "cpu-buffer")
        self.assertEqual(result.target_buffer_id, "gpu-buffer")
        self.assertEqual(result.authorization_request.src_buffer_id, "cpu-buffer")
        self.assertEqual(result.authorization_request.dst_buffer_id, "gpu-buffer")
        self.assertEqual(result.authorization_request.ranges, ())
        self.assertEqual(result.worker_lifecycle.final_state, "complete")
        self.assertEqual(
            result.worker_lifecycle.cleanup_target_id,
            result.lease_token["lease_id"],
        )
        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(
            executor.requests[0].authorization.src_buffer.handle_type,
            "shared_pinned_cpu",
        )
        self.assertEqual(
            executor.requests[0].authorization.dst_buffer.handle_type,
            "cuda_ipc_device",
        )
        self.assertEqual(executor.requests[0].authorization.ranges[0]["bytes"], 16)
        self.assertEqual(
            executor.requests[0].data_plane.plan,
            result.plan["plan"],
        )
        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        status = daemon.transfer_status(result.transfer_id)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "complete")

    def test_fetch_shared_cpu_to_cuda_ipc_accepts_envelope_worker_client(self) -> None:
        daemon = daemon_with_relay_path()
        executor = CompleteExecutor()
        worker_client = EnvelopeWorkerClient(
            WorkerTransferService(
                daemon,
                transfer_client=WorkerTransferClient(daemon, executor=executor),
            )
        )
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=worker_client,
            max_inflight_chunks=8,
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as source:
            source.write(b"TurboBus")
            target = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                device_index=0,
                size_bytes=64,
                device_ptr=4096,
                backend=FakeCudaBackend(),
            )

            result = transfer_client.fetch_shared_cpu_to_cuda_ipc(
                source,
                target,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                chunk_bytes=16,
                mode="relay",
            )

        self.assertEqual(result.state, "complete")
        self.assertEqual(result.bytes_completed, 16)
        self.assertIsNone(result.worker_lifecycle)
        self.assertIsNotNone(result.worker_completion)
        self.assertEqual(result.worker_completion.final_state, "complete")
        self.assertEqual(result.worker_completion.transfer_id, result.transfer_id)
        self.assertEqual(result.worker_completion.lease_id, result.lease_token["lease_id"])
        self.assertEqual(worker_client.envelopes[0].cleanup_target_kind, "reservation")
        self.assertEqual(worker_client.envelopes[0].payload["src_buffer_id"], "cpu-buffer")
        self.assertEqual(worker_client.envelopes[0].payload["dst_buffer_id"], "gpu-buffer")
        self.assertEqual(worker_client.envelopes[0].payload["ranges"], [])
        self.assertEqual(len(executor.requests), 1)
        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    def test_worker_authorization_uses_daemon_plan_chunks(self) -> None:
        daemon = daemon_with_relay_path()
        executor = CompleteExecutor()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(daemon, executor=executor),
            max_inflight_chunks=8,
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as source:
            target = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                device_index=0,
                size_bytes=64,
                device_ptr=4096,
                backend=FakeCudaBackend(),
            )

            result = transfer_client.fetch_shared_cpu_to_cuda_ipc(
                source,
                target,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 64},),
                chunk_bytes=16,
                mode="relay",
            )

        self.assertEqual(result.bytes_completed, 64)
        self.assertEqual(result.authorization_request.ranges, ())
        self.assertEqual(
            tuple(executor.requests[0].authorization.ranges),
            (
                {"src_offset": 0, "dst_offset": 0, "bytes": 16},
                {"src_offset": 16, "dst_offset": 16, "bytes": 16},
                {"src_offset": 32, "dst_offset": 32, "bytes": 16},
                {"src_offset": 48, "dst_offset": 48, "bytes": 16},
            ),
        )
        self.assertEqual(executor.requests[0].data_plane.plan, result.plan["plan"])

    def test_worker_managed_transfer_rejects_pool_plan_and_releases_lease(self) -> None:
        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(daemon, executor=CompleteExecutor()),
            max_inflight_chunks=8,
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as source:
            target = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                device_index=0,
                size_bytes=64,
                device_ptr=4096,
                backend=FakeCudaBackend(),
            )

            with self.assertRaisesRegex(RuntimeError, "relay-only daemon plan"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 64},),
                    chunk_bytes=16,
                    mode="pool",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain sockets are unavailable")
    def test_fetch_shared_cpu_to_cuda_ipc_can_use_worker_socket_boundary(self) -> None:
        daemon = daemon_with_relay_path()
        executor = CompleteExecutor()
        worker_service = WorkerTransferService(
            daemon,
            transfer_client=WorkerTransferClient(daemon, executor=executor),
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "worker.sock")
            transport = WorkerServiceUnixSocketTransport(
                WorkerServiceEndpoint(service=worker_service),
                socket_path,
            )
            stop_event = threading.Event()
            thread = threading.Thread(
                target=transport.serve_forever,
                args=(stop_event,),
                daemon=True,
            )
            thread.start()
            try:
                _wait_for_socket(self, socket_path)
                transfer_client = make_worker_managed_transfer_client(
                    daemon,
                    target_gpu=0,
                    relay_gpus=[1],
                    worker_client=WorkerServiceSocketClient(socket_path),
                    max_inflight_chunks=8,
                )

                with allocator.allocate("cpu-buffer", "job-1", 64) as source:
                    source.write(b"TurboBus")
                    target = CudaIpcDeviceBuffer.from_device_pointer(
                        buffer_id="gpu-buffer",
                        job_id="job-1",
                        device_index=0,
                        size_bytes=64,
                        device_ptr=4096,
                        backend=FakeCudaBackend(),
                    )

                    result = transfer_client.fetch_shared_cpu_to_cuda_ipc(
                        source,
                        target,
                        ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                        chunk_bytes=16,
                        mode="relay",
                    )
            finally:
                stop_event.set()
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result.state, "complete")
        self.assertEqual(result.bytes_completed, 16)
        self.assertIsNone(result.worker_lifecycle)
        self.assertIsNotNone(result.worker_completion)
        self.assertEqual(result.worker_completion.final_state, "complete")
        self.assertEqual(result.worker_completion.transfer_id, result.transfer_id)
        self.assertEqual(result.worker_completion.lease_id, result.lease_token["lease_id"])
        self.assertEqual(result.worker_completion.worker_result["state"], "complete")
        self.assertEqual(len(executor.requests), 1)
        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    def test_worker_managed_transfer_rejects_direct_fallback_without_relay_lease(self) -> None:
        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(daemon, executor=CompleteExecutor()),
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as source:
            target = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                device_index=0,
                size_bytes=64,
                device_ptr=4096,
                backend=FakeCudaBackend(),
            )

            with self.assertRaisesRegex(RuntimeError, "exactly one relay lease"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="direct",
                )

    def test_worker_managed_transfer_surfaces_worker_failure(self) -> None:
        class FailedExecutor:
            def execute(self, request, staging_slot):
                return WorkerTransferResult(
                    transfer_id=request.transfer_id,
                    state=WorkerTransferState.FAILED,
                    error="copy failed",
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(daemon, executor=FailedExecutor()),
        )
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-client-worker-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as source:
            target = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="gpu-buffer",
                job_id="job-1",
                device_index=0,
                size_bytes=64,
                device_ptr=4096,
                backend=FakeCudaBackend(),
            )

            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

    def test_factory_defaults_to_cuda_worker_executor_with_resource_binding(self) -> None:
        daemon = daemon_with_relay_path()

        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
        )

        self.assertIsInstance(
            transfer_client.worker_client.executor,
            CudaWorkerExecutor,
        )
        self.assertIsInstance(
            transfer_client.worker_client.resource_binder,
            WorkerDataPlaneResourceBinder,
        )


def daemon_with_relay_path() -> TurboBusDaemon:
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
    daemon.put_profile(
        target_gpu=0,
        relay_gpus=[1],
        profile={
            "target_device": 0,
            "direct_h2d_bw_gbps": 1.0,
            "direct_d2h_bw_gbps": 1.0,
            "relays": [
                {
                    "relay_device": 1,
                    "target_device": 0,
                    "h2d_bw_gbps": 8.0,
                    "d2h_bw_gbps": 7.0,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 8.0,
                    "effective_d2h_bw_gbps": 7.0,
                    "p2p_enabled": True,
                }
            ],
        },
    )
    return daemon


def _wait_for_socket(test_case: unittest.TestCase, socket_path: str) -> None:
    for _ in range(100):
        if os.path.exists(socket_path):
            return
        time.sleep(0.01)
    test_case.fail(f"worker socket was not created: {socket_path}")


if __name__ == "__main__":
    unittest.main()
