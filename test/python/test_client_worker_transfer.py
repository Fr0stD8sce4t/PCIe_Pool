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
from turbobus.runtime_engine import RuntimeOptions
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
            bytes_completed=planned_bytes(request),
            metadata={"staging_slot_id": staging_slot.slot_id},
        )


class FakeCudaBackend:
    def export_device_ipc_handle(self, device_ptr: int) -> bytes:
        return b"g" * 64


class FakeDirectBackend(FakeCudaBackend):
    def __init__(self) -> None:
        self.initialized = []
        self.fetches = []
        self.offloads = []
        self.registered = []
        self.unregistered = []

    def make_transfer_plan(self, plan):
        return plan

    def create_runtime(self, options):
        return {"options": options}

    def initialize_runtime(self, runtime, target_device, relay_gpus):
        self.initialized.append((target_device, tuple(relay_gpus)))

    def register_host_memory(self, host_ptr, bytes_):
        self.registered.append((host_ptr, bytes_))

    def unregister_host_memory(self, host_ptr):
        self.unregistered.append(host_ptr)

    def fetch_plan_to_gpu(
        self,
        runtime,
        host_ptr,
        host_bytes,
        target_ptr,
        target_bytes,
        plan,
    ):
        self.fetches.append((host_ptr, host_bytes, target_ptr, target_bytes, plan))
        return "fetch-handle"

    def offload_plan_to_cpu(
        self,
        runtime,
        target_ptr,
        target_bytes,
        host_ptr,
        host_bytes,
        plan,
    ):
        self.offloads.append((target_ptr, target_bytes, host_ptr, host_bytes, plan))
        return "offload-handle"

    def wait(self, runtime, handle):
        return None


def planned_bytes(request: WorkerTransferRequest) -> int:
    return sum(
        int(chunk["bytes"])
        for assignment in request.data_plane.plan.get("assignments", ()) or ()
        for chunk in assignment.get("chunks", ()) or ()
    )


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

        closed = transfer_client.close_session()
        self.assertTrue(closed.ok)
        profile = daemon.describe().payload
        self.assertEqual(profile["jobs"], {})
        self.assertEqual(profile["buffers"], {})

    def test_offload_cuda_ipc_to_shared_cpu_runs_daemon_worker_completion(self) -> None:
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

        source = CudaIpcDeviceBuffer.from_device_pointer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            device_index=0,
            size_bytes=64,
            device_ptr=4096,
            backend=FakeCudaBackend(),
        )
        with allocator.allocate("cpu-buffer", "job-1", 64) as target:
            result = transfer_client.offload_cuda_ipc_to_shared_cpu(
                source,
                target,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                chunk_bytes=16,
                mode="relay",
            )

        self.assertEqual(result.state, "complete")
        self.assertEqual(result.bytes_completed, 16)
        self.assertEqual(result.source_buffer_id, "gpu-buffer")
        self.assertEqual(result.target_buffer_id, "cpu-buffer")
        self.assertEqual(result.authorization_request.src_buffer_id, "gpu-buffer")
        self.assertEqual(result.authorization_request.dst_buffer_id, "cpu-buffer")
        self.assertEqual(result.authorization_request.direction, "d2h")
        self.assertEqual(result.authorization_request.ranges, ())
        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(executor.requests[0].authorization.direction, "d2h")
        self.assertEqual(
            executor.requests[0].authorization.src_buffer.handle_type,
            "cuda_ipc_device",
        )
        self.assertEqual(
            executor.requests[0].authorization.dst_buffer.handle_type,
            "shared_pinned_cpu",
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

    def test_worker_managed_transfer_requires_daemon_complete_status(self) -> None:
        class CompletionOnlyWorkerClient:
            def submit_envelope(
                self,
                envelope: WorkerServiceRequestEnvelope,
            ) -> WorkerDataPlaneCompletionEnvelope:
                return WorkerDataPlaneCompletionEnvelope(
                    ok=True,
                    transfer_id=str(envelope.payload["transfer_id"]),
                    lease_id=str(envelope.payload["lease_id"]),
                    final_state="complete",
                    worker_result={
                        "transfer_id": str(envelope.payload["transfer_id"]),
                        "state": "complete",
                        "bytes_completed": 16,
                    },
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=CompletionOnlyWorkerClient(),
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

            with self.assertRaisesRegex(
                RuntimeError,
                "daemon transfer status did not complete",
            ):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

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

    def test_worker_managed_transfer_preserves_requested_range_offsets(self) -> None:
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
                ranges=({"src_offset": 8, "dst_offset": 24, "bytes": 16},),
                chunk_bytes=8,
                mode="relay",
            )

        expected_ranges = (
            {"src_offset": 8, "dst_offset": 24, "bytes": 8},
            {"src_offset": 16, "dst_offset": 32, "bytes": 8},
        )
        self.assertEqual(result.state, "complete")
        self.assertEqual(tuple(executor.requests[0].authorization.ranges), expected_ranges)
        self.assertEqual(
            tuple(
                chunk
                for assignment in result.plan["plan"]["assignments"]
                for chunk in _worker_ranges(assignment["chunks"])
            ),
            expected_ranges,
        )

    def test_worker_managed_transfer_accepts_pool_plan_from_daemon(self) -> None:
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
                mode="pool",
            )

        self.assertEqual(result.state, "complete")
        self.assertEqual(result.plan["stats"]["resolved_mode"], "pool")
        self.assertEqual(result.authorization_request.ranges, ())
        self.assertEqual(
            tuple(executor.requests[0].authorization.ranges),
            tuple(
                chunk
                for assignment in result.plan["plan"]["assignments"]
                if assignment["path"]["kind"] == "relay"
                for chunk in _worker_ranges(assignment["chunks"])
            ),
        )
        self.assertTrue(
            any(
                assignment["path"]["kind"] == "direct"
                for assignment in executor.requests[0].data_plane.plan["assignments"]
            )
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

    def test_worker_managed_transfer_runs_direct_fallback_without_relay_lease(self) -> None:
        daemon = daemon_with_relay_path(max_inflight_chunks_per_relay=1)
        direct_backend = FakeDirectBackend()
        executor = CompleteExecutor()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(daemon, executor=executor),
            backend=direct_backend,
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
                mode="pool",
            )

        self.assertEqual(result.state, "complete")
        self.assertEqual(result.bytes_completed, 64)
        self.assertIsNone(result.lease_token)
        self.assertIsNone(result.authorization_request)
        self.assertIsNone(result.worker_lifecycle)
        self.assertIsNone(result.worker_completion)
        self.assertEqual(result.plan["stats"]["resolved_mode"], "direct")
        self.assertEqual(executor.requests, [])
        self.assertEqual(direct_backend.initialized, [(0, ())])
        self.assertEqual(len(direct_backend.fetches), 1)
        self.assertEqual(direct_backend.fetches[0][2], 4096)
        self.assertEqual(direct_backend.fetches[0][3], 64)
        self.assertEqual(direct_backend.fetches[0][4], result.plan["plan"])
        self.assertEqual(len(direct_backend.registered), 1)
        self.assertEqual(len(direct_backend.unregistered), 1)
        status = daemon.transfer_status(result.transfer_id)
        self.assertTrue(status.ok)
        self.assertEqual(status.payload["status"]["state"], "complete")
        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

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

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    def test_worker_managed_transfer_rejects_partial_complete_result(self) -> None:
        class PartialCompleteExecutor:
            def execute(self, request, staging_slot):
                return WorkerTransferResult(
                    transfer_id=request.transfer_id,
                    state=WorkerTransferState.COMPLETE,
                    bytes_completed=8,
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(
                daemon,
                executor=PartialCompleteExecutor(),
            ),
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

            with self.assertRaisesRegex(RuntimeError, "daemon-planned bytes"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    def test_worker_managed_transfer_cleans_daemon_reservation_on_executor_exception(self) -> None:
        class RaisingExecutor:
            def execute(self, request, staging_slot):
                raise RuntimeError("cuda launch failed")

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=WorkerTransferClient(daemon, executor=RaisingExecutor()),
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

            with self.assertRaisesRegex(RuntimeError, "cuda launch failed"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    def test_worker_managed_transfer_cleans_daemon_reservation_on_worker_boundary_exception(self) -> None:
        class RaisingWorkerClient:
            def submit_envelope(self, envelope):
                raise RuntimeError("worker boundary failed")

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=RaisingWorkerClient(),
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

            with self.assertRaisesRegex(RuntimeError, "worker boundary failed"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)

    def test_worker_managed_transfer_cleans_daemon_reservation_on_incomplete_worker_envelope(self) -> None:
        class IncompleteEnvelopeWorkerClient:
            def submit_envelope(self, envelope):
                return WorkerDataPlaneCompletionEnvelope(
                    ok=True,
                    transfer_id=str(envelope.payload["transfer_id"]),
                    lease_id=str(envelope.payload["lease_id"]),
                    final_state="failed",
                    error="worker helper did not execute",
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=IncompleteEnvelopeWorkerClient(),
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

            with self.assertRaisesRegex(RuntimeError, "worker helper did not execute"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        cleanup_events = profile["cleanup_events"]
        self.assertEqual(cleanup_events[-1]["reason"], "worker_completion_not_complete")

    def test_worker_managed_transfer_rejects_mismatched_worker_completion(self) -> None:
        class MismatchedCompletionWorkerClient:
            def submit_envelope(self, envelope):
                return WorkerDataPlaneCompletionEnvelope(
                    ok=True,
                    transfer_id=str(envelope.payload["transfer_id"]),
                    lease_id="wrong-lease",
                    final_state="complete",
                    worker_result={
                        "transfer_id": str(envelope.payload["transfer_id"]),
                        "state": "complete",
                        "bytes_completed": 16,
                    },
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=MismatchedCompletionWorkerClient(),
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

            with self.assertRaisesRegex(RuntimeError, "worker completion lease mismatch"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        cleanup_events = profile["cleanup_events"]
        self.assertEqual(cleanup_events[-1]["reason"], "worker_completion_invalid")

    def test_worker_managed_transfer_rejects_mismatched_worker_result(self) -> None:
        class MismatchedWorkerResultClient:
            def submit_envelope(self, envelope):
                return WorkerDataPlaneCompletionEnvelope(
                    ok=True,
                    transfer_id=str(envelope.payload["transfer_id"]),
                    lease_id=str(envelope.payload["lease_id"]),
                    final_state="complete",
                    worker_result={
                        "transfer_id": "wrong-transfer",
                        "state": "complete",
                        "bytes_completed": 16,
                    },
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=MismatchedWorkerResultClient(),
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

            with self.assertRaisesRegex(RuntimeError, "worker result transfer mismatch"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        cleanup_events = profile["cleanup_events"]
        self.assertEqual(cleanup_events[-1]["reason"], "worker_completion_invalid")

    def test_worker_managed_transfer_rejects_mismatched_daemon_status_response(self) -> None:
        class MismatchedStatusResponseClient:
            def submit_envelope(self, envelope):
                transfer_id = str(envelope.payload["transfer_id"])
                return WorkerDataPlaneCompletionEnvelope(
                    ok=True,
                    transfer_id=transfer_id,
                    lease_id=str(envelope.payload["lease_id"]),
                    final_state="complete",
                    worker_result={
                        "transfer_id": transfer_id,
                        "state": "complete",
                        "bytes_completed": 16,
                    },
                    daemon_status_response={
                        "ok": True,
                        "payload": {
                            "status": {
                                "transfer_id": "wrong-transfer",
                                "state": "complete",
                                "bytes_completed": 16,
                            }
                        },
                    },
                )

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            worker_client=MismatchedStatusResponseClient(),
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

            with self.assertRaisesRegex(
                RuntimeError,
                "worker daemon status response transfer mismatch",
            ):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        cleanup_events = profile["cleanup_events"]
        self.assertEqual(cleanup_events[-1]["reason"], "worker_completion_invalid")

    def test_worker_managed_transfer_cleans_daemon_reservation_on_status_query_failure(self) -> None:
        class CompletionOnlyWorkerClient:
            def submit_envelope(
                self,
                envelope: WorkerServiceRequestEnvelope,
            ) -> WorkerDataPlaneCompletionEnvelope:
                return WorkerDataPlaneCompletionEnvelope(
                    ok=True,
                    transfer_id=str(envelope.payload["transfer_id"]),
                    lease_id=str(envelope.payload["lease_id"]),
                    final_state="complete",
                    worker_result={
                        "transfer_id": str(envelope.payload["transfer_id"]),
                        "state": "complete",
                        "bytes_completed": 16,
                    },
                )

        class StatusQueryFailingDaemonClient:
            def __init__(self, daemon):
                self.daemon = daemon

            def __getattr__(self, name):
                return getattr(self.daemon, name)

            def transfer_status(self, transfer_id, **kwargs):
                return DaemonResponse(ok=False, error="status unavailable")

        daemon = daemon_with_relay_path()
        transfer_client = make_worker_managed_transfer_client(
            StatusQueryFailingDaemonClient(daemon),
            target_gpu=0,
            relay_gpus=[1],
            worker_client=CompletionOnlyWorkerClient(),
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

            with self.assertRaisesRegex(RuntimeError, "status unavailable"):
                transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    source,
                    target,
                    ranges=({"src_offset": 0, "dst_offset": 0, "bytes": 16},),
                    chunk_bytes=16,
                    mode="relay",
                )

        profile = daemon.describe().payload
        self.assertEqual(profile["reservations"], {})
        self.assertEqual(profile["relay_quotas"][1]["active_chunks"], 0)
        cleanup_events = profile["cleanup_events"]
        self.assertEqual(cleanup_events[-1]["reason"], "daemon_status_query_failed")

    def test_factory_defaults_to_cuda_worker_executor_with_resource_binding(self) -> None:
        daemon = daemon_with_relay_path()
        backend = FakeDirectBackend()
        runtime_options = RuntimeOptions(chunk_bytes=32)

        transfer_client = make_worker_managed_transfer_client(
            daemon,
            target_gpu=0,
            relay_gpus=[1],
            backend=backend,
            runtime_options=runtime_options,
        )

        self.assertIsInstance(
            transfer_client.worker_client.executor,
            CudaWorkerExecutor,
        )
        self.assertIsInstance(
            transfer_client.worker_client.resource_binder,
            WorkerDataPlaneResourceBinder,
        )
        self.assertIs(transfer_client.backend, backend)
        self.assertIs(transfer_client.runtime_options, runtime_options)
        self.assertIs(transfer_client.worker_client.executor.backend, backend)
        self.assertIs(transfer_client.worker_client.executor.options, runtime_options)
        self.assertIs(transfer_client.worker_client.resource_binder.backend, backend)


def daemon_with_relay_path(max_inflight_chunks_per_relay: int = 8) -> TurboBusDaemon:
    daemon = TurboBusDaemon(
        relay_gpus=[1],
        max_sessions_per_relay=1,
        max_inflight_chunks_per_relay=max_inflight_chunks_per_relay,
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


def _worker_ranges(chunks) -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "src_offset": int(chunk["src_offset"]),
            "dst_offset": int(chunk["dst_offset"]),
            "bytes": int(chunk["bytes"]),
        }
        for chunk in chunks
    )


def _wait_for_socket(test_case: unittest.TestCase, socket_path: str) -> None:
    for _ in range(100):
        if os.path.exists(socket_path):
            return
        time.sleep(0.01)
    test_case.fail(f"worker socket was not created: {socket_path}")


if __name__ == "__main__":
    unittest.main()
