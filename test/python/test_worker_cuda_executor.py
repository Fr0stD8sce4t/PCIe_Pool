from __future__ import annotations

import unittest

from turbobus.schema import BufferRegistration, WorkerTransferAuthorization
from turbobus.worker import (
    CudaWorkerExecutor,
    WorkerDataPlaneResources,
    WorkerStagingPool,
    WorkerTransferRequest,
    WorkerTransferState,
)


class FakeRuntime:
    pass


class FakeStats:
    bytes = 16
    relay_bytes = 16
    relay_chunks = 1


class FakeBackend:
    def __init__(self) -> None:
        self.plan_payloads = []
        self.create_runtime_options = []
        self.initialize_calls = []
        self.fetch_calls = []
        self.wait_calls = []
        self.stats_calls = []

    def make_transfer_plan(self, plan):
        self.plan_payloads.append(plan)
        return "native-plan"

    def create_runtime(self, options):
        self.create_runtime_options.append(options)
        return FakeRuntime()

    def initialize_runtime(self, runtime, target_device, relay_gpus):
        self.initialize_calls.append((runtime, target_device, list(relay_gpus)))

    def fetch_plan_to_gpu(
        self,
        runtime,
        host_ptr,
        host_bytes,
        target_ptr,
        target_bytes,
        plan,
    ):
        self.fetch_calls.append(
            (runtime, host_ptr, host_bytes, target_ptr, target_bytes, plan)
        )
        return "handle-1"

    def wait(self, runtime, handle):
        self.wait_calls.append((runtime, handle))

    def stats(self, runtime, handle):
        self.stats_calls.append((runtime, handle))
        return FakeStats()


class FakeCpuBuffer:
    address = 1000
    size_bytes = 64

    def close(self):
        pass


def worker_request(direction: str = "h2d") -> WorkerTransferRequest:
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
            handle_type="shared_pinned_cpu",
            metadata={
                "shared_memory_name": "tb-job-1-src",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 64,
            },
        ),
        dst_buffer=BufferRegistration(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=64,
            device_index=0,
            handle_type="cuda_ipc_device",
            metadata={"cuda_ipc_handle": (b"t" * 64).hex()},
        ),
        direction=direction,
        ranges=({"src_offset": 4, "dst_offset": 8, "bytes": 16},),
        relay_gpu=1,
    )
    return WorkerTransferRequest.from_authorization(authorization)


class CudaWorkerExecutorTest(unittest.TestCase):
    def test_executor_runs_h2d_relay_plan_and_waits(self) -> None:
        request = worker_request()
        slot = WorkerStagingPool(slot_id_factory=lambda: "staging-1").allocate(
            request.data_plane
        )
        resources = WorkerDataPlaneResources(
            request=request.data_plane,
            source_cpu_buffer=FakeCpuBuffer(),
            target_device_ptr=2000,
            target_device_bytes=64,
            cuda_host_registered=True,
        )
        backend = FakeBackend()
        executor = CudaWorkerExecutor(backend=backend)

        result = executor.execute_bound(request, slot, resources)

        self.assertEqual(result.state, WorkerTransferState.COMPLETE)
        self.assertEqual(result.bytes_completed, 16)
        self.assertEqual(result.metadata["executor"], "cuda_worker")
        self.assertEqual(result.metadata["path"], "relay_h2d")
        self.assertEqual(
            backend.plan_payloads,
            [
                {
                    "total_bytes": 16,
                    "chunk_bytes": 16,
                    "assignments": [
                        {
                            "path": {
                                "kind": "relay",
                                "direction": "h2d",
                                "target_device": 0,
                                "relay_device": 1,
                                "enabled": True,
                            },
                            "chunks": [
                                {"src_offset": 4, "dst_offset": 8, "bytes": 16}
                            ],
                            "bytes": 16,
                            "chunk_count": 1,
                        }
                    ],
                }
            ],
        )
        self.assertEqual(backend.initialize_calls[0][1:], (0, [1]))
        self.assertEqual(
            backend.fetch_calls[0][1:],
            (1000, 64, 2000, 64, "native-plan"),
        )
        self.assertEqual(backend.wait_calls[0][1], "handle-1")
        self.assertEqual(backend.stats_calls[0][1], "handle-1")

    def test_executor_fails_without_bound_resources(self) -> None:
        request = worker_request()
        slot = WorkerStagingPool().allocate(request.data_plane)
        result = CudaWorkerExecutor(backend=FakeBackend()).execute(request, slot)

        self.assertEqual(result.state, WorkerTransferState.FAILED)
        self.assertIn("bound data-plane resources", result.error)

    def test_executor_rejects_d2h_until_worker_path_exists(self) -> None:
        request = worker_request(direction="d2h")
        slot = WorkerStagingPool().allocate(request.data_plane)
        resources = WorkerDataPlaneResources(
            request=request.data_plane,
            source_cpu_buffer=FakeCpuBuffer(),
            target_device_ptr=2000,
            target_device_bytes=64,
        )
        result = CudaWorkerExecutor(backend=FakeBackend()).execute_bound(
            request,
            slot,
            resources,
        )

        self.assertEqual(result.state, WorkerTransferState.FAILED)
        self.assertIn("h2d relay transfers", result.error)


if __name__ == "__main__":
    unittest.main()
