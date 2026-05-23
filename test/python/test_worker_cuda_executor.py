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
    direct_bytes = 0
    direct_chunks = 0
    relay_bytes = 16
    relay_chunks = 1


class FakePoolStats:
    bytes = 64
    direct_bytes = 16
    direct_chunks = 1
    relay_bytes = 48
    relay_chunks = 3


class FakeD2HStats:
    bytes = 16
    direct_bytes = 0
    direct_chunks = 0
    relay_bytes = 16
    relay_chunks = 1


class FakeBackend:
    stats_result = FakeStats()

    def __init__(self) -> None:
        self.plan_payloads = []
        self.create_runtime_options = []
        self.initialize_calls = []
        self.fetch_calls = []
        self.offload_calls = []
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

    def offload_plan_to_cpu(
        self,
        runtime,
        target_ptr,
        target_bytes,
        host_ptr,
        host_bytes,
        plan,
    ):
        self.offload_calls.append(
            (runtime, target_ptr, target_bytes, host_ptr, host_bytes, plan)
        )
        return "handle-2"

    def wait(self, runtime, handle):
        self.wait_calls.append((runtime, handle))

    def stats(self, runtime, handle):
        self.stats_calls.append((runtime, handle))
        return self.stats_result


class FakeCpuBuffer:
    address = 1000
    size_bytes = 64

    def close(self):
        pass


def relay_plan(direction: str = "h2d") -> dict[str, object]:
    return {
        "total_bytes": 16,
        "chunk_bytes": 16,
        "assignments": [
            {
                "path": {
                    "kind": "relay",
                    "direction": direction,
                    "target_device": 0,
                    "relay_device": 1,
                    "enabled": True,
                },
                "chunks": [{"src_offset": 4, "dst_offset": 8, "bytes": 16}],
                "bytes": 16,
                "chunk_count": 1,
            }
        ],
    }


def d2h_relay_plan() -> dict[str, object]:
    return {
        "total_bytes": 16,
        "chunk_bytes": 16,
        "assignments": [
            {
                "path": {
                    "kind": "relay",
                    "direction": "d2h",
                    "target_device": 0,
                    "relay_device": 1,
                    "enabled": True,
                },
                "chunks": [{"src_offset": 8, "dst_offset": 4, "bytes": 16}],
                "bytes": 16,
                "chunk_count": 1,
            }
        ],
    }


def pool_plan() -> dict[str, object]:
    return {
        "total_bytes": 64,
        "chunk_bytes": 16,
        "assignments": [
            {
                "path": {
                    "kind": "direct",
                    "direction": "h2d",
                    "target_device": 0,
                    "relay_device": -1,
                    "enabled": True,
                },
                "chunks": [{"src_offset": 0, "dst_offset": 0, "bytes": 16}],
                "bytes": 16,
                "chunk_count": 1,
            },
            {
                "path": {
                    "kind": "relay",
                    "direction": "h2d",
                    "target_device": 0,
                    "relay_device": 1,
                    "enabled": True,
                },
                "chunks": [
                    {"src_offset": 16, "dst_offset": 16, "bytes": 16},
                    {"src_offset": 32, "dst_offset": 32, "bytes": 16},
                    {"src_offset": 48, "dst_offset": 48, "bytes": 16},
                ],
                "bytes": 48,
                "chunk_count": 3,
            },
        ],
    }


def worker_request(
    direction: str = "h2d",
    *,
    plan: dict[str, object] | None = None,
    ranges=({"src_offset": 4, "dst_offset": 8, "bytes": 16},),
) -> WorkerTransferRequest:
    if plan is None:
        plan = relay_plan(direction)
    authorization = WorkerTransferAuthorization(
        transfer_id="transfer-1",
        lease_id="lease-1",
        session_id="session-1",
        job_id="job-1",
        src_buffer=BufferRegistration(
            buffer_id="cpu-buffer" if direction == "h2d" else "gpu-buffer",
            job_id="job-1",
            kind="cpu_pinned" if direction == "h2d" else "gpu",
            size_bytes=64,
            device_index=None if direction == "h2d" else 0,
            pinned=direction == "h2d",
            handle_type="shared_pinned_cpu" if direction == "h2d" else "cuda_ipc_device",
            metadata={
                "shared_memory_name": "tb-job-1-src",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 64,
            }
            if direction == "h2d"
            else {"cuda_ipc_handle": (b"t" * 64).hex()},
        ),
        dst_buffer=BufferRegistration(
            buffer_id="gpu-buffer" if direction == "h2d" else "cpu-buffer",
            job_id="job-1",
            kind="gpu" if direction == "h2d" else "cpu_pinned",
            size_bytes=64,
            device_index=0 if direction == "h2d" else None,
            pinned=direction == "d2h",
            handle_type="cuda_ipc_device" if direction == "h2d" else "shared_pinned_cpu",
            metadata={"cuda_ipc_handle": (b"t" * 64).hex()}
            if direction == "h2d"
            else {
                "shared_memory_name": "tb-job-1-dst",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 64,
            },
        ),
        direction=direction,
        ranges=ranges,
        relay_gpu=1,
        plan=plan,
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
            cpu_buffer=FakeCpuBuffer(),
            device_ptr=2000,
            device_bytes=64,
            cuda_host_registered=True,
        )
        backend = FakeBackend()
        executor = CudaWorkerExecutor(backend=backend)

        result = executor.execute_bound(request, slot, resources)

        self.assertEqual(result.state, WorkerTransferState.COMPLETE)
        self.assertEqual(result.bytes_completed, 16)
        self.assertEqual(result.metadata["executor"], "cuda_worker")
        self.assertEqual(result.metadata["path"], "relay_h2d")
        self.assertEqual(result.metadata["plan_source"], "daemon")
        self.assertEqual(
            backend.plan_payloads,
            [relay_plan()],
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

    def test_executor_requires_daemon_plan(self) -> None:
        request = worker_request(plan={})
        slot = WorkerStagingPool().allocate(request.data_plane)
        resources = WorkerDataPlaneResources(
            request=request.data_plane,
            cpu_buffer=FakeCpuBuffer(),
            device_ptr=2000,
            device_bytes=64,
        )

        result = CudaWorkerExecutor(backend=FakeBackend()).execute_bound(
            request,
            slot,
            resources,
        )

        self.assertEqual(result.state, WorkerTransferState.FAILED)
        self.assertIn("daemon-issued transfer plan", result.error)

    def test_executor_runs_h2d_pool_plan_and_waits(self) -> None:
        request = worker_request(
            plan=pool_plan(),
            ranges=(
                {"src_offset": 16, "dst_offset": 16, "bytes": 16},
                {"src_offset": 32, "dst_offset": 32, "bytes": 16},
                {"src_offset": 48, "dst_offset": 48, "bytes": 16},
            ),
        )
        slot = WorkerStagingPool(slot_id_factory=lambda: "staging-1").allocate(
            request.data_plane
        )
        resources = WorkerDataPlaneResources(
            request=request.data_plane,
            cpu_buffer=FakeCpuBuffer(),
            device_ptr=2000,
            device_bytes=64,
            cuda_host_registered=True,
        )
        backend = FakeBackend()
        backend.stats_result = FakePoolStats()
        executor = CudaWorkerExecutor(backend=backend)

        result = executor.execute_bound(request, slot, resources)

        self.assertEqual(result.state, WorkerTransferState.COMPLETE)
        self.assertEqual(result.bytes_completed, 64)
        self.assertEqual(result.metadata["path"], "pool_h2d")
        self.assertEqual(result.metadata["direct_bytes"], 16)
        self.assertEqual(result.metadata["direct_chunks"], 1)
        self.assertEqual(result.metadata["relay_bytes"], 48)
        self.assertEqual(result.metadata["relay_chunks"], 3)
        self.assertEqual(backend.plan_payloads, [pool_plan()])
        self.assertEqual(backend.initialize_calls[0][1:], (0, [1]))

    def test_executor_runs_d2h_relay_plan_and_waits(self) -> None:
        request = worker_request(
            direction="d2h",
            plan=d2h_relay_plan(),
            ranges=({"src_offset": 8, "dst_offset": 4, "bytes": 16},),
        )
        slot = WorkerStagingPool().allocate(request.data_plane)
        resources = WorkerDataPlaneResources(
            request=request.data_plane,
            cpu_buffer=FakeCpuBuffer(),
            device_ptr=2000,
            device_bytes=64,
        )
        backend = FakeBackend()
        backend.stats_result = FakeD2HStats()

        result = CudaWorkerExecutor(backend=backend).execute_bound(
            request,
            slot,
            resources,
        )

        self.assertEqual(result.state, WorkerTransferState.COMPLETE)
        self.assertEqual(result.metadata["path"], "relay_d2h")
        self.assertEqual(backend.plan_payloads, [d2h_relay_plan()])
        self.assertEqual(backend.initialize_calls[0][1:], (0, [1]))
        self.assertEqual(
            backend.offload_calls[0][1:],
            (2000, 64, 1000, 64, "native-plan"),
        )
        self.assertEqual(backend.wait_calls[0][1], "handle-2")


if __name__ == "__main__":
    unittest.main()
