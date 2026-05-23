from __future__ import annotations

import unittest

from turbobus.client import (
    CudaIpcDeviceBuffer,
    SharedPinnedCpuBuffer,
    SharedPinnedCpuBufferAllocator,
)
from turbobus.schema import BufferRegistration, DaemonResponse


class FakeCudaBackend:
    def __init__(self) -> None:
        self.set_device_calls: list[int] = []
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []
        self.export_ipc_calls: list[int] = []

    def set_device(self, device_index: int) -> None:
        self.set_device_calls.append(int(device_index))

    def register_host_memory(self, host_ptr: int, bytes_: int) -> None:
        self.register_calls.append((int(host_ptr), int(bytes_)))

    def unregister_host_memory(self, host_ptr: int) -> None:
        self.unregister_calls.append(int(host_ptr))

    def export_device_ipc_handle(self, device_ptr: int) -> bytes:
        self.export_ipc_calls.append(int(device_ptr))
        return b"c" * 64


class FakeDaemonClient:
    def __init__(self) -> None:
        self.register_buffer_calls: list[dict[str, object]] = []

    def register_buffer(self, **payload) -> DaemonResponse:
        self.register_buffer_calls.append(dict(payload))
        return DaemonResponse(ok=True, payload={"buffer": dict(payload)})


class SharedPinnedCpuBufferTest(unittest.TestCase):
    def test_allocator_creates_shared_pinned_cpu_registration(self) -> None:
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as buffer:
            registration = buffer.buffer_registration()

            self.assertEqual(registration.buffer_id, "cpu-buffer")
            self.assertEqual(registration.job_id, "job-1")
            self.assertEqual(registration.kind, "cpu_pinned")
            self.assertTrue(registration.pinned)
            self.assertEqual(registration.handle_type, "shared_pinned_cpu")
            self.assertEqual(registration.metadata["offset_bytes"], 0)
            self.assertEqual(registration.metadata["shared_memory_size_bytes"], 64)
            self.assertEqual(
                registration.metadata["shared_memory_name"],
                buffer.shared_memory_name,
            )

    def test_shared_memory_handle_can_be_opened_by_another_owner(self) -> None:
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as buffer:
            buffer.write(b"TurboBus", offset=4)
            opened = SharedPinnedCpuBuffer.open_from_registration(
                buffer.buffer_registration()
            )
            try:
                self.assertFalse(opened.owner)
                self.assertEqual(opened.read(8, offset=4), b"TurboBus")
                opened.write(b"relay", offset=16)
                self.assertEqual(buffer.read(5, offset=16), b"relay")
            finally:
                opened.close()

    def test_open_requires_shared_memory_size_metadata(self) -> None:
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-test")

        with allocator.allocate("cpu-buffer", "job-1", 64) as buffer:
            registration = BufferRegistration(
                buffer_id="cpu-buffer",
                job_id="job-1",
                kind="cpu_pinned",
                size_bytes=64,
                pinned=True,
                handle_type="shared_pinned_cpu",
                metadata={
                    "shared_memory_name": buffer.shared_memory_name,
                    "offset_bytes": 0,
                },
            )

            with self.assertRaisesRegex(ValueError, "shared_memory_size_bytes"):
                SharedPinnedCpuBuffer.open_from_registration(registration)

            self.assertFalse(buffer.closed)

    def test_buffer_registers_shared_memory_with_cuda_backend(self) -> None:
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-test")
        backend = FakeCudaBackend()

        with allocator.allocate("cpu-buffer", "job-1", 64) as buffer:
            self.assertFalse(buffer.closed)
            buffer.register_for_cuda(backend)
            first_address = backend.register_calls[0][0]
            self.assertTrue(buffer.cuda_registered)
            buffer.register_for_cuda(backend)
            buffer.unregister_from_cuda()

            self.assertGreater(first_address, 0)
            self.assertFalse(buffer.cuda_registered)
            self.assertEqual(backend.register_calls, [(first_address, 64)])
            self.assertEqual(backend.unregister_calls, [first_address])
        self.assertTrue(buffer.closed)

    def test_buffer_can_register_itself_with_daemon(self) -> None:
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-test")
        daemon_client = FakeDaemonClient()

        with allocator.allocate("cpu-buffer", "job-1", 64) as buffer:
            response = buffer.register_with_daemon(daemon_client)

            self.assertTrue(response.ok)
            self.assertEqual(len(daemon_client.register_buffer_calls), 1)
            payload = daemon_client.register_buffer_calls[0]
            self.assertEqual(payload["buffer_id"], "cpu-buffer")
            self.assertEqual(payload["job_id"], "job-1")
            self.assertEqual(payload["kind"], "cpu_pinned")
            self.assertTrue(payload["pinned"])
            self.assertEqual(payload["handle_type"], "shared_pinned_cpu")
            self.assertEqual(
                payload["metadata"]["shared_memory_name"],
                buffer.shared_memory_name,
            )

    def test_cuda_ipc_device_buffer_builds_daemon_registration(self) -> None:
        backend = FakeCudaBackend()
        daemon_client = FakeDaemonClient()

        buffer = CudaIpcDeviceBuffer.from_device_pointer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            device_index=3,
            size_bytes=64,
            device_ptr=1234,
            backend=backend,
        )
        registration = buffer.buffer_registration()
        response = buffer.register_with_daemon(daemon_client)

        self.assertEqual(backend.set_device_calls, [3])
        self.assertEqual(backend.export_ipc_calls, [1234])
        self.assertEqual(registration.handle_type, "cuda_ipc_device")
        self.assertEqual(registration.metadata["cuda_ipc_handle"], (b"c" * 64).hex())
        self.assertEqual(registration.address, 1234)
        self.assertTrue(response.ok)
        self.assertEqual(
            daemon_client.register_buffer_calls[0]["metadata"]["cuda_ipc_handle"],
            (b"c" * 64).hex(),
        )

    def test_allocator_rejects_empty_buffers(self) -> None:
        allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-test")

        with self.assertRaisesRegex(ValueError, "size_bytes"):
            allocator.allocate("cpu-buffer", "job-1", 0)


if __name__ == "__main__":
    unittest.main()
