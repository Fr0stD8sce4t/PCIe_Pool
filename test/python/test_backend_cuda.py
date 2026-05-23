from __future__ import annotations

import unittest

from turbobus.backends.cuda import CudaNativeBackend
from turbobus.schema import TransferMode


class FakeNativeRuntime:
    def __init__(self, options) -> None:
        self.options = options


class FakeNativeModule:
    Runtime = FakeNativeRuntime


class FakeHostRegisterNativeModule:
    def __init__(self) -> None:
        self.register_host_memory_calls = []
        self.unregister_host_memory_calls = []

    def register_host_memory(self, host_ptr, bytes_):
        self.register_host_memory_calls.append((host_ptr, bytes_))

    def unregister_host_memory(self, host_ptr):
        self.unregister_host_memory_calls.append(host_ptr)


class FakeRuntimeEngine:
    def __init__(self) -> None:
        self._turbobus = None
        self.torch = None
        self.require_extension_calls = 0
        self.require_torch_calls = 0
        self.range_calls = []
        self.plan_calls = []

    def _require_extension(self) -> None:
        self.require_extension_calls += 1

    def _require_torch(self) -> None:
        self.require_torch_calls += 1

    def _runtime_transfer_mode_value(self, mode):
        return f"native:{TransferMode(mode).value}"

    def _native_ranges(self, ranges, source_bytes, destination_bytes):
        self.range_calls.append((list(ranges), source_bytes, destination_bytes))
        return ["native-range"]

    def _native_transfer_plan(self, plan):
        self.plan_calls.append(plan)
        return "native-plan"


class FakeExactPlanRuntime:
    def __init__(self) -> None:
        self.fetch_plan_calls = []
        self.offload_plan_calls = []

    def fetch_plan_to_gpu(
        self,
        host_ptr,
        host_bytes,
        target_ptr,
        target_bytes,
        plan,
    ):
        self.fetch_plan_calls.append(
            (host_ptr, host_bytes, target_ptr, target_bytes, plan)
        )
        return "fetch-handle"

    def offload_plan_to_cpu(
        self,
        target_ptr,
        target_bytes,
        host_ptr,
        host_bytes,
        plan,
    ):
        self.offload_plan_calls.append(
            (target_ptr, target_bytes, host_ptr, host_bytes, plan)
        )
        return "offload-handle"


class FakeOptions:
    def __init__(self) -> None:
        self.to_native_calls = 0

    def to_native(self):
        self.to_native_calls += 1
        return "native-options"


class CudaNativeBackendTest(unittest.TestCase):
    def test_backend_binds_runtime_engine_modules(self) -> None:
        engine = FakeRuntimeEngine()
        backend = CudaNativeBackend(engine)
        torch_module = object()

        backend.bind_runtime(FakeNativeModule, torch_module)

        self.assertIs(engine._turbobus, FakeNativeModule)
        self.assertIs(engine.torch, torch_module)

    def test_backend_delegates_native_helpers(self) -> None:
        engine = FakeRuntimeEngine()
        backend = CudaNativeBackend(engine)

        self.assertEqual(backend.transfer_mode_value(TransferMode.POOL), "native:pool")
        self.assertEqual(
            backend.make_ranges([(0, 0, 16)], source_bytes=32, destination_bytes=32),
            ["native-range"],
        )
        backend.require_torch()

        self.assertEqual(engine.range_calls, [([(0, 0, 16)], 32, 32)])
        self.assertEqual(engine.require_torch_calls, 1)

    def test_backend_creates_native_runtime_from_options(self) -> None:
        engine = FakeRuntimeEngine()
        engine._turbobus = FakeNativeModule
        backend = CudaNativeBackend(engine)
        options = FakeOptions()

        runtime = backend.create_runtime(options)

        self.assertIsInstance(runtime, FakeNativeRuntime)
        self.assertEqual(runtime.options, "native-options")
        self.assertEqual(options.to_native_calls, 1)
        self.assertEqual(engine.require_extension_calls, 1)

    def test_backend_converts_and_submits_exact_transfer_plans(self) -> None:
        engine = FakeRuntimeEngine()
        backend = CudaNativeBackend(engine)
        runtime = FakeExactPlanRuntime()

        plan_payload = {
            "total_bytes": 16,
            "chunk_bytes": 16,
            "assignments": [
                {
                    "path": {
                        "kind": "direct",
                        "direction": "h2d",
                        "target_device": 0,
                        "relay_device": -1,
                    },
                    "chunks": [{"src_offset": 0, "dst_offset": 0, "bytes": 16}],
                }
            ],
        }
        plan = backend.make_transfer_plan(plan_payload)
        fetch_handle = backend.fetch_plan_to_gpu(
            runtime,
            host_ptr=100,
            host_bytes=16,
            target_ptr=200,
            target_bytes=32,
            plan=plan,
        )
        offload_handle = backend.offload_plan_to_cpu(
            runtime,
            target_ptr=200,
            target_bytes=32,
            host_ptr=100,
            host_bytes=16,
            plan=plan,
        )

        self.assertEqual(plan, "native-plan")
        self.assertEqual(engine.plan_calls, [plan_payload])
        self.assertEqual(fetch_handle, "fetch-handle")
        self.assertEqual(runtime.fetch_plan_calls, [(100, 16, 200, 32, "native-plan")])
        self.assertEqual(offload_handle, "offload-handle")
        self.assertEqual(
            runtime.offload_plan_calls,
            [(200, 32, 100, 16, "native-plan")],
        )

    def test_backend_rejects_missing_exact_plan_submitter(self) -> None:
        backend = CudaNativeBackend(FakeRuntimeEngine())

        with self.assertRaisesRegex(RuntimeError, "exact transfer plans"):
            backend.fetch_plan_to_gpu(
                runtime=object(),
                host_ptr=100,
                host_bytes=16,
                target_ptr=200,
                target_bytes=32,
                plan="native-plan",
            )

    def test_backend_registers_host_memory_through_native_runtime(self) -> None:
        engine = FakeRuntimeEngine()
        native = FakeHostRegisterNativeModule()
        engine._turbobus = native
        backend = CudaNativeBackend(engine)

        backend.register_host_memory(100, 4096)
        backend.unregister_host_memory(100)

        self.assertEqual(native.register_host_memory_calls, [(100, 4096)])
        self.assertEqual(native.unregister_host_memory_calls, [100])
        self.assertEqual(engine.require_extension_calls, 2)

    def test_backend_rejects_missing_host_memory_registration(self) -> None:
        engine = FakeRuntimeEngine()
        engine._turbobus = object()
        backend = CudaNativeBackend(engine)

        with self.assertRaisesRegex(RuntimeError, "host memory registration"):
            backend.register_host_memory(100, 4096)


if __name__ == "__main__":
    unittest.main()
