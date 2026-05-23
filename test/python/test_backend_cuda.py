from __future__ import annotations

import unittest

from turbobus.backends.cuda import CudaNativeBackend
from turbobus.schema import TransferMode


class FakeNativeRuntime:
    def __init__(self, options) -> None:
        self.options = options


class FakeNativeModule:
    Runtime = FakeNativeRuntime


class FakeRuntimeEngine:
    def __init__(self) -> None:
        self._turbobus = None
        self.torch = None
        self.require_extension_calls = 0
        self.require_torch_calls = 0
        self.range_calls = []

    def _require_extension(self) -> None:
        self.require_extension_calls += 1

    def _require_torch(self) -> None:
        self.require_torch_calls += 1

    def _runtime_transfer_mode_value(self, mode):
        return f"native:{TransferMode(mode).value}"

    def _native_ranges(self, ranges, source_bytes, destination_bytes):
        self.range_calls.append((list(ranges), source_bytes, destination_bytes))
        return ["native-range"]


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


if __name__ == "__main__":
    unittest.main()
