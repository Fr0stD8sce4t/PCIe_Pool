from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from turbobus import runtime as runtime_module
from turbobus.runtime import (
    AutoTransferSelector,
    RuntimeOptions,
    TransferHandle,
    TransferMode,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for validation tests
    torch = None


class NativeHandle:
    def __init__(self, handle_id: int = 1) -> None:
        self.id = handle_id


class SuccessfulRuntime:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait(self, handle: TransferHandle) -> None:
        self.wait_calls += 1


class FailingRuntime:
    def wait(self, handle: TransferHandle) -> None:
        raise RuntimeError("simulated wait failure")


class TransferHandleTest(unittest.TestCase):
    def test_wait_marks_complete(self) -> None:
        runtime = SuccessfulRuntime()
        handle = TransferHandle(runtime, NativeHandle())

        self.assertEqual(handle.status, "submitted")
        self.assertFalse(handle.done)

        handle.wait()

        self.assertEqual(handle.status, "complete")
        self.assertTrue(handle.done)
        self.assertEqual(runtime.wait_calls, 1)

        handle.wait()
        self.assertEqual(runtime.wait_calls, 1)

    def test_wait_failure_marks_failed(self) -> None:
        handle = TransferHandle(FailingRuntime(), NativeHandle())

        with self.assertRaises(RuntimeError):
            handle.wait()

        self.assertEqual(handle.status, "failed")
        self.assertEqual(handle.error, "simulated wait failure")


class RuntimeOptionsTest(unittest.TestCase):
    def test_transfer_mode_accepts_string_values(self) -> None:
        options = RuntimeOptions(
            transfer_mode="direct",
            min_chunks_for_relay=3,
            min_pool_bytes=32 * 1024 * 1024,
            relay_min_effective_bw_gbps=6.5,
            relay_min_direct_ratio=0.8,
            enable_dynamic_weights=True,
            dynamic_weight_alpha=0.4,
        )

        self.assertEqual(options.transfer_mode, "direct")
        self.assertEqual(TransferMode(options.transfer_mode), TransferMode.DIRECT)
        self.assertEqual(options.min_chunks_for_relay, 3)
        self.assertEqual(options.min_pool_bytes, 32 * 1024 * 1024)
        self.assertEqual(options.relay_min_effective_bw_gbps, 6.5)
        self.assertEqual(options.relay_min_direct_ratio, 0.8)
        self.assertTrue(options.enable_dynamic_weights)
        self.assertEqual(options.dynamic_weight_alpha, 0.4)

    def test_auto_transfer_selector_prefers_pool_for_large_requests(self) -> None:
        class Relay:
            def __init__(self, relay_device: int, effective_bw_gbps: float) -> None:
                self.relay_device = relay_device
                self.effective_bw_gbps = effective_bw_gbps
                self.p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay(5, 7.6)]

        selector = AutoTransferSelector(
            min_chunks_for_relay=2,
            relay_min_effective_bw_gbps=6.0,
            relay_min_direct_ratio=0.8,
        )
        decision = selector.choose(Profile(), request_bytes=256 * 1024 * 1024, chunk_bytes=4 * 1024 * 1024)

        self.assertEqual(decision.requested_mode, TransferMode.AUTO)
        self.assertEqual(decision.resolved_mode, TransferMode.POOL)
        self.assertIn(5, decision.eligible_relay_devices)
        self.assertGreaterEqual(decision.request_chunks, 2)

    def test_auto_transfer_selector_falls_back_for_small_requests(self) -> None:
        class Profile:
            direct_h2d_bw_gbps = 7.5
            relays = []

        selector = AutoTransferSelector(min_chunks_for_relay=4)
        decision = selector.choose(Profile(), request_bytes=8 * 1024 * 1024, chunk_bytes=8 * 1024 * 1024)

        self.assertEqual(decision.resolved_mode, TransferMode.DIRECT)
        self.assertEqual(decision.reason, "h2d request has only 1 chunk(s)")

    def test_auto_transfer_selector_avoids_pool_for_small_kv_requests(self) -> None:
        class Relay:
            relay_device = 5
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        selector = AutoTransferSelector(min_pool_bytes=16 * 1024 * 1024)
        decision = selector.choose(
            Profile(),
            request_bytes=14 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
            request_chunks=56,
        )

        self.assertEqual(decision.resolved_mode, TransferMode.DIRECT)

    def test_auto_transfer_selector_uses_pool_after_min_bytes(self) -> None:
        class Relay:
            relay_device = 5
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        selector = AutoTransferSelector(min_pool_bytes=16 * 1024 * 1024)
        decision = selector.choose(
            Profile(),
            request_bytes=28 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
            request_chunks=56,
        )

        self.assertEqual(decision.resolved_mode, TransferMode.POOL)

    def test_auto_transfer_mode_uses_explicit_profile_fallback(self) -> None:
        class FakeProfile:
            direct_h2d_bw_gbps = 0.0
            relays = []

        class FakeRuntime:
            def __init__(self) -> None:
                self.mode = None
                self.profile_calls = 0

            def profile(self, bytes: int, force: bool = False):
                self.profile_calls += 1
                return FakeProfile()

            def cached_profile(self):
                return FakeProfile()

            def planner_profile(self):
                return FakeProfile()

            def set_transfer_mode(self, mode):
                self.mode = mode

        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(transfer_mode="auto")
        runtime._runtime = FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO

        decision = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(runtime._runtime.profile_calls, 1)
        self.assertEqual(decision.resolved_mode, TransferMode.DIRECT)
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.DIRECT)

    def test_from_tuning_json_reads_best_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tune.json"
            path.write_text(
                '{"best": {"chunk_bytes": 4194304, "staging_slots": 3}}',
                encoding="utf-8",
            )

            options = RuntimeOptions.from_tuning_json(path)

        self.assertEqual(options.chunk_bytes, 4194304)
        self.assertEqual(options.staging_slots, 3)

    def test_from_profile_json_reads_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            path.write_text(
                '{"config": {"chunk_bytes": 8388608, "profile_bytes": 16777216}}',
                encoding="utf-8",
            )

            options = RuntimeOptions.from_profile_json(path)

        self.assertEqual(options.chunk_bytes, 8388608)
        self.assertEqual(options.profile_bytes, 16777216)
        self.assertEqual(options.staging_slots, 2)


class RangeValidationTest(unittest.TestCase):
    def test_native_ranges_accepts_dicts_and_tuples(self) -> None:
        class NativeRange:
            def __init__(self) -> None:
                self.src_offset = 0
                self.dst_offset = 0
                self.bytes = 0

        old_extension = runtime_module._turbobus
        runtime_module._turbobus = type("Ext", (), {"TransferRange": NativeRange})
        try:
            ranges = runtime_module._native_ranges(
                [
                    {"src_offset": 0, "dst_offset": 16, "bytes": 8},
                    (32, 64, 8),
                ],
                source_bytes=128,
                destination_bytes=128,
            )
        finally:
            runtime_module._turbobus = old_extension

        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0].src_offset, 0)
        self.assertEqual(ranges[0].dst_offset, 16)
        self.assertEqual(ranges[1].src_offset, 32)
        self.assertEqual(ranges[1].dst_offset, 64)

    def test_native_ranges_rejects_out_of_bounds(self) -> None:
        class NativeRange:
            pass

        old_extension = runtime_module._turbobus
        runtime_module._turbobus = type("Ext", (), {"TransferRange": NativeRange})
        try:
            with self.assertRaises(ValueError):
                runtime_module._native_ranges(
                    [(120, 0, 16)],
                    source_bytes=128,
                    destination_bytes=128,
                )
        finally:
            runtime_module._turbobus = old_extension

    def test_range_tensor_validation_does_not_require_equal_sizes_for_d2h(self) -> None:
        class TensorType:
            pass

        class FakeDevice:
            def __init__(self, type_: str, index: int | None = None) -> None:
                self.type = type_
                self.index = index

        class FakeTensor(TensorType):
            def __init__(
                self,
                numel: int,
                *,
                device_type: str,
                device_index: int | None = None,
                pinned: bool = False,
            ) -> None:
                self._numel = numel
                self.device = FakeDevice(device_type, device_index)
                self._pinned = pinned

            def numel(self) -> int:
                return self._numel

            def element_size(self) -> int:
                return 1

            def is_pinned(self) -> bool:
                return self._pinned

            def is_contiguous(self) -> bool:
                return True

        old_torch = runtime_module.torch
        runtime_module.torch = type("Torch", (), {"Tensor": TensorType})
        try:
            cpu = FakeTensor(128, device_type="cpu", pinned=True)
            gpu = FakeTensor(1024, device_type="cuda", device_index=6)

            source_bytes, destination_bytes = runtime_module._validate_range_tensors(
                cpu,
                gpu,
                target_gpu=6,
                direction="d2h",
            )
        finally:
            runtime_module.torch = old_torch

        self.assertEqual(source_bytes, 1024)
        self.assertEqual(destination_bytes, 128)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DummyComputeValidationTest(unittest.TestCase):
    def make_runtime(self):
        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime._runtime = None
        return runtime

    def test_run_dummy_compute_requires_cuda_tensor(self) -> None:
        runtime = self.make_runtime()
        tensor = torch.zeros(8, dtype=torch.float32)

        with self.assertRaises(ValueError):
            runtime.run_dummy_compute(tensor, 1)

    def test_run_dummy_compute_requires_float32(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        runtime = self.make_runtime()
        runtime.target_gpu = torch.cuda.current_device()
        tensor = torch.zeros(8, dtype=torch.float16, device="cuda")

        with self.assertRaises(ValueError):
            runtime.run_dummy_compute(tensor, 1)


if __name__ == "__main__":
    unittest.main()
