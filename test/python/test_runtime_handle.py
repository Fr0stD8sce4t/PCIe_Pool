from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time

from turbobus import runtime as runtime_module
from turbobus.daemon.protocol import DaemonResponse
from turbobus.plan_trace import transfer_plan_to_dict as plan_trace_to_dict
from turbobus.runtime import (
    AutoTransferSelector,
    RuntimeOptions,
    TransferHandle,
    TransferMode,
    transfer_plan_to_dict as runtime_plan_to_dict,
)
from turbobus.transfer import TransferRequest

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for validation tests
    torch = None


class NativeHandle:
    def __init__(self, handle_id: int = 1) -> None:
        self.id = handle_id


class NativeStats:
    bytes = 64
    direct_chunks = 1
    gib_per_second = 7.5


class FakeDevice:
    def __init__(self, type_: str, index: int | None = None) -> None:
        self.type = type_
        self.index = index


class FakeTensor:
    def __init__(
        self,
        ptr: int,
        bytes_: int,
        *,
        device_type: str,
        device_index: int | None = None,
        pinned: bool = False,
    ) -> None:
        self._ptr = ptr
        self._bytes = bytes_
        self.device = FakeDevice(device_type, device_index)
        self._pinned = pinned

    def data_ptr(self) -> int:
        return self._ptr

    def numel(self) -> int:
        return self._bytes

    def element_size(self) -> int:
        return 1

    def is_pinned(self) -> bool:
        return self._pinned

    def is_contiguous(self) -> bool:
        return True


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

    def test_auto_transfer_selector_avoids_pool_below_default_min_bytes(self) -> None:
        class Relay:
            relay_device = 5
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        selector = AutoTransferSelector()
        decision = selector.choose(
            Profile(),
            request_bytes=11 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
            request_chunks=56,
        )

        self.assertEqual(decision.resolved_mode, TransferMode.DIRECT)

    def test_auto_transfer_selector_uses_pool_above_default_min_bytes(self) -> None:
        class Relay:
            relay_device = 5
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        selector = AutoTransferSelector()
        decision = selector.choose(
            Profile(),
            request_bytes=14 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
            request_chunks=56,
        )

        self.assertEqual(decision.resolved_mode, TransferMode.POOL)

    def test_auto_transfer_selector_uses_direction_specific_d2h_bandwidth(self) -> None:
        class Relay:
            relay_device = 5
            effective_bw_gbps = 100.0
            effective_d2h_bw_gbps = 4.0
            p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 100.0
            direct_d2h_bw_gbps = 12.0
            relays = [Relay()]

        selector = AutoTransferSelector(min_relay_speedup=1.05)
        decision = selector.choose(
            Profile(),
            request_bytes=8 * 1024 * 1024,
            chunk_bytes=1024 * 1024,
            request_chunks=8,
            direction="d2h",
        )

        self.assertEqual(decision.resolved_mode, TransferMode.DIRECT)
        self.assertEqual(decision.direct_h2d_bw_gbps, 12.0)
        self.assertEqual(decision.relay_effective_bw_gbps, 4.0)

    def test_auto_transfer_selector_falls_back_to_h2d_profile_for_old_d2h_cache(self) -> None:
        class Relay:
            relay_device = 5
            effective_bw_gbps = 8.0
            effective_d2h_bw_gbps = 0.0
            p2p_enabled = True

        class Profile:
            direct_h2d_bw_gbps = 7.5
            direct_d2h_bw_gbps = 0.0
            relays = [Relay()]

        selector = AutoTransferSelector()
        decision = selector.choose(
            Profile(),
            request_bytes=32 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
            request_chunks=8,
            direction="d2h",
        )

        self.assertEqual(decision.resolved_mode, TransferMode.POOL)
        self.assertEqual(decision.direct_h2d_bw_gbps, 7.5)
        self.assertEqual(decision.relay_effective_bw_gbps, 8.0)

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

    def test_auto_transfer_mode_profiles_when_cached_relays_are_missing(self) -> None:
        class Relay:
            relay_device = 1
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class EmptyRelayProfile:
            direct_h2d_bw_gbps = 7.5
            relays = []

        class RelayProfile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        class FakeRuntime:
            def __init__(self) -> None:
                self.mode = None
                self.profile_calls = 0
                self.profile_force = None

            def profile(self, bytes: int, force: bool = False):
                self.profile_calls += 1
                self.profile_force = force
                return RelayProfile()

            def cached_profile(self):
                return RelayProfile() if self.profile_calls else EmptyRelayProfile()

            def planner_profile(self):
                return self.cached_profile()

            def set_transfer_mode(self, mode):
                self.mode = mode

        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
        )
        runtime._runtime = FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO

        decision = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(runtime._runtime.profile_calls, 1)
        self.assertTrue(runtime._runtime.profile_force)
        self.assertEqual(decision.resolved_mode, TransferMode.POOL)
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.POOL)
        self.assertEqual(runtime.last_auto_decision_dict()["auto_resolved_mode"], "pool")
        self.assertEqual(runtime.last_auto_decision_dict()["auto_eligible_relays"], "1")

    def test_auto_transfer_mode_reuses_native_pool_mode_without_redundant_switches(self) -> None:
        class Relay:
            relay_device = 1
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class RelayProfile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        class FakeRuntime:
            def __init__(self) -> None:
                self.modes = []
                self.profile_obj = RelayProfile()

            def profile(self, bytes: int, force: bool = False):
                return self.profile_obj

            def cached_profile(self):
                return self.profile_obj

            def planner_profile(self):
                return self.profile_obj

            def set_transfer_mode(self, mode):
                self.modes.append(mode)

        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
        )
        runtime._runtime = FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO
        runtime._last_native_transfer_mode = TransferMode.POOL
        runtime._last_auto_decision = None
        runtime._forced_transfer_mode = None

        first = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")
        second = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(first.resolved_mode, TransferMode.POOL)
        self.assertEqual(second.resolved_mode, TransferMode.POOL)
        self.assertEqual(runtime._runtime.modes, [])
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.POOL)

    def test_auto_transfer_mode_reuses_cached_decision_for_same_request_shape(self) -> None:
        class Relay:
            relay_device = 1
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class RelayProfile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        class FakeRuntime:
            def __init__(self) -> None:
                self.modes = []
                self.profile_obj = RelayProfile()

            def profile(self, bytes: int, force: bool = False):
                return self.profile_obj

            def cached_profile(self):
                return self.profile_obj

            def planner_profile(self):
                return self.profile_obj

            def set_transfer_mode(self, mode):
                self.modes.append(mode)

        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
        )
        runtime._runtime = FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO
        runtime._last_native_transfer_mode = TransferMode.POOL
        runtime._last_auto_decision = None
        runtime._last_auto_decision_profile = None
        runtime._last_auto_decision_key = None
        runtime._forced_transfer_mode = None

        decision = runtime_module.AutoTransferDecision(
            requested_mode=TransferMode.AUTO,
            resolved_mode=TransferMode.POOL,
            request_bytes=32 * 1024 * 1024,
            request_chunks=8,
            direct_h2d_bw_gbps=7.5,
            relay_effective_bw_gbps=7.6,
            eligible_relay_devices=(1,),
            reason="pool speedup 1.500 >= 1.150",
        )

        with mock.patch.object(
            runtime_module.AutoTransferSelector,
            "choose",
            autospec=True,
            return_value=decision,
        ) as choose:
            first = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")
            second = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(choose.call_count, 1)
        self.assertIs(first, decision)
        self.assertIs(second, decision)
        self.assertEqual(runtime._runtime.modes, [])
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.POOL)

    def test_auto_transfer_mode_stays_direct_when_relays_remain_missing(self) -> None:
        class EmptyRelayProfile:
            direct_h2d_bw_gbps = 7.5
            relays = []

        class FakeRuntime:
            def __init__(self) -> None:
                self.mode = None
                self.profile_calls = 0
                self.profile_force = None

            def profile(self, bytes: int, force: bool = False):
                self.profile_calls += 1
                self.profile_force = force
                return EmptyRelayProfile()

            def cached_profile(self):
                return EmptyRelayProfile()

            def planner_profile(self):
                return EmptyRelayProfile()

            def set_transfer_mode(self, mode):
                self.mode = mode

        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
        )
        runtime._runtime = FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO
        runtime._last_auto_decision = None
        runtime._forced_transfer_mode = None

        decision = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(runtime._runtime.profile_calls, 1)
        self.assertTrue(runtime._runtime.profile_force)
        self.assertEqual(decision.resolved_mode, TransferMode.DIRECT)
        self.assertEqual(decision.reason, "h2d has no eligible relay paths")
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.DIRECT)
        self.assertEqual(runtime.last_auto_decision_dict()["auto_resolved_mode"], "direct")

    def test_batch_transfer_mode_keeps_outer_auto_decision(self) -> None:
        class Relay:
            relay_device = 1
            effective_bw_gbps = 7.6
            p2p_enabled = True

        class RelayProfile:
            direct_h2d_bw_gbps = 7.5
            relays = [Relay()]

        class FakeRuntime:
            def profile(self, bytes: int, force: bool = False):
                return RelayProfile()

            def cached_profile(self):
                return RelayProfile()

            def planner_profile(self):
                return RelayProfile()

            def set_transfer_mode(self, mode):
                self.mode = mode

        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
        )
        runtime._runtime = FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO
        runtime._last_auto_decision = None
        runtime._forced_transfer_mode = None

        with runtime.batch_transfer_mode(32 * 1024 * 1024, "h2d", 56):
            inner = runtime.resolve_transfer_mode(1024 * 1024, "h2d", 2)

        self.assertEqual(inner.resolved_mode, TransferMode.POOL)
        self.assertEqual(runtime.last_auto_decision_dict()["auto_request_bytes"], 32 * 1024 * 1024)
        self.assertEqual(runtime.last_auto_decision_dict()["auto_request_chunks"], 56)

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


class RuntimeBackendFacadeTest(unittest.TestCase):
    class FakeNativeRuntime:
        def __init__(self) -> None:
            self.init_calls = []

        def init(self, target_gpu, relay_gpus) -> None:
            self.init_calls.append((target_gpu, list(relay_gpus)))

    class FakeBackend:
        def __init__(self) -> None:
            self.bind_calls = []
            self.create_calls = []
            self.native_runtime = RuntimeBackendFacadeTest.FakeNativeRuntime()

        def bind_runtime(self, native_module, torch_module) -> None:
            self.bind_calls.append((native_module, torch_module))

        def create_runtime(self, options):
            self.create_calls.append(options)
            return self.native_runtime

    def test_runtime_creates_native_runtime_through_backend_facade(self) -> None:
        backend = self.FakeBackend()
        old_backend = runtime_module.default_cuda_backend
        runtime_module.default_cuda_backend = backend
        try:
            runtime = runtime_module.Runtime(
                target_gpu=0,
                relay_gpus=[1],
                options=RuntimeOptions(),
            )
        finally:
            runtime_module.default_cuda_backend = old_backend

        self.assertIs(runtime._backend, backend)
        self.assertEqual(backend.create_calls, [runtime.options])
        self.assertEqual(backend.native_runtime.init_calls, [(0, [1])])
        self.assertEqual(len(backend.bind_calls), 1)


class RuntimeDaemonReservationTest(unittest.TestCase):
    class Relay:
        relay_device = 1
        effective_bw_gbps = 7.6
        p2p_enabled = True

    class RelayProfile:
        direct_h2d_bw_gbps = 7.5
        relays = []

    def setUp(self) -> None:
        self.RelayProfile.relays = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.mode = None
            self.wait_calls = 0
            self.fetch_calls = []
            self.offload_calls = []

        def profile(self, bytes: int, force: bool = False):
            return RuntimeDaemonReservationTest.RelayProfile()

        def cached_profile(self):
            return RuntimeDaemonReservationTest.RelayProfile()

        def planner_profile(self):
            return RuntimeDaemonReservationTest.RelayProfile()

        def set_transfer_mode(self, mode):
            self.mode = mode

        def fetch_to_gpu(self, host_ptr, target_ptr, bytes_):
            self.fetch_calls.append((host_ptr, target_ptr, bytes_))
            return NativeHandle(11)

        def offload_to_cpu(self, target_ptr, host_ptr, bytes_):
            self.offload_calls.append((target_ptr, host_ptr, bytes_))
            return NativeHandle(12)

        def wait(self, handle):
            self.wait_calls += 1

        def stats(self, handle):
            return NativeStats()

    class FakeDaemonClient:
        def __init__(self, deny: bool = False) -> None:
            self.deny = deny
            self.register_calls = []
            self.close_calls = []
            self.reserve_calls = []
            self.release_calls = []

        def register_session(self, target_gpu, relay_gpus, max_inflight_chunks=8):
            self.register_calls.append(
                {
                    "target_gpu": target_gpu,
                    "relay_gpus": relay_gpus,
                    "max_inflight_chunks": max_inflight_chunks,
                }
            )
            return DaemonResponse(
                ok=True,
                payload={"session": {"session_id": "session-1"}},
            )

        def close_session(self, session_id):
            self.close_calls.append(session_id)
            return DaemonResponse(ok=True, payload={"session_id": session_id})

        def reserve_transfer(self, session_id, relay_gpu, chunks, bytes_=0, direction="unknown"):
            self.reserve_calls.append(
                {
                    "session_id": session_id,
                    "relay_gpu": relay_gpu,
                    "chunks": chunks,
                    "bytes": bytes_,
                    "direction": direction,
                }
            )
            if self.deny:
                return DaemonResponse(ok=False, error="relay chunk quota is unavailable")
            reservation_id = f"res-{len(self.reserve_calls)}"
            return DaemonResponse(
                ok=True,
                payload={"reservation": {"reservation_id": reservation_id}},
            )

        def release_transfer(self, reservation_id):
            self.release_calls.append(reservation_id)
            return DaemonResponse(ok=True, payload={"reservation_id": reservation_id})

    class PlanningDaemonClient(FakeDaemonClient):
        def __init__(self, plan_response: DaemonResponse) -> None:
            super().__init__()
            self.plan_response = plan_response
            self.plan_calls = []

        def plan_transfer_request(self, session_id, request, mode=None):
            payload = request.daemon_payload(mode=mode)
            self.plan_calls.append(
                {
                    "session_id": session_id,
                    "total_bytes": payload["total_bytes"],
                    "chunk_bytes": payload["chunk_bytes"],
                    "mode": payload["mode"],
                    "direction": payload["direction"],
                    "request_chunks": payload["request_chunks"],
                    "job_id": payload.get("job_id"),
                    "ranges": payload.get("ranges", []),
                    "api": "request",
                }
            )
            return self.plan_response

        def plan_transfer(
            self,
            session_id,
            total_bytes,
            chunk_bytes,
            mode="pool",
            direction="unknown",
            job_id=None,
        ):
            self.plan_calls.append(
                {
                    "session_id": session_id,
                    "total_bytes": total_bytes,
                    "chunk_bytes": chunk_bytes,
                    "mode": mode,
                    "direction": direction,
                    "job_id": job_id,
                    "api": "legacy",
                }
            )
            return self.plan_response

    def make_runtime(self, client):
        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
            daemon_socket_path="/tmp/turbobusd-test.sock",
        )
        runtime._daemon_client = client
        runtime._daemon_session_id = "session-1"
        runtime._runtime = self.FakeRuntime()
        runtime._last_resolved_transfer_mode = TransferMode.AUTO
        runtime._last_auto_decision = None
        runtime._forced_transfer_mode = None
        runtime._last_daemon_reservation = {}
        runtime._backend = runtime_module.default_cuda_backend
        runtime._pending_daemon_plan = None
        return runtime

    def test_daemon_reservation_granted_for_pool_transfer(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.FakeDaemonClient()
        runtime = self.make_runtime(client)

        reservations = runtime._resolve_transfer_with_daemon(
            32 * 1024 * 1024,
            direction="h2d",
            range_count=8,
        )

        self.assertEqual(reservations, ["res-1"])
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.POOL)
        self.assertEqual(client.reserve_calls[0]["session_id"], "session-1")
        self.assertEqual(client.reserve_calls[0]["relay_gpu"], 1)
        self.assertEqual(client.reserve_calls[0]["chunks"], 4)
        self.assertEqual(client.reserve_calls[0]["direction"], "h2d")
        info = runtime.last_daemon_reservation_dict()
        self.assertEqual(info["daemon_reservation_status"], "granted")
        self.assertEqual(info["daemon_reserved_relays"], "1")

    def test_daemon_plan_response_is_used_before_legacy_reserve(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.PlanningDaemonClient(
            DaemonResponse(
                ok=True,
                payload={
                    "stats": {"resolved_mode": "pool", "fallback_reason": None},
                    "leases": [
                        {
                            "lease_id": "lease-1",
                            "relay_device": 1,
                            "chunk_limit": 4,
                            "bytes_limit": 16 * 1024 * 1024,
                        }
                    ],
                    "reservations": [{"reservation_id": "lease-1"}],
                },
            )
        )
        runtime = self.make_runtime(client)

        reservations = runtime._resolve_transfer_with_daemon(
            32 * 1024 * 1024,
            direction="h2d",
            range_count=8,
        )

        self.assertEqual(reservations, ["lease-1"])
        self.assertEqual(client.reserve_calls, [])
        self.assertEqual(client.plan_calls[0]["mode"], "pool")
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.POOL)
        info = runtime.last_daemon_reservation_dict()
        self.assertEqual(info["daemon_reservation_status"], "granted")
        self.assertEqual(info["daemon_reservation_ids"], "lease-1")
        self.assertEqual(info["daemon_reserved_relays"], "1")

    def test_daemon_plan_uses_transfer_request_shape(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.PlanningDaemonClient(
            DaemonResponse(
                ok=True,
                payload={
                    "stats": {"resolved_mode": "pool", "fallback_reason": None},
                    "leases": [{"relay_device": 1, "chunk_limit": 2, "bytes_limit": 16}],
                    "reservations": [{"reservation_id": "lease-1"}],
                },
            )
        )
        runtime = self.make_runtime(client)
        request = TransferRequest(
            total_bytes=32 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
            direction="h2d",
            mode="pool",
            request_chunks=8,
            job_id="job-1",
        )

        reservations = runtime._resolve_transfer_request_with_daemon(request)

        self.assertEqual(reservations, ["lease-1"])
        self.assertEqual(client.plan_calls[0]["total_bytes"], 32 * 1024 * 1024)
        self.assertEqual(client.plan_calls[0]["chunk_bytes"], 4 * 1024 * 1024)
        self.assertEqual(client.plan_calls[0]["request_chunks"], 8)
        self.assertEqual(client.plan_calls[0]["job_id"], "job-1")
        self.assertEqual(client.plan_calls[0]["api"], "request")

    def test_fetch_uses_daemon_exact_plan_without_native_replanning(self) -> None:
        self.RelayProfile.relays = [self.Relay()]

        class FakeBackend:
            def __init__(self) -> None:
                self.plan_payloads = []
                self.fetch_plan_calls = []

            def make_transfer_plan(self, plan):
                self.plan_payloads.append(plan)
                return {"native_plan": plan}

            def fetch_plan_to_gpu(
                self,
                runtime,
                host_ptr,
                host_bytes,
                target_ptr,
                target_bytes,
                plan,
            ):
                self.fetch_plan_calls.append(
                    {
                        "runtime": runtime,
                        "host_ptr": host_ptr,
                        "host_bytes": host_bytes,
                        "target_ptr": target_ptr,
                        "target_bytes": target_bytes,
                        "plan": plan,
                    }
                )
                return NativeHandle(22)

        plan_payload = {
            "total_bytes": 32,
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
                        {"src_offset": 0, "dst_offset": 0, "bytes": 16},
                        {"src_offset": 16, "dst_offset": 16, "bytes": 16},
                    ],
                }
            ],
        }
        client = self.PlanningDaemonClient(
            DaemonResponse(
                ok=True,
                payload={
                    "plan": plan_payload,
                    "stats": {"resolved_mode": "pool", "fallback_reason": None},
                    "leases": [
                        {
                            "lease_id": "lease-1",
                            "relay_device": 1,
                            "chunk_limit": 2,
                            "bytes_limit": 32,
                        }
                    ],
                    "reservations": [{"reservation_id": "lease-1"}],
                },
            )
        )
        runtime = self.make_runtime(client)
        runtime.options.transfer_mode = TransferMode.POOL
        backend = FakeBackend()
        runtime._backend = backend
        old_torch = runtime_module.torch
        runtime_module.torch = type("Torch", (), {"Tensor": FakeTensor})
        try:
            handle = runtime.fetch_to_gpu(
                FakeTensor(100, 32, device_type="cpu", pinned=True),
                FakeTensor(200, 64, device_type="cuda", device_index=0),
            )
        finally:
            runtime_module.torch = old_torch

        self.assertEqual(handle.native.id, 22)
        self.assertEqual(runtime._runtime.fetch_calls, [])
        self.assertEqual(backend.plan_payloads, [plan_payload])
        self.assertEqual(backend.fetch_plan_calls[0]["runtime"], runtime._runtime)
        self.assertEqual(backend.fetch_plan_calls[0]["host_ptr"], 100)
        self.assertEqual(backend.fetch_plan_calls[0]["host_bytes"], 32)
        self.assertEqual(backend.fetch_plan_calls[0]["target_ptr"], 200)
        self.assertEqual(backend.fetch_plan_calls[0]["target_bytes"], 64)
        self.assertEqual(
            backend.fetch_plan_calls[0]["plan"],
            {"native_plan": plan_payload},
        )

    def test_daemon_plan_direct_fallback_updates_auto_decision(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.PlanningDaemonClient(
            DaemonResponse(
                ok=True,
                payload={
                    "stats": {
                        "resolved_mode": "direct",
                        "fallback_reason": "relay chunk quota is unavailable",
                    },
                    "leases": [],
                    "reservations": [],
                },
            )
        )
        runtime = self.make_runtime(client)

        reservations = runtime._resolve_transfer_with_daemon(
            32 * 1024 * 1024,
            direction="h2d",
            range_count=8,
        )

        self.assertEqual(reservations, [])
        self.assertEqual(client.reserve_calls, [])
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.DIRECT)
        self.assertIn(
            "daemon_reservation_denied",
            runtime.last_auto_decision_dict()["auto_reason"],
        )
        info = runtime.last_daemon_reservation_dict()
        self.assertEqual(info["daemon_reservation_status"], "denied")
        self.assertIn("relay chunk quota", info["daemon_reservation_error"])

    def test_unsupported_daemon_plan_falls_back_to_legacy_reserve(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.PlanningDaemonClient(
            DaemonResponse(ok=False, error="unsupported request: PLAN_TRANSFER")
        )
        runtime = self.make_runtime(client)

        reservations = runtime._resolve_transfer_with_daemon(
            32 * 1024 * 1024,
            direction="h2d",
            range_count=8,
        )

        self.assertEqual(reservations, ["res-1"])
        self.assertEqual(len(client.plan_calls), 1)
        self.assertEqual(len(client.reserve_calls), 1)
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.POOL)

    def test_daemon_reservation_denial_falls_back_to_direct(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.FakeDaemonClient(deny=True)
        runtime = self.make_runtime(client)

        reservations = runtime._resolve_transfer_with_daemon(
            32 * 1024 * 1024,
            direction="h2d",
            range_count=8,
        )

        self.assertEqual(reservations, [])
        self.assertEqual(runtime.last_transfer_mode(), TransferMode.DIRECT)
        self.assertEqual(runtime.last_auto_decision_dict()["auto_resolved_mode"], "direct")
        self.assertIn("daemon_reservation_denied", runtime.last_auto_decision_dict()["auto_reason"])
        info = runtime.last_daemon_reservation_dict()
        self.assertEqual(info["daemon_reservation_status"], "denied")
        self.assertIn("relay chunk quota", info["daemon_reservation_error"])

    def test_transfer_handle_wait_releases_daemon_reservation(self) -> None:
        self.RelayProfile.relays = [self.Relay()]
        client = self.FakeDaemonClient()
        runtime = self.make_runtime(client)
        reservations = runtime._resolve_transfer_with_daemon(
            32 * 1024 * 1024,
            direction="h2d",
            range_count=8,
        )
        handle = TransferHandle(runtime, NativeHandle(), reservations)

        handle.wait()

        self.assertEqual(client.release_calls, ["res-1"])
        self.assertEqual(handle._daemon_reservations, [])
        self.assertTrue(handle.done)
        self.assertEqual(handle.stats.bytes, 64)
        self.assertEqual(handle.stats.gib_per_second, 7.5)
        self.assertEqual(handle.stats.daemon_session_id, "session-1")
        self.assertEqual(handle.stats.daemon_reservation_status, "granted")
        self.assertEqual(
            handle.stats.daemon_reservation_info["daemon_reservation_status"],
            "granted",
        )
        self.assertEqual(handle.daemon_reservation_info["daemon_reservation_status"], "granted")

    def test_runtime_initializes_and_closes_daemon_session(self) -> None:
        client = self.FakeDaemonClient()
        old_client = runtime_module.TurboBusDaemonClient
        runtime_module.TurboBusDaemonClient = lambda socket_path: client
        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            daemon_socket_path="/tmp/turbobusd-test.sock",
            daemon_max_inflight_chunks=3,
        )
        runtime._daemon_client = None
        runtime._daemon_session_id = None
        try:
            runtime._init_daemon_session()
            self.assertEqual(runtime._daemon_session_id, "session-1")
            self.assertEqual(client.register_calls[0]["target_gpu"], 0)
            self.assertEqual(client.register_calls[0]["relay_gpus"], [1])
            self.assertEqual(client.register_calls[0]["max_inflight_chunks"], 3)

            runtime.close()
            self.assertEqual(client.close_calls, ["session-1"])
            self.assertIsNone(runtime._daemon_session_id)
        finally:
            runtime_module.TurboBusDaemonClient = old_client


class RuntimeDaemonProfileCacheTest(unittest.TestCase):
    class Relay:
        relay_device = 1
        target_device = 0
        h2d_bw_gbps = 7.6
        d2h_bw_gbps = 8.6
        p2p_bw_gbps = 40.0
        effective_bw_gbps = 7.6
        effective_d2h_bw_gbps = 8.6
        p2p_enabled = True

    class Profile:
        target_device = 0
        direct_h2d_bw_gbps = 7.5
        direct_d2h_bw_gbps = 8.5
        relays = []

    class EmptyProfile:
        target_device = 0
        direct_h2d_bw_gbps = 0.0
        direct_d2h_bw_gbps = 0.0
        relays = []

    class FakeNativeRuntime:
        def __init__(self) -> None:
            self.mode = None
            self.profile_calls = 0
            self.cached_profiles = []

        def profile(self, bytes: int, force: bool = False):
            self.profile_calls += 1
            return RuntimeDaemonProfileCacheTest.Profile()

        def cached_profile(self):
            if self.cached_profiles:
                return self.cached_profiles[-1]
            return RuntimeDaemonProfileCacheTest.EmptyProfile()

        def planner_profile(self):
            return self.cached_profile()

        def set_cached_profile(self, profile):
            self.cached_profiles.append(profile)

        def set_transfer_mode(self, mode):
            self.mode = mode

    class FakeDaemonClient:
        def __init__(self, entry=None) -> None:
            self.entry = entry
            self.get_calls = []
            self.put_calls = []
            self.register_calls = []

        def register_session(self, target_gpu, relay_gpus, max_inflight_chunks=8):
            self.register_calls.append((target_gpu, relay_gpus, max_inflight_chunks))
            return DaemonResponse(ok=True, payload={"session": {"session_id": "session-1"}})

        def get_profile(self, target_gpu, relay_gpus):
            self.get_calls.append((target_gpu, relay_gpus))
            return DaemonResponse(ok=True, payload={"profile": self.entry})

        def put_profile(self, target_gpu, relay_gpus, profile, profile_bytes=0):
            self.put_calls.append(
                {
                    "target_gpu": target_gpu,
                    "relay_gpus": relay_gpus,
                    "profile": profile,
                    "profile_bytes": profile_bytes,
                }
            )
            return DaemonResponse(ok=True, payload={"profile": {"profile": profile}})

    def setUp(self) -> None:
        self.Profile.relays = [self.Relay()]

    @staticmethod
    def daemon_entry(*, updated_at=None, direct_h2d=7.5):
        return {
            "target_gpu": 0,
            "relay_gpus": [1],
            "profile_bytes": 4096,
            "updated_at": time.time() if updated_at is None else updated_at,
            "profile": {
                "target_device": 0,
                "direct_h2d_bw_gbps": direct_h2d,
                "direct_d2h_bw_gbps": 8.5,
                "relays": [
                    {
                        "relay_device": 1,
                        "target_device": 0,
                        "h2d_bw_gbps": 7.6,
                        "d2h_bw_gbps": 8.6,
                        "p2p_bw_gbps": 40.0,
                        "effective_bw_gbps": 7.6,
                        "effective_d2h_bw_gbps": 8.6,
                        "p2p_enabled": True,
                    }
                ],
            },
        }

    def make_runtime(self, client):
        runtime = object.__new__(runtime_module.Runtime)
        runtime.target_gpu = 0
        runtime.relay_gpus = [1]
        runtime.options = RuntimeOptions(
            transfer_mode="auto",
            chunk_bytes=4 * 1024 * 1024,
            daemon_socket_path="/tmp/turbobusd-test.sock",
            daemon_profile_max_age_seconds=60.0,
        )
        runtime._daemon_client = client
        runtime._daemon_session_id = "session-1"
        runtime._runtime = self.FakeNativeRuntime()
        runtime._daemon_profile = None
        runtime._last_daemon_profile = {}
        runtime._last_resolved_transfer_mode = TransferMode.AUTO
        runtime._last_auto_decision = None
        runtime._forced_transfer_mode = None
        return runtime

    def test_daemon_profile_cache_hit_feeds_auto_selector_and_native_runtime(self) -> None:
        client = self.FakeDaemonClient(self.daemon_entry())
        runtime = self.make_runtime(client)

        runtime._load_daemon_profile()
        decision = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(runtime.last_daemon_profile_dict()["daemon_profile_status"], "hit")
        self.assertEqual(runtime._runtime.profile_calls, 0)
        self.assertEqual(len(runtime._runtime.cached_profiles), 1)
        self.assertEqual(decision.resolved_mode, TransferMode.POOL)
        self.assertEqual(decision.eligible_relay_devices, (1,))

    def test_daemon_profile_cache_miss_profiles_and_publishes(self) -> None:
        client = self.FakeDaemonClient(None)
        runtime = self.make_runtime(client)

        runtime._load_daemon_profile()
        decision = runtime.resolve_transfer_mode(32 * 1024 * 1024, direction="h2d")

        self.assertEqual(runtime._runtime.profile_calls, 1)
        self.assertEqual(client.put_calls[0]["profile_bytes"], runtime.options.profile_bytes)
        self.assertEqual(
            client.put_calls[0]["profile"]["direct_h2d_bw_gbps"],
            7.5,
        )
        self.assertEqual(decision.resolved_mode, TransferMode.POOL)

    def test_stale_daemon_profile_is_ignored(self) -> None:
        client = self.FakeDaemonClient(self.daemon_entry(updated_at=time.time() - 3600.0))
        runtime = self.make_runtime(client)

        runtime._load_daemon_profile()

        self.assertEqual(runtime.last_daemon_profile_dict()["daemon_profile_status"], "stale")
        self.assertIsNone(runtime._daemon_profile)
        self.assertEqual(runtime._runtime.cached_profiles, [])

    def test_invalid_daemon_profile_is_ignored(self) -> None:
        client = self.FakeDaemonClient(self.daemon_entry(direct_h2d=0.0))
        runtime = self.make_runtime(client)

        runtime._load_daemon_profile()

        self.assertEqual(runtime.last_daemon_profile_dict()["daemon_profile_status"], "invalid")
        self.assertIn("direct_h2d", runtime.last_daemon_profile_dict()["daemon_profile_error"])
        self.assertIsNone(runtime._daemon_profile)


class PlanTraceTest(unittest.TestCase):
    def test_transfer_plan_to_dict_keeps_runtime_import_path(self) -> None:
        plan = SimpleNamespace(
            total_bytes=32,
            chunk_bytes=16,
            assignments=[
                SimpleNamespace(
                    path=SimpleNamespace(
                        kind="direct",
                        direction="h2d",
                        target_device=6,
                        relay_device=-1,
                        h2d_bw_gbps=7.5,
                        d2h_bw_gbps=8.5,
                        p2p_bw_gbps=0.0,
                        effective_bw_gbps=7.5,
                        enabled=True,
                    ),
                    chunks=[
                        SimpleNamespace(src_offset=0, dst_offset=0, bytes=16),
                        SimpleNamespace(src_offset=16, dst_offset=16, bytes=16),
                    ],
                )
            ],
        )

        expected = {
            "total_bytes": 32,
            "chunk_bytes": 16,
            "assignments": [
                {
                    "path": {
                        "kind": "direct",
                        "direction": "h2d",
                        "target_device": 6,
                        "relay_device": -1,
                        "h2d_bw_gbps": 7.5,
                        "d2h_bw_gbps": 8.5,
                        "p2p_bw_gbps": 0.0,
                        "effective_bw_gbps": 7.5,
                        "enabled": True,
                    },
                    "chunks": [
                        {"src_offset": 0, "dst_offset": 0, "bytes": 16},
                        {"src_offset": 16, "dst_offset": 16, "bytes": 16},
                    ],
                    "bytes": 32,
                    "chunk_count": 2,
                }
            ],
        }

        self.assertIs(runtime_plan_to_dict, plan_trace_to_dict)
        self.assertEqual(runtime_plan_to_dict(plan), expected)


class RangeValidationTest(unittest.TestCase):
    def test_runtime_range_fields_wrapper_accepts_dicts(self) -> None:
        fields = runtime_module._range_fields(
            {"src_offset": 1, "dst_offset": 2, "bytes": 3}
        )

        self.assertEqual(fields, (1, 2, 3))

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

    def test_native_transfer_plan_preserves_daemon_assignments(self) -> None:
        class NativePlan:
            def __init__(self) -> None:
                self.total_bytes = 0
                self.chunk_bytes = 0
                self.assignments = []

        class NativeAssignment:
            def __init__(self) -> None:
                self.path = None
                self.chunks = []

        class NativePath:
            def __init__(self) -> None:
                self.kind_value = None
                self.direction_value = None
                self.target_device = -1
                self.relay_device = -1
                self.h2d_bw_gbps = 0.0
                self.d2h_bw_gbps = 0.0
                self.p2p_bw_gbps = 0.0
                self.effective_bw_gbps = 0.0
                self.enabled = False

        class NativeChunk:
            def __init__(self) -> None:
                self.src_offset = 0
                self.dst_offset = 0
                self.bytes = 0

        class PathKind:
            RelayH2DThenP2P = "relay-h2d"
            RelayP2PThenD2H = "relay-d2h"
            DirectH2D = "direct-h2d"
            DirectD2H = "direct-d2h"

        class TransferDirection:
            H2D = "h2d"
            D2H = "d2h"

        old_extension = runtime_module._runtime_engine._turbobus
        runtime_module._runtime_engine._turbobus = type(
            "Ext",
            (),
            {
                "TransferPlan": NativePlan,
                "PathAssignment": NativeAssignment,
                "Path": NativePath,
                "Chunk": NativeChunk,
                "PathKind": PathKind,
                "TransferDirection": TransferDirection,
            },
        )
        try:
            plan = runtime_module._runtime_engine._native_transfer_plan(
                {
                    "total_bytes": 32,
                    "chunk_bytes": 16,
                    "assignments": [
                        {
                            "path": {
                                "kind": "relay",
                                "direction": "h2d",
                                "target_device": 0,
                                "relay_device": 1,
                                "h2d_bw_gbps": 12.0,
                                "p2p_bw_gbps": 50.0,
                                "effective_bw_gbps": 10.0,
                                "enabled": True,
                            },
                            "chunks": [
                                {"src_offset": 0, "dst_offset": 8, "bytes": 16},
                            ],
                        }
                    ],
                }
            )
        finally:
            runtime_module._runtime_engine._turbobus = old_extension

        self.assertEqual(plan.total_bytes, 32)
        self.assertEqual(plan.chunk_bytes, 16)
        self.assertEqual(len(plan.assignments), 1)
        assignment = plan.assignments[0]
        self.assertEqual(assignment.path.kind_value, "relay-h2d")
        self.assertEqual(assignment.path.direction_value, "h2d")
        self.assertEqual(assignment.path.relay_device, 1)
        self.assertEqual(assignment.chunks[0].src_offset, 0)
        self.assertEqual(assignment.chunks[0].dst_offset, 8)
        self.assertEqual(assignment.chunks[0].bytes, 16)

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
