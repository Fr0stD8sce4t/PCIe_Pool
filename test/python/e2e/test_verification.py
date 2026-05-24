from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from turbobus.verification import (
    WorkerManagedH2DRelayVerificationResult,
    _build_verification_daemon,
    _cuda_environment_relay_gpu,
    _required_cuda_device_count,
    _resolve_verification_buffer_sizes,
    _worker_helper_required,
    main,
)


class WorkerManagedH2DRelayVerificationTest(unittest.TestCase):
    def test_verification_daemon_seeds_relay_plan_profile(self) -> None:
        daemon = _build_verification_daemon(
            target_gpu=0,
            relay_gpu=1,
            max_inflight_chunks=8,
            profile_bytes=4096,
        )
        session = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        self.assertTrue(session.ok)
        session_id = session.payload["session"]["session_id"]
        daemon.register_job("job-1", session_id=session_id)
        daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=4096,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata={
                "shared_memory_name": "tb-verification-test",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 4096,
            },
        )
        daemon.register_buffer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=4096,
            device_index=0,
            handle_type="cuda_ipc_device",
            metadata={"cuda_ipc_handle": (b"c" * 64).hex()},
        )

        plan = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=4096,
            chunk_bytes=4096,
            mode="relay",
            direction="h2d",
            job_id="job-1",
            buffer_ids=("cpu-buffer", "gpu-buffer"),
        )

        self.assertTrue(plan.ok)
        self.assertEqual(plan.payload["stats"]["resolved_mode"], "relay")
        self.assertEqual(len(plan.payload["lease_tokens"]), 1)
        self.assertEqual(plan.payload["lease_tokens"][0]["relay_gpu"], 1)
        self.assertEqual(
            plan.payload["plan"]["assignments"][0]["path"]["kind"],
            "relay",
        )

    def test_verification_daemon_seeds_d2h_relay_plan_profile(self) -> None:
        daemon = _build_verification_daemon(
            target_gpu=0,
            relay_gpu=1,
            max_inflight_chunks=8,
            profile_bytes=4096,
        )
        session = daemon.register_session(
            target_gpu=0,
            requested_relays=[1],
            max_inflight_chunks=8,
        )
        self.assertTrue(session.ok)
        session_id = session.payload["session"]["session_id"]
        daemon.register_job("job-1", session_id=session_id)
        daemon.register_buffer(
            buffer_id="gpu-buffer",
            job_id="job-1",
            kind="gpu",
            size_bytes=4096,
            device_index=0,
            handle_type="cuda_ipc_device",
            metadata={"cuda_ipc_handle": (b"c" * 64).hex()},
        )
        daemon.register_buffer(
            buffer_id="cpu-buffer",
            job_id="job-1",
            kind="cpu_pinned",
            size_bytes=4096,
            pinned=True,
            handle_type="shared_pinned_cpu",
            metadata={
                "shared_memory_name": "tb-verification-test",
                "offset_bytes": 0,
                "shared_memory_size_bytes": 4096,
            },
        )

        plan = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=4096,
            chunk_bytes=4096,
            mode="relay",
            direction="d2h",
            job_id="job-1",
            buffer_ids=("gpu-buffer", "cpu-buffer"),
        )

        self.assertTrue(plan.ok)
        self.assertEqual(plan.payload["stats"]["resolved_mode"], "relay")
        self.assertEqual(len(plan.payload["lease_tokens"]), 1)
        self.assertEqual(plan.payload["lease_tokens"][0]["relay_gpu"], 1)
        assignment = plan.payload["plan"]["assignments"][0]
        self.assertEqual(assignment["path"]["kind"], "relay")
        self.assertEqual(assignment["path"]["direction"], "d2h")

    def test_verification_daemon_seeds_h2d_pool_plan_profile(self) -> None:
        daemon = _build_verification_daemon(
            target_gpu=0,
            relay_gpu=1,
            max_inflight_chunks=8,
            profile_bytes=4096,
        )
        session_id = _register_verification_job_buffers(
            daemon,
            direction="h2d",
            total_bytes=4096,
        )

        plan = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=4096,
            chunk_bytes=1024,
            mode="pool",
            direction="h2d",
            job_id="job-1",
            buffer_ids=("cpu-buffer", "gpu-buffer"),
        )

        self.assertTrue(plan.ok)
        self.assertEqual(plan.payload["stats"]["resolved_mode"], "pool")
        self.assertEqual(len(plan.payload["lease_tokens"]), 1)
        self.assertEqual(
            {
                assignment["path"]["kind"]
                for assignment in plan.payload["plan"]["assignments"]
            },
            {"direct", "relay"},
        )

    def test_verification_daemon_seeds_d2h_pool_plan_profile(self) -> None:
        daemon = _build_verification_daemon(
            target_gpu=0,
            relay_gpu=1,
            max_inflight_chunks=8,
            profile_bytes=4096,
        )
        session_id = _register_verification_job_buffers(
            daemon,
            direction="d2h",
            total_bytes=4096,
        )

        plan = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=4096,
            chunk_bytes=1024,
            mode="pool",
            direction="d2h",
            job_id="job-1",
            buffer_ids=("gpu-buffer", "cpu-buffer"),
        )

        self.assertTrue(plan.ok)
        self.assertEqual(plan.payload["stats"]["resolved_mode"], "pool")
        self.assertEqual(len(plan.payload["lease_tokens"]), 1)
        assignments = plan.payload["plan"]["assignments"]
        self.assertEqual(
            {assignment["path"]["kind"] for assignment in assignments},
            {"direct", "relay"},
        )
        self.assertTrue(
            all(assignment["path"]["direction"] == "d2h" for assignment in assignments)
        )

    def test_verification_daemon_plans_offset_ranges(self) -> None:
        daemon = _build_verification_daemon(
            target_gpu=0,
            relay_gpu=1,
            max_inflight_chunks=8,
            profile_bytes=64,
        )
        session_id = _register_verification_job_buffers(
            daemon,
            direction="h2d",
            total_bytes=64,
        )

        plan = daemon.plan_transfer(
            session_id=session_id,
            total_bytes=16,
            chunk_bytes=8,
            mode="relay",
            direction="h2d",
            job_id="job-1",
            buffer_ids=("cpu-buffer", "gpu-buffer"),
            ranges=({"src_offset": 8, "dst_offset": 24, "bytes": 16},),
        )

        self.assertTrue(plan.ok)
        self.assertEqual(
            tuple(
                {
                    "src_offset": chunk["src_offset"],
                    "dst_offset": chunk["dst_offset"],
                    "bytes": chunk["bytes"],
                }
                for chunk in plan.payload["plan"]["assignments"][0]["chunks"]
            ),
            (
                {"src_offset": 8, "dst_offset": 24, "bytes": 8},
                {"src_offset": 16, "dst_offset": 32, "bytes": 8},
            ),
        )

    def test_verification_buffer_sizes_cover_offsets(self) -> None:
        self.assertEqual(
            _resolve_verification_buffer_sizes(
                bytes_to_copy=16,
                src_offset=8,
                dst_offset=24,
                source_buffer_bytes=None,
                destination_buffer_bytes=None,
            ),
            (24, 40),
        )
        with self.assertRaisesRegex(ValueError, "source_buffer_bytes"):
            _resolve_verification_buffer_sizes(
                bytes_to_copy=16,
                src_offset=8,
                dst_offset=0,
                source_buffer_bytes=23,
                destination_buffer_bytes=None,
            )
        with self.assertRaisesRegex(ValueError, "destination_buffer_bytes"):
            _resolve_verification_buffer_sizes(
                bytes_to_copy=16,
                src_offset=0,
                dst_offset=24,
                source_buffer_bytes=None,
                destination_buffer_bytes=39,
            )

    def test_worker_helper_is_not_required_for_direct_verification(self) -> None:
        self.assertFalse(_worker_helper_required("direct"))
        self.assertTrue(_worker_helper_required("relay"))
        self.assertTrue(_worker_helper_required("pool"))

    def test_direct_cuda_environment_does_not_require_relay_gpu(self) -> None:
        self.assertIsNone(_cuda_environment_relay_gpu("direct", 7))
        self.assertEqual(_cuda_environment_relay_gpu("relay", 7), 7)
        self.assertEqual(_cuda_environment_relay_gpu("pool", 7), 7)
        self.assertEqual(_required_cuda_device_count(0, None), 1)
        self.assertEqual(_required_cuda_device_count(0, 7), 8)

    def test_cli_forwards_range_offsets_to_h2d_verifier(self) -> None:
        result = WorkerManagedH2DRelayVerificationResult(
            direction="h2d",
            transfer_mode="relay",
            transfer_id="transfer-1",
            job_id="job-1",
            bytes_requested=16,
            bytes_completed=16,
            src_offset=8,
            dst_offset=24,
            source_buffer_bytes=64,
            destination_buffer_bytes=96,
            target_gpu=0,
            relay_gpu=1,
            state="complete",
            worker_final_state="complete",
            worker_path="relay_h2d",
            worker_direct_bytes=0,
            worker_direct_chunks=0,
            worker_relay_bytes=16,
            worker_relay_chunks=2,
            daemon_reservations_released=True,
            daemon_relay_active_chunks=0,
        )
        stdout = io.StringIO()
        with patch("turbobus.verification.verify_worker_managed_h2d_relay") as verifier:
            verifier.return_value = result
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--direction",
                        "h2d",
                        "--mode",
                        "relay",
                        "--bytes",
                        "16",
                        "--chunk-bytes",
                        "8",
                        "--src-offset",
                        "8",
                        "--dst-offset",
                        "24",
                        "--source-buffer-bytes",
                        "64",
                        "--destination-buffer-bytes",
                        "96",
                    ]
                )

        self.assertEqual(exit_code, 0)
        verifier.assert_called_once()
        self.assertEqual(verifier.call_args.kwargs["src_offset"], 8)
        self.assertEqual(verifier.call_args.kwargs["dst_offset"], 24)
        self.assertEqual(verifier.call_args.kwargs["source_buffer_bytes"], 64)
        self.assertEqual(verifier.call_args.kwargs["destination_buffer_bytes"], 96)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["src_offset"], 8)
        self.assertEqual(payload["dst_offset"], 24)


def _register_verification_job_buffers(
    daemon,
    *,
    direction: str,
    total_bytes: int,
) -> str:
    session = daemon.register_session(
        target_gpu=0,
        requested_relays=[1],
        max_inflight_chunks=8,
    )
    assert session.ok
    session_id = session.payload["session"]["session_id"]
    daemon.register_job("job-1", session_id=session_id)
    if direction == "h2d":
        _register_cpu_buffer(daemon, total_bytes)
        _register_gpu_buffer(daemon, total_bytes)
    else:
        _register_gpu_buffer(daemon, total_bytes)
        _register_cpu_buffer(daemon, total_bytes)
    return session_id


def _register_cpu_buffer(daemon, total_bytes: int) -> None:
    daemon.register_buffer(
        buffer_id="cpu-buffer",
        job_id="job-1",
        kind="cpu_pinned",
        size_bytes=total_bytes,
        pinned=True,
        handle_type="shared_pinned_cpu",
        metadata={
            "shared_memory_name": "tb-verification-test",
            "offset_bytes": 0,
            "shared_memory_size_bytes": total_bytes,
        },
    )


def _register_gpu_buffer(daemon, total_bytes: int) -> None:
    daemon.register_buffer(
        buffer_id="gpu-buffer",
        job_id="job-1",
        kind="gpu",
        size_bytes=total_bytes,
        device_index=0,
        handle_type="cuda_ipc_device",
        metadata={"cuda_ipc_handle": (b"c" * 64).hex()},
    )


if __name__ == "__main__":
    unittest.main()
