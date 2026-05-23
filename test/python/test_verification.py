from __future__ import annotations

import unittest

from turbobus.verification import _build_verification_daemon


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
