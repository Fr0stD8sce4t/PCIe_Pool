from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from .backends.cuda import default_cuda_backend
from .client import CudaIpcDeviceBuffer, SharedPinnedCpuBufferAllocator
from .client_transfer import make_worker_managed_transfer_client
from .daemon import TurboBusDaemonClient
from .daemon.server import TurboBusDaemon
from .daemon.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
    StaticTopologyProvider,
)
from .worker import WorkerServiceSocketClient, run_worker_helper_process


@dataclass(frozen=True)
class WorkerManagedH2DRelayVerificationResult:
    transfer_id: str
    job_id: str
    bytes_requested: int
    bytes_completed: int
    target_gpu: int
    relay_gpu: int
    state: str
    worker_final_state: str | None
    worker_relay_bytes: int
    worker_relay_chunks: int
    daemon_reservations_released: bool
    daemon_relay_active_chunks: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_worker_managed_h2d_relay(
    *,
    target_gpu: int = 0,
    relay_gpu: int = 1,
    bytes_to_copy: int = 1024 * 1024,
    chunk_bytes: int = 1024 * 1024,
    max_inflight_chunks: int = 8,
    socket_dir: str | None = None,
    startup_timeout_seconds: float = 10.0,
) -> WorkerManagedH2DRelayVerificationResult:
    """Run the first real helper-socket H2D relay verification.

    The verifier starts a daemon socket and a worker helper process, then moves
    bytes from TurboBus shared CPU memory through a daemon-approved relay plan
    into a CUDA IPC target tensor.
    """

    _require_unix_sockets()
    target = int(target_gpu)
    relay = int(relay_gpu)
    total_bytes = int(bytes_to_copy)
    chunk_size = int(chunk_bytes)
    if total_bytes <= 0:
        raise ValueError("bytes_to_copy must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_bytes must be positive")
    if int(max_inflight_chunks) <= 0:
        raise ValueError("max_inflight_chunks must be positive")

    torch = _require_cuda_environment(target, relay)
    pattern = _make_pattern(total_bytes)
    job_id = f"verify-worker-h2d-{os.getpid()}-{time.time_ns()}"
    allocator = SharedPinnedCpuBufferAllocator(name_prefix="tb-worker-h2d-verify")

    with tempfile.TemporaryDirectory(dir=socket_dir) as tmpdir:
        daemon_socket = os.path.join(tmpdir, "turbobusd.sock")
        worker_socket = os.path.join(tmpdir, "turbobus-worker.sock")
        process_context = multiprocessing.get_context("spawn")
        daemon_process = process_context.Process(
            target=_serve_verification_daemon,
            args=(
                daemon_socket,
                target,
                relay,
                int(max_inflight_chunks),
                total_bytes,
            ),
            daemon=True,
        )
        worker_process = process_context.Process(
            target=run_worker_helper_process,
            args=(daemon_socket, worker_socket),
            daemon=True,
        )
        transfer_client = None
        source = None
        try:
            daemon_process.start()
            _wait_for_socket(daemon_socket, daemon_process, startup_timeout_seconds)
            worker_process.start()
            _wait_for_socket(worker_socket, worker_process, startup_timeout_seconds)

            daemon_client = TurboBusDaemonClient(daemon_socket)
            transfer_client = make_worker_managed_transfer_client(
                daemon_client,
                target_gpu=target,
                relay_gpus=[relay],
                worker_client=WorkerServiceSocketClient(worker_socket),
                max_inflight_chunks=int(max_inflight_chunks),
            )

            source = allocator.allocate("verify-cpu-source", job_id, total_bytes)
            source.write(pattern)
            torch.cuda.set_device(target)
            target_tensor = torch.empty(
                total_bytes,
                dtype=torch.uint8,
                device=f"cuda:{target}",
            )
            target_tensor.zero_()
            torch.cuda.synchronize(target)
            target_buffer = CudaIpcDeviceBuffer.from_device_pointer(
                buffer_id="verify-gpu-target",
                job_id=job_id,
                device_index=target,
                size_bytes=total_bytes,
                device_ptr=int(target_tensor.data_ptr()),
                backend=default_cuda_backend,
            )

            transfer = transfer_client.fetch_shared_cpu_to_cuda_ipc(
                source,
                target_buffer,
                ranges=({"src_offset": 0, "dst_offset": 0, "bytes": total_bytes},),
                chunk_bytes=chunk_size,
                mode="relay",
                job_id=job_id,
            )
            torch.cuda.synchronize(target)
            _assert_target_matches(torch, target_tensor, pattern)
            if transfer.state != "complete":
                raise RuntimeError(f"transfer did not complete: {transfer.state}")
            if transfer.bytes_completed != total_bytes:
                raise RuntimeError(
                    "transfer completed an unexpected byte count: "
                    f"{transfer.bytes_completed} != {total_bytes}"
                )
            if transfer.worker_completion is None:
                raise RuntimeError("worker helper did not return a completion envelope")
            if transfer.worker_completion.final_state != "complete":
                raise RuntimeError(
                    "worker helper did not complete: "
                    f"{transfer.worker_completion.final_state}"
                )

            daemon_profile = daemon_client.describe()
            if not daemon_profile.ok:
                raise RuntimeError(daemon_profile.error or "daemon describe failed")
            relay_quota = _relay_quota(daemon_profile.payload, relay)
            reservations_released = not bool(daemon_profile.payload.get("reservations"))
            active_chunks = int(relay_quota.get("active_chunks", -1))
            if not reservations_released:
                raise RuntimeError("daemon still has active reservations")
            if active_chunks != 0:
                raise RuntimeError("daemon relay active chunk count was not released")

            worker_completion = transfer.worker_completion
            worker_result = (
                {}
                if worker_completion.worker_result is None
                else dict(worker_completion.worker_result)
            )
            metadata = dict(worker_result.get("metadata") or {})
            if metadata.get("path") != "relay_h2d":
                raise RuntimeError("worker did not report the relay_h2d executor path")
            relay_bytes = int(metadata.get("relay_bytes", 0) or 0)
            relay_chunks = int(metadata.get("relay_chunks", 0) or 0)
            if relay_bytes != total_bytes:
                raise RuntimeError(
                    f"worker relay bytes mismatch: {relay_bytes} != {total_bytes}"
                )
            if relay_chunks <= 0:
                raise RuntimeError("worker did not report relay chunks")
            return WorkerManagedH2DRelayVerificationResult(
                transfer_id=transfer.transfer_id,
                job_id=job_id,
                bytes_requested=total_bytes,
                bytes_completed=transfer.bytes_completed,
                target_gpu=target,
                relay_gpu=relay,
                state=transfer.state,
                worker_final_state=(
                    None if worker_completion is None else worker_completion.final_state
                ),
                worker_relay_bytes=relay_bytes,
                worker_relay_chunks=relay_chunks,
                daemon_reservations_released=reservations_released,
                daemon_relay_active_chunks=active_chunks,
            )
        finally:
            if transfer_client is not None:
                try:
                    transfer_client.close_session()
                except Exception:
                    pass
            if source is not None:
                source.release()
            _terminate_process(worker_process)
            _terminate_process(daemon_process)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify TurboBus worker-managed CUDA H2D relay over helper socket",
    )
    parser.add_argument("--target-gpu", type=int, default=0)
    parser.add_argument("--relay-gpu", type=int, default=1)
    parser.add_argument("--bytes", type=int, default=1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--max-inflight-chunks", type=int, default=8)
    parser.add_argument("--socket-dir", default=None)
    parser.add_argument("--startup-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    result = verify_worker_managed_h2d_relay(
        target_gpu=args.target_gpu,
        relay_gpu=args.relay_gpu,
        bytes_to_copy=args.bytes,
        chunk_bytes=args.chunk_bytes,
        max_inflight_chunks=args.max_inflight_chunks,
        socket_dir=args.socket_dir,
        startup_timeout_seconds=args.startup_timeout_seconds,
    )
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


def _serve_verification_daemon(
    socket_path: str,
    target_gpu: int,
    relay_gpu: int,
    max_inflight_chunks: int,
    profile_bytes: int,
) -> None:
    daemon = _build_verification_daemon(
        target_gpu=target_gpu,
        relay_gpu=relay_gpu,
        max_inflight_chunks=max_inflight_chunks,
        profile_bytes=profile_bytes,
    )
    daemon.serve_forever(socket_path)


def _build_verification_daemon(
    *,
    target_gpu: int,
    relay_gpu: int,
    max_inflight_chunks: int,
    profile_bytes: int,
) -> TurboBusDaemon:
    target = int(target_gpu)
    relay = int(relay_gpu)
    daemon = TurboBusDaemon(
        relay_gpus=[relay],
        max_sessions_per_relay=1,
        max_inflight_chunks_per_relay=int(max_inflight_chunks),
        topology_provider=StaticTopologyProvider(
            DaemonResourceInventory(
                gpus=(
                    GpuInventoryRecord(
                        device_id=target,
                        backend="cuda",
                        vendor="nvidia",
                        role="target",
                    ),
                    GpuInventoryRecord(
                        device_id=relay,
                        backend="cuda",
                        vendor="nvidia",
                        role="relay",
                    ),
                ),
                pcie_paths=(
                    PciePathRecord(device_id=relay, bandwidth_gbps=16.0),
                ),
                fabric_links=(
                    FabricLinkRecord(
                        src_device_id=relay,
                        dst_device_id=target,
                        fabric="cuda_p2p",
                        bandwidth_gbps=40.0,
                        enabled=True,
                    ),
                ),
                source="verification_static",
            )
        ),
    )
    response = daemon.put_profile(
        target_gpu=target,
        relay_gpus=[relay],
        profile={
            "target_device": target,
            "direct_h2d_bw_gbps": 1.0,
            "direct_d2h_bw_gbps": 1.0,
            "relays": [
                {
                    "relay_device": relay,
                    "target_device": target,
                    "h2d_bw_gbps": 16.0,
                    "d2h_bw_gbps": 16.0,
                    "p2p_bw_gbps": 40.0,
                    "effective_bw_gbps": 16.0,
                    "effective_d2h_bw_gbps": 16.0,
                    "p2p_enabled": True,
                }
            ],
        },
        profile_bytes=int(profile_bytes),
    )
    if not response.ok:
        raise RuntimeError(response.error or "failed to seed daemon profile")
    return daemon


def _require_unix_sockets() -> None:
    if not hasattr(socket, "AF_UNIX"):
        raise RuntimeError("worker helper verification requires Unix domain sockets")


def _require_cuda_environment(target_gpu: int, relay_gpu: int):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CUDA verification") from exc
    default_cuda_backend.require_available()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_count = int(torch.cuda.device_count())
    required_devices = max(int(target_gpu), int(relay_gpu)) + 1
    if device_count < required_devices:
        raise RuntimeError(
            f"CUDA verification needs at least {required_devices} visible devices"
        )
    return torch


def _make_pattern(size_bytes: int) -> bytearray:
    size = int(size_bytes)
    pattern = bytearray(size)
    for index in range(size):
        pattern[index] = (index * 131 + 17) & 0xFF
    return pattern


def _assert_target_matches(torch, target_tensor, pattern: bytearray) -> None:
    actual = target_tensor.detach().cpu().contiguous()
    expected = _expected_tensor(torch, pattern)
    if torch.equal(actual, expected):
        return
    mismatch = (actual != expected).nonzero(as_tuple=False)
    index = int(mismatch[0].item()) if mismatch.numel() else -1
    expected_value = int(expected[index].item()) if index >= 0 else -1
    actual_value = int(actual[index].item()) if index >= 0 else -1
    raise AssertionError(
        "worker-managed H2D relay verification failed at byte "
        f"{index}: expected {expected_value}, got {actual_value}"
    )


def _expected_tensor(torch, pattern: bytearray):
    from_buffer = getattr(torch, "frombuffer", None)
    if callable(from_buffer):
        return from_buffer(pattern, dtype=torch.uint8).clone()
    return torch.tensor(list(pattern), dtype=torch.uint8)


def _wait_for_socket(
    socket_path: str,
    process: multiprocessing.Process,
    timeout_seconds: float,
) -> None:
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if process.exitcode is not None:
            raise RuntimeError(
                f"process exited before socket was ready: {socket_path}"
            )
        if os.path.exists(socket_path):
            return
        time.sleep(0.01)
    raise TimeoutError(f"socket was not created: {socket_path}")


def _terminate_process(process: multiprocessing.Process) -> None:
    if process.exitcode is not None:
        return
    process.terminate()
    process.join(timeout=5)
    if process.exitcode is None:
        process.kill()
        process.join(timeout=5)


def _relay_quota(payload: dict[str, object], relay_gpu: int) -> dict[str, object]:
    quotas = payload.get("relay_quotas", {})
    if not isinstance(quotas, dict):
        raise RuntimeError("daemon profile did not include relay quotas")
    quota = quotas.get(relay_gpu)
    if quota is None:
        quota = quotas.get(str(relay_gpu))
    if not isinstance(quota, dict):
        raise RuntimeError(f"daemon profile did not include relay {relay_gpu}")
    return quota


__all__ = [
    "WorkerManagedH2DRelayVerificationResult",
    "verify_worker_managed_h2d_relay",
]


if __name__ == "__main__":
    raise SystemExit(main())
