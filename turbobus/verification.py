from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

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
class WorkerManagedRelayVerificationResult:
    direction: str
    transfer_mode: str
    transfer_id: str
    job_id: str
    bytes_requested: int
    bytes_completed: int
    src_offset: int
    dst_offset: int
    source_buffer_bytes: int
    destination_buffer_bytes: int
    target_gpu: int
    relay_gpu: int
    state: str
    worker_final_state: str | None
    worker_path: str
    worker_direct_bytes: int
    worker_direct_chunks: int
    worker_relay_bytes: int
    worker_relay_chunks: int
    daemon_reservations_released: bool
    daemon_relay_active_chunks: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerManagedH2DRelayVerificationResult(WorkerManagedRelayVerificationResult):
    pass


@dataclass(frozen=True)
class WorkerManagedD2HRelayVerificationResult(WorkerManagedRelayVerificationResult):
    pass


def verify_worker_managed_h2d_relay(
    *,
    target_gpu: int = 0,
    relay_gpu: int = 1,
    bytes_to_copy: int = 1024 * 1024,
    chunk_bytes: int = 1024 * 1024,
    mode: str = "relay",
    src_offset: int = 0,
    dst_offset: int = 0,
    source_buffer_bytes: int | None = None,
    destination_buffer_bytes: int | None = None,
    max_inflight_chunks: int = 8,
    socket_dir: str | None = None,
    startup_timeout_seconds: float = 10.0,
) -> WorkerManagedH2DRelayVerificationResult:
    """Run the first real helper-socket H2D relay verification.

    The verifier starts a daemon socket and a worker helper process, then moves
    bytes from TurboBus shared CPU memory through a daemon-approved relay plan
    into a CUDA IPC target tensor.
    """

    return _verify_worker_managed_relay(
        direction="h2d",
        result_type=WorkerManagedH2DRelayVerificationResult,
        target_gpu=target_gpu,
        relay_gpu=relay_gpu,
        bytes_to_copy=bytes_to_copy,
        chunk_bytes=chunk_bytes,
        mode=mode,
        src_offset=src_offset,
        dst_offset=dst_offset,
        source_buffer_bytes=source_buffer_bytes,
        destination_buffer_bytes=destination_buffer_bytes,
        max_inflight_chunks=max_inflight_chunks,
        socket_dir=socket_dir,
        startup_timeout_seconds=startup_timeout_seconds,
    )


def verify_worker_managed_d2h_relay(
    *,
    target_gpu: int = 0,
    relay_gpu: int = 1,
    bytes_to_copy: int = 1024 * 1024,
    chunk_bytes: int = 1024 * 1024,
    mode: str = "relay",
    src_offset: int = 0,
    dst_offset: int = 0,
    source_buffer_bytes: int | None = None,
    destination_buffer_bytes: int | None = None,
    max_inflight_chunks: int = 8,
    socket_dir: str | None = None,
    startup_timeout_seconds: float = 10.0,
) -> WorkerManagedD2HRelayVerificationResult:
    """Run helper-socket D2H relay verification through the daemon plan path."""

    return _verify_worker_managed_relay(
        direction="d2h",
        result_type=WorkerManagedD2HRelayVerificationResult,
        target_gpu=target_gpu,
        relay_gpu=relay_gpu,
        bytes_to_copy=bytes_to_copy,
        chunk_bytes=chunk_bytes,
        mode=mode,
        src_offset=src_offset,
        dst_offset=dst_offset,
        source_buffer_bytes=source_buffer_bytes,
        destination_buffer_bytes=destination_buffer_bytes,
        max_inflight_chunks=max_inflight_chunks,
        socket_dir=socket_dir,
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _verify_worker_managed_relay(
    *,
    direction: str,
    result_type: type[WorkerManagedRelayVerificationResult],
    target_gpu: int,
    relay_gpu: int,
    bytes_to_copy: int,
    chunk_bytes: int,
    mode: str,
    src_offset: int,
    dst_offset: int,
    source_buffer_bytes: int | None,
    destination_buffer_bytes: int | None,
    max_inflight_chunks: int,
    socket_dir: str | None,
    startup_timeout_seconds: float,
) -> WorkerManagedRelayVerificationResult:
    _require_unix_sockets()
    direction = str(direction).lower()
    if direction not in {"h2d", "d2h"}:
        raise ValueError("direction must be h2d or d2h")
    mode = str(mode).lower()
    if mode not in {"direct", "relay", "pool"}:
        raise ValueError("mode must be direct, relay, or pool")
    target = int(target_gpu)
    relay = int(relay_gpu)
    total_bytes = int(bytes_to_copy)
    chunk_size = int(chunk_bytes)
    if total_bytes <= 0:
        raise ValueError("bytes_to_copy must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_bytes must be positive")
    if mode == "pool" and total_bytes <= chunk_size:
        raise ValueError("pool verification requires at least two chunks")
    if int(max_inflight_chunks) <= 0:
        raise ValueError("max_inflight_chunks must be positive")
    src_offset = int(src_offset)
    dst_offset = int(dst_offset)
    source_size, destination_size = _resolve_verification_buffer_sizes(
        bytes_to_copy=total_bytes,
        src_offset=src_offset,
        dst_offset=dst_offset,
        source_buffer_bytes=source_buffer_bytes,
        destination_buffer_bytes=destination_buffer_bytes,
    )

    torch = _require_cuda_environment(
        target,
        _cuda_environment_relay_gpu(mode, relay),
    )
    pattern = _make_pattern(total_bytes)
    job_id = f"verify-worker-{direction}-{mode}-{os.getpid()}-{time.time_ns()}"
    allocator = SharedPinnedCpuBufferAllocator(
        name_prefix=f"tb-worker-{direction}-{mode}-verify"
    )

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
        worker_process = (
            None
            if not _worker_helper_required(mode)
            else process_context.Process(
                target=run_worker_helper_process,
                args=(daemon_socket, worker_socket),
                daemon=True,
            )
        )
        transfer_client = None
        cpu_buffer = None
        try:
            daemon_process.start()
            _wait_for_socket(daemon_socket, daemon_process, startup_timeout_seconds)
            worker_client = _UnusedWorkerClient()
            if worker_process is not None:
                worker_process.start()
                _wait_for_socket(worker_socket, worker_process, startup_timeout_seconds)
                worker_client = WorkerServiceSocketClient(worker_socket)

            daemon_client = TurboBusDaemonClient(daemon_socket)
            transfer_client = make_worker_managed_transfer_client(
                daemon_client,
                target_gpu=target,
                relay_gpus=[relay],
                worker_client=worker_client,
                max_inflight_chunks=int(max_inflight_chunks),
            )

            torch.cuda.set_device(target)
            if direction == "h2d":
                cpu_buffer = allocator.allocate(
                    "verify-cpu-source",
                    job_id,
                    source_size,
                )
                _zero_shared_cpu_buffer(cpu_buffer)
                cpu_buffer.write(pattern, offset=src_offset)
                target_tensor = torch.empty(
                    destination_size,
                    dtype=torch.uint8,
                    device=f"cuda:{target}",
                )
                target_tensor.zero_()
                torch.cuda.synchronize(target)
                gpu_buffer = CudaIpcDeviceBuffer.from_device_pointer(
                    buffer_id="verify-gpu-target",
                    job_id=job_id,
                    device_index=target,
                    size_bytes=destination_size,
                    device_ptr=int(target_tensor.data_ptr()),
                    backend=default_cuda_backend,
                )

                transfer = transfer_client.fetch_shared_cpu_to_cuda_ipc(
                    cpu_buffer,
                    gpu_buffer,
                    ranges=(
                        {
                            "src_offset": src_offset,
                            "dst_offset": dst_offset,
                            "bytes": total_bytes,
                        },
                    ),
                    chunk_bytes=chunk_size,
                    mode=mode,
                    job_id=job_id,
                )
                torch.cuda.synchronize(target)
                _assert_target_matches(
                    torch,
                    target_tensor,
                    _expected_payload(destination_size, pattern, dst_offset),
                )
            else:
                source_tensor = torch.empty(
                    source_size,
                    dtype=torch.uint8,
                    device=f"cuda:{target}",
                )
                source_tensor.zero_()
                source_pattern = _expected_tensor(torch, pattern).to(
                    device=f"cuda:{target}"
                )
                source_tensor[src_offset : src_offset + total_bytes].copy_(
                    source_pattern
                )
                torch.cuda.synchronize(target)
                gpu_buffer = CudaIpcDeviceBuffer.from_device_pointer(
                    buffer_id="verify-gpu-source",
                    job_id=job_id,
                    device_index=target,
                    size_bytes=source_size,
                    device_ptr=int(source_tensor.data_ptr()),
                    backend=default_cuda_backend,
                )
                cpu_buffer = allocator.allocate(
                    "verify-cpu-destination",
                    job_id,
                    destination_size,
                )
                _zero_shared_cpu_buffer(cpu_buffer)

                transfer = transfer_client.offload_cuda_ipc_to_shared_cpu(
                    gpu_buffer,
                    cpu_buffer,
                    ranges=(
                        {
                            "src_offset": src_offset,
                            "dst_offset": dst_offset,
                            "bytes": total_bytes,
                        },
                    ),
                    chunk_bytes=chunk_size,
                    mode=mode,
                    job_id=job_id,
                )
                torch.cuda.synchronize(target)
                _assert_shared_cpu_matches(
                    cpu_buffer,
                    _expected_payload(destination_size, pattern, dst_offset),
                )
            resolved_mode = _resolved_transfer_mode(transfer.plan, mode)
            _assert_transfer_complete(
                transfer,
                total_bytes,
                require_worker_completion=resolved_mode != "direct",
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
            if worker_completion is None:
                worker_path = f"direct_{direction}"
                direct_bytes = _planned_path_bytes(transfer.plan["plan"], "direct")
                direct_chunks = _planned_path_chunks(transfer.plan["plan"], "direct")
                relay_bytes = 0
                relay_chunks = 0
            else:
                worker_result = (
                    {}
                    if worker_completion.worker_result is None
                    else dict(worker_completion.worker_result)
                )
                metadata = dict(worker_result.get("metadata") or {})
                worker_path = str(metadata.get("path", ""))
                direct_bytes = int(metadata.get("direct_bytes", 0) or 0)
                direct_chunks = int(metadata.get("direct_chunks", 0) or 0)
                relay_bytes = int(metadata.get("relay_bytes", 0) or 0)
                relay_chunks = int(metadata.get("relay_chunks", 0) or 0)
            _assert_worker_path_split(
                direction=direction,
                mode=resolved_mode,
                total_bytes=total_bytes,
                worker_path=worker_path,
                direct_bytes=direct_bytes,
                direct_chunks=direct_chunks,
                relay_bytes=relay_bytes,
                relay_chunks=relay_chunks,
            )
            return result_type(
                direction=direction,
                transfer_mode=resolved_mode,
                transfer_id=transfer.transfer_id,
                job_id=job_id,
                bytes_requested=total_bytes,
                bytes_completed=transfer.bytes_completed,
                src_offset=src_offset,
                dst_offset=dst_offset,
                source_buffer_bytes=source_size,
                destination_buffer_bytes=destination_size,
                target_gpu=target,
                relay_gpu=relay,
                state=transfer.state,
                worker_final_state=(
                    None if worker_completion is None else worker_completion.final_state
                ),
                worker_path=worker_path,
                worker_direct_bytes=direct_bytes,
                worker_direct_chunks=direct_chunks,
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
            if cpu_buffer is not None:
                cpu_buffer.release()
            if worker_process is not None:
                _terminate_process(worker_process)
            _terminate_process(daemon_process)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify TurboBus worker-managed CUDA relay over helper socket",
    )
    parser.add_argument("--direction", choices=["h2d", "d2h"], default="h2d")
    parser.add_argument("--mode", choices=["direct", "relay", "pool"], default="relay")
    parser.add_argument("--target-gpu", type=int, default=0)
    parser.add_argument("--relay-gpu", type=int, default=1)
    parser.add_argument("--bytes", type=int, default=1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--src-offset", type=int, default=0)
    parser.add_argument("--dst-offset", type=int, default=0)
    parser.add_argument("--source-buffer-bytes", type=int, default=None)
    parser.add_argument("--destination-buffer-bytes", type=int, default=None)
    parser.add_argument("--max-inflight-chunks", type=int, default=8)
    parser.add_argument("--socket-dir", default=None)
    parser.add_argument("--startup-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    verifier = (
        verify_worker_managed_h2d_relay
        if args.direction == "h2d"
        else verify_worker_managed_d2h_relay
    )
    result = verifier(
        target_gpu=args.target_gpu,
        relay_gpu=args.relay_gpu,
        bytes_to_copy=args.bytes,
        chunk_bytes=args.chunk_bytes,
        mode=args.mode,
        src_offset=args.src_offset,
        dst_offset=args.dst_offset,
        source_buffer_bytes=args.source_buffer_bytes,
        destination_buffer_bytes=args.destination_buffer_bytes,
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


def _require_cuda_environment(target_gpu: int, relay_gpu: int | None = None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CUDA verification") from exc
    default_cuda_backend.require_available()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_count = int(torch.cuda.device_count())
    required_devices = _required_cuda_device_count(target_gpu, relay_gpu)
    if device_count < required_devices:
        raise RuntimeError(
            f"CUDA verification needs at least {required_devices} visible devices"
        )
    return torch


def _cuda_environment_relay_gpu(mode: str, relay_gpu: int) -> int | None:
    return int(relay_gpu) if _worker_helper_required(mode) else None


def _required_cuda_device_count(target_gpu: int, relay_gpu: int | None = None) -> int:
    devices = [int(target_gpu)]
    if relay_gpu is not None:
        devices.append(int(relay_gpu))
    return max(devices) + 1


def _make_pattern(size_bytes: int) -> bytearray:
    size = int(size_bytes)
    pattern = bytearray(size)
    for index in range(size):
        pattern[index] = (index * 131 + 17) & 0xFF
    return pattern


def _resolve_verification_buffer_sizes(
    *,
    bytes_to_copy: int,
    src_offset: int,
    dst_offset: int,
    source_buffer_bytes: int | None,
    destination_buffer_bytes: int | None,
) -> tuple[int, int]:
    total = int(bytes_to_copy)
    src_offset = int(src_offset)
    dst_offset = int(dst_offset)
    if src_offset < 0 or dst_offset < 0:
        raise ValueError("range offsets must be non-negative")
    required_source = src_offset + total
    required_destination = dst_offset + total
    source_size = (
        required_source
        if source_buffer_bytes is None
        else int(source_buffer_bytes)
    )
    destination_size = (
        required_destination
        if destination_buffer_bytes is None
        else int(destination_buffer_bytes)
    )
    if source_size < required_source:
        raise ValueError("source_buffer_bytes must cover src_offset + bytes_to_copy")
    if destination_size < required_destination:
        raise ValueError(
            "destination_buffer_bytes must cover dst_offset + bytes_to_copy"
        )
    return source_size, destination_size


def _worker_helper_required(mode: str) -> bool:
    return str(mode).lower() != "direct"


class _UnusedWorkerClient:
    def submit_envelope(self, envelope):
        raise RuntimeError("direct verification should not use a worker helper")

    def submit_report_cleanup_lifecycle(
        self,
        request,
        cleanup_target_kind="reservation",
    ):
        raise RuntimeError("direct verification should not use a worker helper")


def _expected_payload(
    size_bytes: int,
    pattern: bytearray,
    offset: int,
) -> bytearray:
    expected = bytearray(int(size_bytes))
    offset = int(offset)
    expected[offset : offset + len(pattern)] = pattern
    return expected


def _zero_shared_cpu_buffer(cpu_buffer) -> None:
    view = cpu_buffer.view
    try:
        block_size = min(len(view), 1024 * 1024)
        if block_size <= 0:
            return
        block = b"\x00" * block_size
        for offset in range(0, len(view), block_size):
            size = min(block_size, len(view) - offset)
            view[offset : offset + size] = block[:size]
    finally:
        view.release()


def _assert_target_matches(torch, target_tensor, expected: bytearray) -> None:
    actual = target_tensor.detach().cpu().contiguous()
    expected_tensor = _expected_tensor(torch, expected)
    if torch.equal(actual, expected_tensor):
        return
    mismatch = (actual != expected_tensor).nonzero(as_tuple=False)
    index = int(mismatch[0].item()) if mismatch.numel() else -1
    expected_value = int(expected_tensor[index].item()) if index >= 0 else -1
    actual_value = int(actual[index].item()) if index >= 0 else -1
    raise AssertionError(
        "worker-managed H2D relay verification failed at byte "
        f"{index}: expected {expected_value}, got {actual_value}"
    )


def _assert_shared_cpu_matches(cpu_buffer, expected: bytearray) -> None:
    actual = cpu_buffer.read()
    expected_bytes = bytes(expected)
    if actual == expected_bytes:
        return
    mismatch_index = -1
    expected_value = -1
    actual_value = -1
    for index, (actual_byte, expected_byte) in enumerate(zip(actual, expected_bytes)):
        if actual_byte != expected_byte:
            mismatch_index = index
            expected_value = expected_byte
            actual_value = actual_byte
            break
    if mismatch_index < 0 and len(actual) != len(expected_bytes):
        mismatch_index = min(len(actual), len(expected_bytes))
    raise AssertionError(
        "worker-managed D2H relay verification failed at byte "
        f"{mismatch_index}: expected {expected_value}, got {actual_value}"
    )


def _resolved_transfer_mode(
    plan_payload: Mapping[str, object],
    requested_mode: str,
) -> str:
    stats = plan_payload.get("stats")
    if isinstance(stats, Mapping):
        return str(stats.get("resolved_mode", requested_mode)).lower()
    return str(requested_mode).lower()


def _planned_path_bytes(plan_payload: Mapping[str, object], path_kind: str) -> int:
    total = 0
    for assignment in plan_payload.get("assignments", ()) or ():
        if not isinstance(assignment, Mapping):
            continue
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            continue
        if str(path.get("kind", "")).lower() != path_kind:
            continue
        for chunk in assignment.get("chunks", ()) or ():
            if isinstance(chunk, Mapping):
                total += int(chunk.get("bytes", 0))
    return total


def _planned_path_chunks(plan_payload: Mapping[str, object], path_kind: str) -> int:
    total = 0
    for assignment in plan_payload.get("assignments", ()) or ():
        if not isinstance(assignment, Mapping):
            continue
        path = assignment.get("path")
        if not isinstance(path, Mapping):
            continue
        if str(path.get("kind", "")).lower() == path_kind:
            total += len(assignment.get("chunks", ()) or ())
    return total


def _assert_transfer_complete(
    transfer,
    total_bytes: int,
    *,
    require_worker_completion: bool,
) -> None:
    if transfer.state != "complete":
        raise RuntimeError(f"transfer did not complete: {transfer.state}")
    if transfer.bytes_completed != total_bytes:
        raise RuntimeError(
            "transfer completed an unexpected byte count: "
            f"{transfer.bytes_completed} != {total_bytes}"
        )
    if require_worker_completion and transfer.worker_completion is None:
        raise RuntimeError("worker helper did not return a completion envelope")
    if (
        transfer.worker_completion is not None
        and transfer.worker_completion.final_state != "complete"
    ):
        raise RuntimeError(
            "worker helper did not complete: "
            f"{transfer.worker_completion.final_state}"
        )


def _assert_worker_path_split(
    *,
    direction: str,
    mode: str,
    total_bytes: int,
    worker_path: str,
    direct_bytes: int,
    direct_chunks: int,
    relay_bytes: int,
    relay_chunks: int,
) -> None:
    expected_path = f"{mode}_{direction}"
    if worker_path != expected_path:
        raise RuntimeError(f"worker did not report the {expected_path} executor path")
    if mode == "direct":
        if relay_bytes != 0 or relay_chunks != 0:
            raise RuntimeError("direct verification unexpectedly reported relay work")
        if direct_chunks <= 0:
            raise RuntimeError("direct verification did not report direct chunks")
        if direct_bytes != total_bytes:
            raise RuntimeError(
                f"worker direct bytes mismatch: {direct_bytes} != {total_bytes}"
            )
        return
    if relay_chunks <= 0:
        raise RuntimeError("worker did not report relay chunks")
    if relay_bytes <= 0:
        raise RuntimeError("worker did not report relay bytes")
    if mode == "relay":
        if direct_bytes != 0 or direct_chunks != 0:
            raise RuntimeError("relay verification unexpectedly reported direct work")
        if relay_bytes != total_bytes:
            raise RuntimeError(
                f"worker relay bytes mismatch: {relay_bytes} != {total_bytes}"
            )
        return
    if direct_chunks <= 0 or direct_bytes <= 0:
        raise RuntimeError("pool verification did not report direct chunks")
    if direct_bytes + relay_bytes != total_bytes:
        raise RuntimeError(
            "worker pool byte split mismatch: "
            f"{direct_bytes} + {relay_bytes} != {total_bytes}"
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
    "WorkerManagedD2HRelayVerificationResult",
    "WorkerManagedH2DRelayVerificationResult",
    "WorkerManagedRelayVerificationResult",
    "verify_worker_managed_d2h_relay",
    "verify_worker_managed_h2d_relay",
]


if __name__ == "__main__":
    raise SystemExit(main())
