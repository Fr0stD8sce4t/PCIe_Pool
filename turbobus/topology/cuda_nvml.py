from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from . import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
    TopologyProvider,
)


class TopologyDiscoveryError(RuntimeError):
    pass


class CudaTopologyProbe(Protocol):
    def query_gpu_inventory(self) -> Sequence[Mapping[str, object]]:
        ...

    def query_topology_matrix(self) -> Sequence[str]:
        ...


@dataclass(frozen=True)
class NvidiaSmiProbe:
    executable: str = "nvidia-smi"
    timeout_seconds: float = 5.0

    def query_gpu_inventory(self) -> Sequence[Mapping[str, object]]:
        command = [
            self.executable,
            "--query-gpu=index,uuid,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits",
        ]
        output = self._run(command)
        rows = []
        for row in csv.reader(output.splitlines()):
            if not row:
                continue
            if len(row) < 4:
                raise TopologyDiscoveryError(
                    "nvidia-smi gpu inventory output is missing fields"
                )
            rows.append(
                {
                    "device_id": row[0].strip(),
                    "uuid": row[1].strip(),
                    "pci_bus_id": row[2].strip(),
                    "memory_mib": row[3].strip(),
                }
            )
        return rows

    def query_topology_matrix(self) -> Sequence[str]:
        try:
            return self._run([self.executable, "topo", "-m"]).splitlines()
        except TopologyDiscoveryError:
            return ()

    def _run(self, command: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                check=False,
                text=True,
                timeout=float(self.timeout_seconds),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TopologyDiscoveryError(f"failed to run {command[0]}: {exc}") from exc
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout or "").strip()
            raise TopologyDiscoveryError(
                f"{command[0]} failed with exit code {completed.returncode}: {error}"
            )
        return completed.stdout


class CudaNvmlTopologyProvider(TopologyProvider):
    def __init__(
        self,
        probe: CudaTopologyProbe | None = None,
        *,
        cache_max_age_seconds: float = 30.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._probe = NvidiaSmiProbe() if probe is None else probe
        self._cache_max_age_seconds = max(0.0, float(cache_max_age_seconds))
        self._now = time.time if now is None else now
        self._cached_inventory: DaemonResourceInventory | None = None
        self._version = 0

    def snapshot(self) -> DaemonResourceInventory:
        now = float(self._now())
        if self._cached_inventory is not None:
            age = now - self._cached_inventory.discovered_at
            if self._cache_max_age_seconds > 0.0 and age <= self._cache_max_age_seconds:
                return self._cached_inventory

        gpus = tuple(_gpu_record(row) for row in self._probe.query_gpu_inventory())
        if not gpus:
            raise TopologyDiscoveryError("cuda-nvml topology discovery found no GPUs")
        self._version += 1
        inventory = DaemonResourceInventory(
            gpus=gpus,
            pcie_paths=tuple(_pcie_path_for_gpu(gpu) for gpu in gpus),
            fabric_links=tuple(
                _fabric_links_from_topology_matrix(
                    gpus,
                    self._probe.query_topology_matrix(),
                )
            ),
            source="cuda_nvml",
            discovered_at=now,
            snapshot_id=f"topology-cuda_nvml-v{self._version}-{now:.6f}",
            version=self._version,
            metadata={
                "provider": "cuda-nvml",
                "gpu_count": len(gpus),
                "cache_max_age_seconds": self._cache_max_age_seconds,
            },
        )
        self._cached_inventory = inventory
        return inventory

    def invalidate(self) -> None:
        self._cached_inventory = None


def _gpu_record(row: Mapping[str, object]) -> GpuInventoryRecord:
    device_id = int(row.get("device_id", row.get("index", 0)))
    memory_bytes = _memory_bytes(row)
    return GpuInventoryRecord(
        device_id=device_id,
        backend="cuda",
        vendor="nvidia",
        uuid=_optional_str(row.get("uuid")),
        pci_bus_id=_optional_str(row.get("pci_bus_id")),
        memory_bytes=memory_bytes,
        role="general",
        visible=bool(row.get("visible", True)),
    )


def _pcie_path_for_gpu(gpu: GpuInventoryRecord) -> PciePathRecord:
    return PciePathRecord(
        device_id=gpu.device_id,
        root_complex=_root_complex_from_pci_bus_id(gpu.pci_bus_id),
    )


def _memory_bytes(row: Mapping[str, object]) -> int | None:
    if "memory_bytes" in row and row["memory_bytes"] is not None:
        return int(row["memory_bytes"])
    if "memory_mib" in row and row["memory_mib"] is not None:
        return int(float(str(row["memory_mib"]).strip())) * 1024 * 1024
    return None


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _root_complex_from_pci_bus_id(pci_bus_id: str | None) -> str | None:
    if not pci_bus_id:
        return None
    parts = str(pci_bus_id).split(":")
    if len(parts) < 2:
        return str(pci_bus_id)
    return ":".join(parts[:2])


def _fabric_links_from_topology_matrix(
    gpus: Sequence[GpuInventoryRecord],
    matrix_lines: Sequence[str],
) -> tuple[FabricLinkRecord, ...]:
    gpu_ids = {gpu.device_id for gpu in gpus}
    rows = [line.split() for line in matrix_lines if str(line).strip()]
    header = next((row for row in rows if row and row[0].startswith("GPU")), None)
    if not header:
        return ()

    links: list[FabricLinkRecord] = []
    for row in rows:
        if not row or not row[0].startswith("GPU"):
            continue
        src = _matrix_gpu_id(row[0])
        if src is None or src not in gpu_ids:
            continue
        for column, header_name in enumerate(header):
            if column + 1 >= len(row):
                continue
            dst = _matrix_gpu_id(header_name)
            if dst is None or dst not in gpu_ids or dst == src:
                continue
            if src > dst:
                continue
            token = row[column + 1]
            fabric = _fabric_name(token)
            if fabric is None:
                continue
            links.append(
                FabricLinkRecord(
                    src_device_id=src,
                    dst_device_id=dst,
                    fabric=fabric,
                    bandwidth_gbps=0.0,
                    bidirectional=True,
                    enabled=True,
                )
            )
    return tuple(links)


def _matrix_gpu_id(value: str) -> int | None:
    text = str(value).strip()
    if not text.startswith("GPU"):
        return None
    try:
        return int(text[3:])
    except ValueError:
        return None


def _fabric_name(token: str) -> str | None:
    normalized = str(token).strip().upper()
    if not normalized or normalized in {"X", "N/A", "SYS", "NODE"}:
        return None
    if normalized.startswith("NV"):
        return "nvlink"
    if normalized in {"PIX", "PXB", "PHB"}:
        return "cuda_p2p"
    return None


__all__ = [
    "CudaNvmlTopologyProvider",
    "CudaTopologyProbe",
    "NvidiaSmiProbe",
    "TopologyDiscoveryError",
]
