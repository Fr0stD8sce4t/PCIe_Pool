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


_GPU_QUERY_FIELD_SETS = (
    (
        "index",
        "uuid",
        "pci.bus_id",
        "memory.total",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ),
    ("index", "uuid", "pci.bus_id", "memory.total"),
)

_GPU_QUERY_FIELD_KEYS = {
    "index": "device_id",
    "uuid": "uuid",
    "pci.bus_id": "pci_bus_id",
    "memory.total": "memory_mib",
    "pcie.link.gen.current": "pcie_link_gen_current",
    "pcie.link.width.current": "pcie_link_width_current",
}

_PCIE_EFFECTIVE_GBPS_PER_LANE = {
    1: 0.25,
    2: 0.5,
    3: 0.985,
    4: 1.969,
    5: 3.938,
    6: 7.877,
}

_PCIE_RAW_GT_PER_SECOND = {
    1: 2.5,
    2: 5.0,
    3: 8.0,
    4: 16.0,
    5: 32.0,
    6: 64.0,
}


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
        last_error: TopologyDiscoveryError | None = None
        for fields in _GPU_QUERY_FIELD_SETS:
            try:
                return self._query_gpu_inventory_fields(fields)
            except TopologyDiscoveryError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return ()

    def _query_gpu_inventory_fields(
        self,
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, object]]:
        output = self._run(
            [
                self.executable,
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ]
        )
        rows = []
        for row in csv.reader(output.splitlines()):
            if not row:
                continue
            if len(row) < len(fields):
                raise TopologyDiscoveryError(
                    "nvidia-smi gpu inventory output is missing fields"
                )
            rows.append(_inventory_row_from_query(fields, row))
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

        gpu_rows = tuple(self._probe.query_gpu_inventory())
        gpus = tuple(_gpu_record(row) for row in gpu_rows)
        if not gpus:
            raise TopologyDiscoveryError("cuda-nvml topology discovery found no GPUs")
        self._version += 1
        inventory = DaemonResourceInventory(
            gpus=gpus,
            pcie_paths=tuple(
                _pcie_path_for_gpu(gpu, row)
                for gpu, row in zip(gpus, gpu_rows)
            ),
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
        backend=_optional_str(row.get("backend")) or "cuda",
        vendor=_optional_str(row.get("vendor")) or "nvidia",
        uuid=_optional_str(row.get("uuid")),
        pci_bus_id=_optional_str(row.get("pci_bus_id")),
        numa_node=_first_int(row, ("numa_node", "numa")),
        memory_bytes=memory_bytes,
        role=_optional_str(row.get("role")) or "general",
        visible=_optional_bool(row.get("visible"), default=True),
    )


def _pcie_path_for_gpu(
    gpu: GpuInventoryRecord,
    row: Mapping[str, object],
) -> PciePathRecord:
    link_generation = _first_int(
        row,
        (
            "pcie_link_gen_current",
            "pcie.link.gen.current",
            "link_generation",
        ),
    )
    link_width = _first_int(
        row,
        (
            "pcie_link_width_current",
            "pcie.link.width.current",
            "link_width",
        ),
    )
    explicit_bandwidth = _first_float(
        row,
        ("pcie_bandwidth_gbps", "bandwidth_gbps"),
    )
    estimated_bandwidth = _estimate_pcie_bandwidth_gbps(
        link_generation,
        link_width,
    )
    bandwidth = (
        explicit_bandwidth
        if explicit_bandwidth is not None
        else estimated_bandwidth or 0.0
    )
    bandwidth_source = None
    if explicit_bandwidth is not None:
        bandwidth_source = "provider"
    elif estimated_bandwidth is not None:
        bandwidth_source = "estimated_from_pcie_generation_width"
    negotiated_speed_gtps = _first_float(
        row,
        (
            "pcie_speed_gtps",
            "negotiated_speed_gtps",
            "pcie.link.speed.current",
        ),
    )
    if negotiated_speed_gtps is None:
        negotiated_speed_gtps = _pcie_raw_gtps_for_generation(link_generation)
    return PciePathRecord(
        device_id=gpu.device_id,
        numa_node=gpu.numa_node,
        root_complex=(
            _optional_str(row.get("root_complex"))
            or _root_complex_from_pci_bus_id(gpu.pci_bus_id)
        ),
        link_generation=link_generation,
        link_width=link_width,
        bandwidth_gbps=bandwidth,
        negotiated_speed_gtps=negotiated_speed_gtps,
        switch_hierarchy=_switch_hierarchy(row.get("switch_hierarchy")),
        bandwidth_source=bandwidth_source,
    )


def _inventory_row_from_query(
    fields: Sequence[str],
    values: Sequence[str],
) -> Mapping[str, object]:
    row: dict[str, object] = {}
    for field, value in zip(fields, values):
        row[_GPU_QUERY_FIELD_KEYS.get(field, field.replace(".", "_"))] = value.strip()
    return row


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
    if text.upper() in {"N/A", "NA", "NONE", "UNKNOWN", "[NOT SUPPORTED]"}:
        return None
    return text or None


def _optional_bool(value: object | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "visible"}:
        return True
    if text in {"0", "false", "no", "n", "hidden"}:
        return False
    return bool(value)


def _first_int(
    row: Mapping[str, object],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = _optional_str(value)
        if text is None:
            continue
        try:
            return int(float(text))
        except ValueError:
            continue
    return None


def _first_float(
    row: Mapping[str, object],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = _optional_str(value)
        if text is None:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


def _switch_hierarchy(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.replace(">", ",").replace("/", ",").split(",")
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(value, Sequence):
        return tuple(
            text
            for item in value
            if (text := str(item).strip())
        )
    text = str(value).strip()
    return (text,) if text else ()


def _estimate_pcie_bandwidth_gbps(
    link_generation: int | None,
    link_width: int | None,
) -> float | None:
    if link_generation is None or link_width is None:
        return None
    per_lane = _PCIE_EFFECTIVE_GBPS_PER_LANE.get(int(link_generation))
    if per_lane is None or int(link_width) <= 0:
        return None
    return round(per_lane * int(link_width), 3)


def _pcie_raw_gtps_for_generation(link_generation: int | None) -> float | None:
    if link_generation is None:
        return None
    return _PCIE_RAW_GT_PER_SECOND.get(int(link_generation))


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
            capability = _fabric_capability(token)
            if capability is None:
                continue
            links.append(
                FabricLinkRecord(
                    src_device_id=src,
                    dst_device_id=dst,
                    fabric=str(capability["fabric"]),
                    bandwidth_gbps=0.0,
                    bidirectional=True,
                    enabled=True,
                    link_count=capability.get("link_count"),
                    capability=str(capability["capability"]),
                    raw_link_type=str(capability["raw_link_type"]),
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


def _fabric_capability(token: str) -> dict[str, object] | None:
    normalized = str(token).strip().upper()
    if not normalized or normalized in {"X", "N/A", "SYS", "NODE"}:
        return None
    if normalized.startswith("NV"):
        fabric = "nvswitch" if normalized in {"NVSW", "NVS", "NVSWITCH"} else "nvlink"
        return {
            "fabric": fabric,
            "capability": fabric,
            "link_count": _nvlink_count(normalized),
            "raw_link_type": normalized,
        }
    if normalized in {"PIX", "PXB", "PHB"}:
        return {
            "fabric": "cuda_p2p",
            "capability": f"cuda_p2p_{normalized.lower()}",
            "link_count": None,
            "raw_link_type": normalized,
        }
    return None


def _nvlink_count(token: str) -> int | None:
    suffix = str(token).strip().upper()[2:]
    if not suffix.isdigit():
        return None
    return int(suffix)


__all__ = [
    "CudaNvmlTopologyProvider",
    "CudaTopologyProbe",
    "NvidiaSmiProbe",
    "TopologyDiscoveryError",
]
