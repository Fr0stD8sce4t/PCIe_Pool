from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class GpuInventoryRecord:
    device_id: int
    backend: str = "unknown"
    vendor: str = "unknown"
    pci_bus_id: str | None = None
    numa_node: int | None = None
    memory_bytes: int | None = None
    role: str = "general"
    visible: bool = True

    def __post_init__(self) -> None:
        device_id = int(self.device_id)
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if self.numa_node is not None and int(self.numa_node) < 0:
            raise ValueError("numa_node must be non-negative")
        if self.memory_bytes is not None and int(self.memory_bytes) < 0:
            raise ValueError("memory_bytes must be non-negative")
        if not str(self.backend).strip():
            raise ValueError("backend must be non-empty")
        if not str(self.vendor).strip():
            raise ValueError("vendor must be non-empty")
        if not str(self.role).strip():
            raise ValueError("role must be non-empty")
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "vendor", str(self.vendor))
        if self.pci_bus_id is not None:
            object.__setattr__(self, "pci_bus_id", str(self.pci_bus_id))
        if self.numa_node is not None:
            object.__setattr__(self, "numa_node", int(self.numa_node))
        if self.memory_bytes is not None:
            object.__setattr__(self, "memory_bytes", int(self.memory_bytes))
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True)
class PciePathRecord:
    device_id: int
    numa_node: int | None = None
    root_complex: str | None = None
    link_generation: int | None = None
    link_width: int | None = None
    bandwidth_gbps: float = 0.0

    def __post_init__(self) -> None:
        device_id = int(self.device_id)
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if self.numa_node is not None and int(self.numa_node) < 0:
            raise ValueError("numa_node must be non-negative")
        if self.link_generation is not None and int(self.link_generation) < 0:
            raise ValueError("link_generation must be non-negative")
        if self.link_width is not None and int(self.link_width) < 0:
            raise ValueError("link_width must be non-negative")
        bandwidth = float(self.bandwidth_gbps)
        if bandwidth < 0.0:
            raise ValueError("bandwidth_gbps must be non-negative")
        object.__setattr__(self, "device_id", device_id)
        if self.numa_node is not None:
            object.__setattr__(self, "numa_node", int(self.numa_node))
        if self.root_complex is not None:
            object.__setattr__(self, "root_complex", str(self.root_complex))
        if self.link_generation is not None:
            object.__setattr__(self, "link_generation", int(self.link_generation))
        if self.link_width is not None:
            object.__setattr__(self, "link_width", int(self.link_width))
        object.__setattr__(self, "bandwidth_gbps", bandwidth)


@dataclass(frozen=True)
class FabricLinkRecord:
    src_device_id: int
    dst_device_id: int
    fabric: str = "unknown"
    bandwidth_gbps: float = 0.0
    bidirectional: bool = True
    enabled: bool = False

    def __post_init__(self) -> None:
        src_device_id = int(self.src_device_id)
        dst_device_id = int(self.dst_device_id)
        if src_device_id < 0 or dst_device_id < 0:
            raise ValueError("fabric link device ids must be non-negative")
        if src_device_id == dst_device_id:
            raise ValueError("fabric link endpoints must be distinct")
        if not str(self.fabric).strip():
            raise ValueError("fabric must be non-empty")
        bandwidth = float(self.bandwidth_gbps)
        if bandwidth < 0.0:
            raise ValueError("bandwidth_gbps must be non-negative")
        object.__setattr__(self, "src_device_id", src_device_id)
        object.__setattr__(self, "dst_device_id", dst_device_id)
        object.__setattr__(self, "fabric", str(self.fabric))
        object.__setattr__(self, "bandwidth_gbps", bandwidth)
        object.__setattr__(self, "bidirectional", bool(self.bidirectional))
        object.__setattr__(self, "enabled", bool(self.enabled))


@dataclass(frozen=True)
class DaemonResourceInventory:
    gpus: tuple[GpuInventoryRecord, ...] = field(default_factory=tuple)
    pcie_paths: tuple[PciePathRecord, ...] = field(default_factory=tuple)
    fabric_links: tuple[FabricLinkRecord, ...] = field(default_factory=tuple)
    source: str = "discovered"
    discovered_at: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ValueError("source must be non-empty")
        discovered_at = float(self.discovered_at)
        if discovered_at < 0.0:
            raise ValueError("discovered_at must be non-negative")
        object.__setattr__(self, "gpus", tuple(self.gpus))
        object.__setattr__(self, "pcie_paths", tuple(self.pcie_paths))
        object.__setattr__(self, "fabric_links", tuple(self.fabric_links))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "discovered_at", discovered_at)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def eligible_relay_devices(
        self,
        target_device: int,
        requested_relays: Iterable[int],
    ) -> tuple[int, ...]:
        return tuple(
            item["relay_gpu"]
            for item in self.relay_eligibility(target_device, requested_relays)[
                "eligible_relays"
            ]
        )

    def relay_eligibility(
        self,
        target_device: int,
        requested_relays: Iterable[int],
    ) -> dict[str, object]:
        candidates = tuple(sorted({int(gpu) for gpu in requested_relays}))
        if not candidates:
            return {
                "requested_relays": [],
                "eligible_relays": [],
                "filtered_relays": [],
                "inventory_source": self.source,
                "inventory_discovered_at": self.discovered_at,
            }

        filtered: list[dict[str, object]] = []
        if self.gpus:
            known_gpus = {gpu.device_id for gpu in self.gpus}
            candidates, removed = _partition_candidates(candidates, known_gpus)
            filtered.extend(
                {"relay_gpu": gpu, "reason": "unknown gpu"}
                for gpu in removed
            )
        if self.pcie_paths:
            pcie_devices = {path.device_id for path in self.pcie_paths}
            candidates, removed = _partition_candidates(candidates, pcie_devices)
            filtered.extend(
                {"relay_gpu": gpu, "reason": "missing pcie path"}
                for gpu in removed
            )
        if self.fabric_links:
            target = int(target_device)
            eligible = tuple(
                gpu
                for gpu in candidates
                if _has_enabled_fabric_link(self, gpu, target)
            )
            filtered.extend(
                {"relay_gpu": gpu, "reason": "missing enabled fabric link"}
                for gpu in candidates
                if gpu not in eligible
            )
            candidates = eligible
        return {
            "requested_relays": list(sorted({int(gpu) for gpu in requested_relays})),
            "eligible_relays": [
                {"relay_gpu": gpu, "reason": "eligible"}
                for gpu in candidates
            ],
            "filtered_relays": filtered,
            "inventory_source": self.source,
            "inventory_discovered_at": self.discovered_at,
        }


class TopologyProvider:
    def snapshot(self) -> DaemonResourceInventory:
        raise NotImplementedError


def _has_enabled_fabric_link(
    inventory: DaemonResourceInventory,
    relay_device: int,
    target_device: int,
) -> bool:
    relay = int(relay_device)
    target = int(target_device)
    for link in inventory.fabric_links:
        if not link.enabled:
            continue
        if link.src_device_id == relay and link.dst_device_id == target:
            return True
        if (
            link.bidirectional
            and link.src_device_id == target
            and link.dst_device_id == relay
        ):
            return True
    return False


def _partition_candidates(
    candidates: tuple[int, ...],
    allowed: set[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    kept = tuple(gpu for gpu in candidates if gpu in allowed)
    removed = tuple(gpu for gpu in candidates if gpu not in allowed)
    return kept, removed


__all__ = [
    "DaemonResourceInventory",
    "FabricLinkRecord",
    "GpuInventoryRecord",
    "PciePathRecord",
    "TopologyProvider",
]
