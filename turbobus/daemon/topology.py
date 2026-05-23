from __future__ import annotations

import time
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
    source: str = "static"
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


class TopologyProvider:
    def snapshot(self) -> DaemonResourceInventory:
        raise NotImplementedError


class StaticTopologyProvider(TopologyProvider):
    def __init__(self, inventory: DaemonResourceInventory) -> None:
        self._inventory = inventory

    @classmethod
    def from_relay_gpus(cls, relay_gpus: Iterable[int]) -> "StaticTopologyProvider":
        relays = tuple(sorted({int(gpu) for gpu in relay_gpus}))
        inventory = DaemonResourceInventory(
            gpus=tuple(
                GpuInventoryRecord(device_id=gpu, role="relay")
                for gpu in relays
            ),
            pcie_paths=tuple(PciePathRecord(device_id=gpu) for gpu in relays),
            source="configured",
            discovered_at=time.time(),
            metadata={"discovery": "static relay configuration"},
        )
        return cls(inventory)

    def snapshot(self) -> DaemonResourceInventory:
        return self._inventory
