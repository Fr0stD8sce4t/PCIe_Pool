from __future__ import annotations

import time
from typing import Iterable

from turbobus.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
    TopologyProvider,
)


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
            source="test_fixture_static",
            discovered_at=time.time(),
            metadata={"discovery": "static test fixture"},
        )
        return cls(inventory)

    def snapshot(self) -> DaemonResourceInventory:
        return self._inventory

    def invalidate(self) -> None:
        return None


__all__ = [
    "DaemonResourceInventory",
    "FabricLinkRecord",
    "GpuInventoryRecord",
    "PciePathRecord",
    "StaticTopologyProvider",
]
