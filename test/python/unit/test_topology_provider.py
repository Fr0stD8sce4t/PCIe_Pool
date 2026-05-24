from __future__ import annotations

import unittest

from turbobus.daemon.startup import (
    DaemonStartupConfig,
    DaemonStartupError,
    create_production_daemon,
    relay_candidates_for_policy,
)
from turbobus.daemon.__main__ import build_parser, startup_config_from_args
from turbobus.topology import (
    DaemonResourceInventory,
    FabricLinkRecord,
    GpuInventoryRecord,
    PciePathRecord,
)
from turbobus.topology.cuda_nvml import CudaNvmlTopologyProvider
from test.python.fixtures.topology import StaticTopologyProvider


class TopologyProviderTest(unittest.TestCase):
    def test_cuda_provider_builds_versioned_inventory_and_schema_snapshot(self) -> None:
        clock = FakeClock(100.0)
        provider = CudaNvmlTopologyProvider(
            FakeCudaProbe(),
            cache_max_age_seconds=10.0,
            now=clock,
        )

        first = provider.snapshot()
        second = provider.snapshot()
        topology = first.to_topology_snapshot()

        self.assertIs(first, second)
        self.assertEqual(first.source, "cuda_nvml")
        self.assertEqual(first.version, 1)
        self.assertEqual(first.snapshot_id, "topology-cuda_nvml-v1-100.000000")
        self.assertEqual(first.gpus[0].uuid, "GPU-0000")
        self.assertEqual(first.gpus[0].pci_bus_id, "0000:01:00.0")
        self.assertEqual(first.gpus[0].memory_bytes, 80 * 1024 * 1024)
        self.assertEqual(first.pcie_paths[0].root_complex, "0000:01")
        self.assertEqual(first.fabric_links[0].fabric, "nvlink")
        self.assertEqual(first.fabric_links[0].src_device_id, 0)
        self.assertEqual(first.fabric_links[0].dst_device_id, 1)
        self.assertEqual(topology.snapshot_id, first.snapshot_id)
        self.assertEqual(topology.version, 1)
        self.assertEqual(topology.devices[0]["uuid"], "GPU-0000")
        self.assertEqual(topology.pcie_links[0]["root_complex"], "0000:01")

    def test_cuda_provider_invalidates_cached_snapshot(self) -> None:
        clock = FakeClock(100.0)
        provider = CudaNvmlTopologyProvider(
            FakeCudaProbe(),
            cache_max_age_seconds=10.0,
            now=clock,
        )

        first = provider.snapshot()
        clock.value = 101.0
        provider.invalidate()
        second = provider.snapshot()

        self.assertIsNot(first, second)
        self.assertEqual(second.version, 2)
        self.assertEqual(second.snapshot_id, "topology-cuda_nvml-v2-101.000000")

    def test_production_startup_selects_eligible_relays_from_provider_inventory(self) -> None:
        config = DaemonStartupConfig(
            target_gpu=0,
            min_relay_count=1,
            require_fabric=True,
            require_pcie=True,
        )
        provider = StaticTopologyProvider(production_inventory())

        daemon = create_production_daemon(config, topology_provider=provider)
        discovered = daemon.discover_relays(target_gpu=0)

        self.assertTrue(discovered.ok)
        payload = discovered.payload["relay_discovery"]
        self.assertEqual(payload["topology_snapshot_id"], "topology-production-v7")
        self.assertEqual(payload["topology_version"], 7)
        self.assertEqual(payload["requested_relays"], [1])
        self.assertEqual(payload["summary"]["eligible_relay_count"], 1)
        self.assertEqual(payload["relays"][0]["eligibility"]["reason"], "eligible")

    def test_production_startup_rejects_fixture_topology(self) -> None:
        config = DaemonStartupConfig(target_gpu=0, min_relay_count=1)

        with self.assertRaisesRegex(DaemonStartupError, "synthetic topology fixtures"):
            create_production_daemon(
                config,
                topology_provider=StaticTopologyProvider.from_relay_gpus([1]),
            )

    def test_production_startup_reports_filtered_relay_reasons(self) -> None:
        inventory = DaemonResourceInventory(
            gpus=(
                GpuInventoryRecord(device_id=0, backend="cuda", vendor="nvidia"),
                GpuInventoryRecord(device_id=1, backend="cuda", vendor="nvidia"),
            ),
            pcie_paths=(PciePathRecord(device_id=1),),
            fabric_links=(
                FabricLinkRecord(
                    src_device_id=1,
                    dst_device_id=0,
                    fabric="nvlink",
                    enabled=False,
                ),
            ),
            source="cuda_nvml",
            discovered_at=1.0,
            snapshot_id="topology-cuda_nvml-v1",
            version=1,
        )
        config = DaemonStartupConfig(target_gpu=0, min_relay_count=1)

        with self.assertRaisesRegex(DaemonStartupError, "missing enabled fabric link"):
            relay_candidates_for_policy(inventory, config)

    def test_daemon_cli_builds_production_topology_startup_config(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--target-gpu",
                "0",
                "--min-relays",
                "2",
                "--allow-missing-fabric",
                "--allow-missing-pcie",
                "--topology-provider",
                "cuda-nvml",
            ]
        )
        config = startup_config_from_args(args)

        self.assertEqual(config.topology_provider, "cuda-nvml")
        self.assertEqual(config.target_gpu, 0)
        self.assertEqual(config.min_relay_count, 2)
        self.assertFalse(config.require_fabric)
        self.assertFalse(config.require_pcie)
        self.assertFalse(hasattr(args, "relay_gpus"))


class FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeCudaProbe:
    def query_gpu_inventory(self):
        return (
            {
                "device_id": 0,
                "uuid": "GPU-0000",
                "pci_bus_id": "0000:01:00.0",
                "memory_mib": "80",
            },
            {
                "device_id": 1,
                "uuid": "GPU-1111",
                "pci_bus_id": "0000:02:00.0",
                "memory_mib": "40",
            },
        )

    def query_topology_matrix(self):
        return (
            "        GPU0    GPU1",
            "GPU0    X       NV2",
            "GPU1    NV2     X",
        )


def production_inventory() -> DaemonResourceInventory:
    return DaemonResourceInventory(
        gpus=(
            GpuInventoryRecord(
                device_id=0,
                backend="cuda",
                vendor="nvidia",
                uuid="GPU-0000",
                pci_bus_id="0000:01:00.0",
            ),
            GpuInventoryRecord(
                device_id=1,
                backend="cuda",
                vendor="nvidia",
                uuid="GPU-1111",
                pci_bus_id="0000:02:00.0",
            ),
        ),
        pcie_paths=(
            PciePathRecord(device_id=0, root_complex="0000:01"),
            PciePathRecord(device_id=1, root_complex="0000:02"),
        ),
        fabric_links=(
            FabricLinkRecord(
                src_device_id=1,
                dst_device_id=0,
                fabric="nvlink",
                bandwidth_gbps=100.0,
                enabled=True,
            ),
        ),
        source="cuda_nvml",
        discovered_at=7.0,
        snapshot_id="topology-production-v7",
        version=7,
    )


if __name__ == "__main__":
    unittest.main()
