from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .server import TurboBusDaemon
from ..topology import DaemonResourceInventory, TopologyProvider
from ..topology.cuda_nvml import CudaNvmlTopologyProvider, TopologyDiscoveryError


class DaemonStartupError(RuntimeError):
    pass


@dataclass(frozen=True)
class DaemonStartupConfig:
    topology_provider: str = "cuda-nvml"
    target_gpu: int | None = None
    min_relay_count: int = 1
    require_fabric: bool = True
    require_pcie: bool = True
    max_sessions_per_relay: int = 1
    max_inflight_chunks_per_relay: int = 8
    session_timeout_seconds: float = 0.0
    profile_max_age_seconds: float = 0.0

    def __post_init__(self) -> None:
        min_relay_count = int(self.min_relay_count)
        if min_relay_count < 0:
            raise ValueError("min_relay_count must be non-negative")
        if self.target_gpu is not None and int(self.target_gpu) < 0:
            raise ValueError("target_gpu must be non-negative")
        object.__setattr__(
            self,
            "topology_provider",
            str(self.topology_provider).strip().lower().replace("_", "-"),
        )
        object.__setattr__(self, "min_relay_count", min_relay_count)
        if self.target_gpu is not None:
            object.__setattr__(self, "target_gpu", int(self.target_gpu))


def build_topology_provider(name: str) -> TopologyProvider:
    normalized = str(name).strip().lower().replace("_", "-")
    if normalized in {"cuda-nvml", "nvidia-smi"}:
        return CudaNvmlTopologyProvider()
    raise DaemonStartupError(f"unsupported topology provider: {name}")


def create_production_daemon(
    config: DaemonStartupConfig,
    *,
    topology_provider: TopologyProvider | None = None,
) -> TurboBusDaemon:
    provider = topology_provider or build_topology_provider(config.topology_provider)
    inventory = _snapshot_or_startup_error(provider)
    relays = relay_candidates_for_policy(inventory, config)
    return TurboBusDaemon(
        relays,
        max_sessions_per_relay=config.max_sessions_per_relay,
        max_inflight_chunks_per_relay=config.max_inflight_chunks_per_relay,
        session_timeout_seconds=config.session_timeout_seconds,
        profile_max_age_seconds=config.profile_max_age_seconds,
        topology_provider=provider,
    )


def relay_candidates_for_policy(
    inventory: DaemonResourceInventory,
    config: DaemonStartupConfig,
) -> tuple[int, ...]:
    _reject_fixture_inventory(inventory)
    if not inventory.gpus:
        raise DaemonStartupError("topology discovery found no GPUs")
    if config.require_pcie and not inventory.pcie_paths:
        raise DaemonStartupError("topology discovery did not report PCIe paths")
    if config.require_fabric and not inventory.fabric_links:
        raise DaemonStartupError("topology discovery did not report GPU fabric links")

    visible_gpus = tuple(sorted(gpu.device_id for gpu in inventory.gpus if gpu.visible))
    if config.target_gpu is None:
        if len(visible_gpus) < config.min_relay_count:
            raise DaemonStartupError(
                "topology discovery found fewer visible GPUs than min_relay_count"
            )
        return visible_gpus

    target = int(config.target_gpu)
    if target not in {gpu.device_id for gpu in inventory.gpus}:
        raise DaemonStartupError(f"target GPU {target} was not discovered")
    eligibility = inventory.relay_eligibility(
        target_device=target,
        requested_relays=visible_gpus,
    )
    relays = tuple(item["relay_gpu"] for item in eligibility["eligible_relays"])
    if len(relays) < config.min_relay_count:
        filtered = ", ".join(
            f"{item['relay_gpu']}:{item['reason']}"
            for item in eligibility["filtered_relays"]
        )
        raise DaemonStartupError(
            "topology discovery could not satisfy relay policy: "
            f"target_gpu={target} min_relay_count={config.min_relay_count} "
            f"eligible_relays={list(relays)} filtered_relays=[{filtered}]"
        )
    return relays


def _snapshot_or_startup_error(provider: TopologyProvider) -> DaemonResourceInventory:
    try:
        return provider.snapshot()
    except TopologyDiscoveryError as exc:
        raise DaemonStartupError(str(exc)) from exc


def _reject_fixture_inventory(inventory: DaemonResourceInventory) -> None:
    source = str(inventory.source).lower()
    discovery = str(inventory.metadata.get("discovery", "")).lower()
    if source.startswith("test_fixture") or "test fixture" in discovery:
        raise DaemonStartupError(
            "production daemon startup cannot use synthetic topology fixtures"
        )


__all__ = [
    "DaemonStartupConfig",
    "DaemonStartupError",
    "build_topology_provider",
    "create_production_daemon",
    "relay_candidates_for_policy",
]
