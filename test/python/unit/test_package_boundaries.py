from __future__ import annotations

import unittest

from turbobus.daemon.server import TurboBusDaemon


class PackageBoundaryTest(unittest.TestCase):
    def test_daemon_does_not_construct_synthetic_topology_by_default(self) -> None:
        daemon = TurboBusDaemon(relay_gpus=[1])

        inventory = daemon.get_inventory()
        discovered = daemon.discover_relays(target_gpu=0, requested_relays=[1])

        self.assertFalse(inventory.ok)
        self.assertFalse(discovered.ok)
        self.assertIn("topology provider is required", inventory.error)
        self.assertIn("topology provider is required", discovered.error)


if __name__ == "__main__":
    unittest.main()
