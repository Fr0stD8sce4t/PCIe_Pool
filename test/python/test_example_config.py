from __future__ import annotations

import unittest
from unittest import mock

from turbobus.example_config import configure_cuda_runtime_mapping, parse_gpu_list


class ExampleConfigTest(unittest.TestCase):
    def test_parse_gpu_list_ignores_empty_items(self) -> None:
        self.assertEqual(parse_gpu_list("5,,7,"), [5, 7])

    def test_configure_cuda_runtime_mapping_sets_visible_devices(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            mapping = configure_cuda_runtime_mapping(6, "5,7")

        self.assertEqual(mapping.physical_target_gpu, 6)
        self.assertEqual(mapping.physical_relay_gpus, (5, 7))
        self.assertEqual(mapping.runtime_target_gpu, 0)
        self.assertEqual(mapping.runtime_relay_gpus, (1, 2))
        self.assertEqual(mapping.cuda_visible_devices, "6,5,7")

    def test_configure_cuda_runtime_mapping_preserves_existing_visible_devices(self) -> None:
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "6,5"}):
            mapping = configure_cuda_runtime_mapping(6, "5")

        self.assertEqual(mapping.runtime_target_gpu, 6)
        self.assertEqual(mapping.runtime_relay_gpus, (5,))
        self.assertEqual(mapping.cuda_visible_devices, "6,5")


if __name__ == "__main__":
    unittest.main()
