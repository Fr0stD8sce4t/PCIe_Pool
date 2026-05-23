from __future__ import annotations

import unittest

import turbobus
from turbobus import inference as root_inference
from turbobus import model_loading as root_model_loading
from turbobus import training_offload as root_training_offload
from turbobus import vllm as root_vllm
from turbobus import vllm_connector as root_vllm_connector
from turbobus import vllm_integration as root_vllm_integration
from turbobus import vllm_kv_connector as root_vllm_kv_connector
from turbobus.adapters import (
    FrameworkAdapter,
    InferenceKVSlotAdapter,
    ModelWeightLoader,
    TurboBusConnector,
    TurboBusConnectorConfig,
    TurboBusSavedPrefix,
    TrainingOffloadManager,
    VllmKVSlotAdapter,
    VllmTurboBusConnector,
    VllmTurboBusIntegration,
)
from turbobus.adapters import inference as adapter_inference
from turbobus.adapters import model_loading as adapter_model_loading
from turbobus.adapters import training_offload as adapter_training_offload
from turbobus.adapters import vllm as adapter_vllm
from turbobus.adapters import vllm_connector as adapter_vllm_connector
from turbobus.adapters import vllm_integration as adapter_vllm_integration
from turbobus.adapters import vllm_kv_connector as adapter_vllm_kv_connector


class AdaptersPackageTest(unittest.TestCase):
    def test_adapter_package_reexports_current_framework_entry_points(self) -> None:
        self.assertIs(root_inference, adapter_inference)
        self.assertIs(root_model_loading, adapter_model_loading)
        self.assertIs(root_training_offload, adapter_training_offload)
        self.assertIs(root_vllm, adapter_vllm)
        self.assertIs(root_vllm_connector, adapter_vllm_connector)
        self.assertIs(root_vllm_integration, adapter_vllm_integration)
        self.assertIs(root_vllm_kv_connector, adapter_vllm_kv_connector)
        self.assertIs(
            adapter_inference.InferenceKVSlotAdapter,
            root_inference.InferenceKVSlotAdapter,
        )
        self.assertIs(
            adapter_model_loading.ModelWeightLoader,
            root_model_loading.ModelWeightLoader,
        )
        self.assertIs(
            adapter_training_offload.TrainingOffloadManager,
            root_training_offload.TrainingOffloadManager,
        )
        self.assertIs(adapter_vllm.VllmKVSlotAdapter, root_vllm.VllmKVSlotAdapter)
        self.assertIs(
            adapter_vllm_connector.VllmTurboBusConnector,
            root_vllm_connector.VllmTurboBusConnector,
        )
        self.assertIs(
            adapter_vllm_integration.VllmTurboBusIntegration,
            root_vllm_integration.VllmTurboBusIntegration,
        )
        self.assertIs(
            adapter_vllm_kv_connector.TurboBusConnector,
            root_vllm_kv_connector.TurboBusConnector,
        )

    def test_top_level_exports_now_flow_through_adapter_boundary(self) -> None:
        self.assertIs(ModelWeightLoader, root_model_loading.ModelWeightLoader)
        self.assertIs(InferenceKVSlotAdapter, root_inference.InferenceKVSlotAdapter)
        self.assertIs(
            TrainingOffloadManager,
            root_training_offload.TrainingOffloadManager,
        )
        self.assertIs(VllmKVSlotAdapter, root_vllm.VllmKVSlotAdapter)
        self.assertIs(VllmTurboBusConnector, root_vllm_connector.VllmTurboBusConnector)
        self.assertIs(
            VllmTurboBusIntegration,
            root_vllm_integration.VllmTurboBusIntegration,
        )
        self.assertIs(TurboBusConnector, root_vllm_kv_connector.TurboBusConnector)
        self.assertIs(
            TurboBusConnectorConfig,
            root_vllm_kv_connector.TurboBusConnectorConfig,
        )
        self.assertIs(TurboBusSavedPrefix, root_vllm_kv_connector.TurboBusSavedPrefix)
        self.assertIs(turbobus.ModelWeightLoader, ModelWeightLoader)
        self.assertIn("FrameworkAdapter", turbobus.__all__)
        self.assertIsNotNone(FrameworkAdapter)


if __name__ == "__main__":
    unittest.main()
