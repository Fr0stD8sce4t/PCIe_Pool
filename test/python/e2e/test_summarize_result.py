from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import summarize_result  # noqa: E402


class SummarizeResultTest(unittest.TestCase):
    def test_dispatches_model_loading_json_to_daemon_first_summary(self) -> None:
        data = {
            "config": {
                "source_buffer_id": "cpu-buffer",
                "destination_buffer_id": "gpu-buffer",
            },
            "summary": {},
        }

        with mock.patch.object(
            summarize_result.model_loading,
            "compact_summary",
            return_value="model-summary",
        ) as compact_summary:
            self.assertEqual(
                summarize_result.compact_summary_for_result(data),
                "model-summary",
            )

        compact_summary.assert_called_once_with(data)

    def test_dispatches_training_offload_json_to_daemon_first_summary(self) -> None:
        data = {
            "config": {
                "cpu_buffer_id": "cpu-buffer",
                "gpu_buffer_id": "gpu-buffer",
            },
            "summary": {
                "prefetch": {},
                "offload": {},
            },
        }

        with mock.patch.object(
            summarize_result.training_offload,
            "compact_summary",
            return_value="training-summary",
        ) as compact_summary:
            self.assertEqual(
                summarize_result.compact_summary_for_result(data),
                "training-summary",
            )

        compact_summary.assert_called_once_with(data)

    def test_dispatches_paper_validation_json_to_daemon_first_summary(self) -> None:
        data = {
            "config": {
                "workloads": ["model-loading"],
            },
            "workloads": [],
        }

        with mock.patch.object(
            summarize_result.paper_validation,
            "compact_summary",
            return_value="paper-summary",
        ) as compact_summary:
            self.assertEqual(
                summarize_result.compact_summary_for_result(data),
                "paper-summary",
            )

        compact_summary.assert_called_once_with(data)

    def test_rejects_old_or_unknown_json_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported daemon-first"):
            summarize_result.compact_summary_for_result(
                {
                    "config": {
                        "target_gpu": 0,
                        "relay_gpus": [1],
                        "mode": "pool",
                    },
                    "modes": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
