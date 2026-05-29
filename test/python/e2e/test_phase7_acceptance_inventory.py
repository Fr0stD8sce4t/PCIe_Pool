from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_acceptance_inventory  # noqa: E402


class Phase7AcceptanceInventoryTest(unittest.TestCase):
    def test_accepts_real_bundle_and_explicit_server_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle_path = write_json(tmp / "2gpu" / "bundle-gate.json", bundle_gate("2gpu"))
            manifest = {
                "server_classes": [
                    {
                        "server_class": "2gpu",
                        "status": "accepted",
                        "real_artifacts": True,
                        "bundle_gate": "2gpu/bundle-gate.json",
                    },
                    blocked_entry("4gpu", "4 GPU server is not currently available"),
                    blocked_entry("8gpu", "8 GPU server is missing vLLM installation"),
                ],
            }

            report = phase7_acceptance_inventory.build_inventory(
                manifest,
                base_dir=tmp,
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["blocked_count"], 2)
            self.assertTrue(report["real_workload_accepted"])
            self.assertEqual(
                report["server_classes"][0]["bundle_gate"]["path"],
                str(bundle_path),
            )
            self.assertEqual(len(report["next_commands"]), 2)
            self.assertIn(
                "4 GPU server is not currently available",
                report["server_classes"][1]["block_reason"],
            )

    def test_rejects_missing_server_class_gap(self) -> None:
        manifest = {
            "server_classes": [
                {
                    "server_class": "2gpu",
                    "status": "accepted",
                    "real_artifacts": True,
                    "bundle_gate": "2gpu/bundle-gate.json",
                },
                blocked_entry("4gpu", "4 GPU server is not currently available"),
            ],
        }

        report = phase7_acceptance_inventory.build_inventory(
            manifest,
            base_dir=Path("/does/not/matter"),
        )

        self.assertFalse(report["ok"])
        self.assertIn("8gpu:missing_manifest_entry", report["errors"])
        self.assertTrue(
            any(error.startswith("2gpu:bundle_gate_missing:") for error in report["errors"])
        )

    def test_rejects_accepted_bundle_without_real_vllm_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_json(
                tmp / "bundle-gate.json",
                bundle_gate(
                    "2gpu",
                    workloads=("model-loading", "training-offload", "optimizer-offload"),
                ),
            )
            manifest = {
                "server_classes": [
                    {
                        "server_class": "2gpu",
                        "status": "accepted",
                        "real_artifacts": True,
                        "bundle_gate": "bundle-gate.json",
                    },
                    blocked_entry("4gpu", "4 GPU server is not currently available"),
                    blocked_entry("8gpu", "8 GPU server is not currently available"),
                ],
            }

            report = phase7_acceptance_inventory.build_inventory(
                manifest,
                base_dir=tmp,
            )

            self.assertFalse(report["ok"])
            self.assertIn("2gpu:accepted_missing_vllm_kv_bundle", report["errors"])
            self.assertIn(
                "phase7:no_accepted_real_llm_framework_bundle",
                report["errors"],
            )

    def test_rejects_missing_environment_gap_and_next_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_json(tmp / "bundle-gate.json", bundle_gate("2gpu"))
            manifest = {
                "server_classes": {
                    "2gpu": {
                        "status": "accepted",
                        "real_artifacts": True,
                        "bundle_gate": "bundle-gate.json",
                    },
                    "4gpu": {"status": "blocked"},
                    "8gpu": {
                        "status": "missing",
                        "environment_gaps": ["hardware_unavailable"],
                    },
                },
            }

            report = phase7_acceptance_inventory.build_inventory(
                manifest,
                base_dir=tmp,
            )

            self.assertFalse(report["ok"])
            self.assertIn("4gpu:blocked_missing_environment_gap", report["errors"])
            self.assertIn("4gpu:blocked_missing_next_commands", report["errors"])
            self.assertIn("8gpu:missing_missing_next_commands", report["errors"])

    def test_cli_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_json(tmp / "2gpu" / "bundle-gate.json", bundle_gate("2gpu"))
            manifest_path = write_json(
                tmp / "acceptance-manifest.json",
                {
                    "server_classes": [
                        {
                            "server_class": "2gpu",
                            "status": "accepted",
                            "real_artifacts": True,
                            "bundle_gate": "2gpu/bundle-gate.json",
                        },
                        blocked_entry("4gpu", "4 GPU server is not currently available"),
                        blocked_entry("8gpu", "8 GPU server is not currently available"),
                    ],
                },
            )
            output_path = tmp / "inventory.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_acceptance_inventory.py"),
                    "--manifest",
                    str(manifest_path),
                    "--json-output",
                    str(output_path),
                ],
                cwd=BENCHMARKS.parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["manifest_path"], str(manifest_path))


WORKLOADS = (
    "model-loading",
    "training-offload",
    "optimizer-offload",
    "vllm-kv",
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def blocked_entry(server_class: str, reason: str) -> dict:
    return {
        "server_class": server_class,
        "status": "blocked",
        "block_reason": reason,
        "environment_gaps": ["hardware_or_environment_unavailable"],
        "next_commands": [f"run Phase 7 bundle gate on {server_class} server"],
        "remaining_risks": [reason],
    }


def bundle_gate(
    server_class: str,
    workloads: tuple[str, ...] = WORKLOADS,
    *,
    ok: bool = True,
) -> dict:
    return {
        "ok": ok,
        "errors": [] if ok else ["bundle failed"],
        "warnings": [],
        "server_class": server_class,
        "baseline": {"workloads": list(workloads)},
        "turbobus": {"workloads": list(workloads)},
        "comparison": {"ok": ok, "workloads": list(workloads)},
        "evidence": {"ok": ok, "workloads": list(workloads)},
        "correctness": {"provided": True, "ok": True},
        "artifacts": {
            "baseline_result": f"{server_class}/paper-baseline/result.json",
            "turbobus_result": f"{server_class}/turbobus-daemon/result.json",
            "comparison": f"{server_class}/comparison.json",
            "evidence": [f"{server_class}/turbobus-daemon/evidence.json"],
            "correctness": [f"{server_class}/correctness.json"],
        },
    }


if __name__ == "__main__":
    unittest.main()
