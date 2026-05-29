from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_ingest_artifacts  # noqa: E402


class Phase7IngestArtifactsTest(unittest.TestCase):
    def test_ingests_accepted_bundle_and_runs_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle_path = write_json(tmp / "2gpu" / "bundle-gate.json", bundle_gate("2gpu"))
            manifest_path = write_json(
                tmp / "acceptance-manifest.json",
                {
                    "server_classes": [
                        blocked_entry("4gpu", "4 GPU server is unavailable"),
                        blocked_entry("8gpu", "8 GPU server is unavailable"),
                    ],
                },
            )
            inventory_output = tmp / "acceptance-inventory.json"
            entry = phase7_ingest_artifacts.build_entry(
                server_class="2gpu",
                status="accepted",
                manifest_path=manifest_path,
                bundle_gate=str(bundle_path),
                real_artifacts=True,
            )

            report = phase7_ingest_artifacts.ingest_entry_file(
                manifest_path=manifest_path,
                entry=entry,
                inventory_output_path=inventory_output,
            )

            self.assertTrue(report["ok"])
            self.assertTrue(report["written"])
            self.assertTrue(report["inventory_ok"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["server_classes"][0]["server_class"], "2gpu")
            self.assertEqual(
                manifest["server_classes"][0]["bundle_gate"],
                "2gpu/bundle-gate.json",
            )
            inventory = json.loads(inventory_output.read_text(encoding="utf-8"))
            self.assertTrue(inventory["ok"])
            self.assertEqual(inventory["accepted_count"], 1)

    def test_rejects_accepted_bundle_without_real_artifact_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle_path = write_json(tmp / "2gpu" / "bundle-gate.json", bundle_gate("2gpu"))
            manifest_path = tmp / "acceptance-manifest.json"
            entry = phase7_ingest_artifacts.build_entry(
                server_class="2gpu",
                status="accepted",
                manifest_path=manifest_path,
                bundle_gate=str(bundle_path),
                real_artifacts=False,
            )

            report = phase7_ingest_artifacts.ingest_entry_file(
                manifest_path=manifest_path,
                entry=entry,
            )

            self.assertFalse(report["ok"])
            self.assertFalse(report["written"])
            self.assertIn("2gpu:accepted_requires_real_artifacts", report["errors"])
            self.assertFalse(manifest_path.exists())

    def test_rejects_blocked_entry_without_gap_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "acceptance-manifest.json"
            entry = phase7_ingest_artifacts.build_entry(
                server_class="4gpu",
                status="blocked",
                manifest_path=manifest_path,
            )

            report = phase7_ingest_artifacts.ingest_entry_file(
                manifest_path=manifest_path,
                entry=entry,
            )

            self.assertFalse(report["ok"])
            self.assertIn("4gpu:blocked_missing_environment_gap", report["errors"])
            self.assertIn("4gpu:blocked_missing_next_commands", report["errors"])
            self.assertFalse(manifest_path.exists())

    def test_allow_incomplete_inventory_writes_incremental_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "acceptance-manifest.json"
            entry = phase7_ingest_artifacts.build_entry(
                server_class="4gpu",
                status="blocked",
                manifest_path=manifest_path,
                block_reason="4 GPU server is unavailable",
                environment_gaps=["hardware_unavailable"],
                next_commands=["run Phase 7 bundle gate on a 4 GPU server"],
            )

            report = phase7_ingest_artifacts.ingest_entry_file(
                manifest_path=manifest_path,
                entry=entry,
                allow_incomplete_inventory=True,
            )

            self.assertTrue(report["ok"])
            self.assertTrue(report["written"])
            self.assertFalse(report["inventory_ok"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["server_classes"][0]["server_class"], "4gpu")

    def test_cli_writes_manifest_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle_path = write_json(tmp / "2gpu" / "bundle-gate.json", bundle_gate("2gpu"))
            manifest_path = write_json(
                tmp / "acceptance-manifest.json",
                {
                    "server_classes": [
                        blocked_entry("4gpu", "4 GPU server is unavailable"),
                        blocked_entry("8gpu", "8 GPU server is unavailable"),
                    ],
                },
            )
            inventory_output = tmp / "acceptance-inventory.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_ingest_artifacts.py"),
                    "--manifest",
                    str(manifest_path),
                    "--server-class",
                    "2gpu",
                    "--status",
                    "accepted",
                    "--bundle-gate",
                    str(bundle_path),
                    "--real-artifacts",
                    "--inventory-output",
                    str(inventory_output),
                ],
                cwd=BENCHMARKS.parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["ok"])
            self.assertTrue(inventory_output.exists())


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
        "environment_gaps": ["hardware_unavailable"],
        "next_commands": [f"run Phase 7 bundle gate on {server_class} server"],
    }


def bundle_gate(server_class: str, workloads: tuple[str, ...] = WORKLOADS) -> dict:
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "server_class": server_class,
        "baseline": {"workloads": list(workloads)},
        "turbobus": {"workloads": list(workloads)},
        "comparison": {"ok": True, "workloads": list(workloads)},
        "evidence": {"ok": True, "workloads": list(workloads)},
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
