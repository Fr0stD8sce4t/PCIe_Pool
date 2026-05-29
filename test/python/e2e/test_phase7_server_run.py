from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import phase7_server_run  # noqa: E402


class Phase7ServerRunTest(unittest.TestCase):
    def test_plan_contains_full_artifact_chain_without_route_controls(self) -> None:
        args = default_args()

        plan = phase7_server_run.build_server_run_plan(args)

        self.assertTrue(plan["ok"])
        self.assertEqual(
            [step["name"] for step in plan["steps"]],
            [
                "baseline_paper_validation",
                "baseline_result_check",
                "turbobus_paper_validation",
                "turbobus_result_check",
                "comparison",
                "daemon_evidence",
                "bundle_gate",
                "acceptance_ingest",
            ],
        )
        bundle_outputs = [Path(path).as_posix() for path in plan["steps"][6]["outputs"]]
        self.assertIn("benchmarks/results/phase7/2gpu/bundle-gate.json", bundle_outputs)
        self.assertIn("--real-artifacts", plan["steps"][7]["command"])
        self.assertNotIn("--allow-incomplete-inventory", plan["steps"][7]["command"])
        for step in plan["steps"]:
            self.assertForbiddenRouteControlsAbsent(step["command"])

    def test_plan_uses_saved_profile_for_evidence(self) -> None:
        args = default_args(profile="benchmarks/results/phase7/2gpu/profile.json")

        plan = phase7_server_run.build_server_run_plan(args)

        evidence_command = plan["steps"][5]["command"]
        self.assertIn("--profile", evidence_command)
        self.assertNotIn("--daemon-socket-path", evidence_command)

    def test_plan_can_allow_incomplete_inventory(self) -> None:
        args = default_args(allow_incomplete_inventory=True)

        plan = phase7_server_run.build_server_run_plan(args)

        ingest_command = plan["steps"][7]["command"]
        self.assertIn("--allow-incomplete-inventory", ingest_command)

    def test_cli_dry_run_writes_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan_path = tmp / "plan.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "phase7_server_run.py"),
                    "--server-class",
                    "2gpu",
                    "--output-root",
                    str(tmp / "phase7"),
                    "--vllm-model",
                    "test-model",
                    "--dry-run",
                    "--plan-output",
                    str(plan_path),
                ],
                cwd=BENCHMARKS.parent,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            stdout_plan = json.loads(completed.stdout)
            file_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_plan["steps"], file_plan["steps"])
            self.assertEqual(stdout_plan["server_class"], "2gpu")

    def test_cli_rejects_missing_vllm_model(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARKS / "phase7_server_run.py"),
                "--server-class",
                "2gpu",
                "--dry-run",
            ],
            cwd=BENCHMARKS.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--vllm-model is required", completed.stderr)

    def assertForbiddenRouteControlsAbsent(self, command: list[str]) -> None:
        forbidden = {"--target-gpu", "--relay-gpu", "--relay-gpus", "--mode", "--modes"}
        self.assertTrue(forbidden.isdisjoint(command), command)


def default_args(**overrides: object) -> argparse.Namespace:
    values = {
        "server_class": "2gpu",
        "output_root": "benchmarks/results/phase7",
        "daemon_socket_path": "/tmp/turbobusd.sock",
        "profile": None,
        "correctness": None,
        "workloads": "all",
        "bucket_count": 8,
        "bucket_bytes": 32 * 1024 * 1024,
        "chunk_bytes": 4 * 1024 * 1024,
        "warmup": 1,
        "iterations": 5,
        "vllm_model": "test-model",
        "vllm_job_count": 1,
        "vllm_restore_blocks": 8,
        "vllm_matched_tokens": 128,
        "vllm_prompt_repeat": 64,
        "vllm_enforce_eager": True,
        "allow_incomplete_inventory": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


if __name__ == "__main__":
    unittest.main()
