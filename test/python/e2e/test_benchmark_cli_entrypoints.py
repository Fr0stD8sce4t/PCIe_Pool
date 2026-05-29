from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]


class BenchmarkCliEntrypointTest(unittest.TestCase):
    def test_benchmark_help_runs_from_repo_root_without_pythonpath(self) -> None:
        for script in (
            "benchmarks/paper_validation.py",
            "benchmarks/model_loading.py",
            "benchmarks/training_offload.py",
        ):
            with self.subTest(script=script):
                completed = subprocess.run(
                    [sys.executable, script, "--help"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
