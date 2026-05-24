from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import model_loading
import paper_validation
import training_offload


def compact_summary_for_result(data: dict) -> str:
    if not isinstance(data, dict):
        raise ValueError("benchmark JSON must be an object")

    config = data.get("config", {}) or {}
    summary = data.get("summary", {}) or {}
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("benchmark JSON has invalid config or summary fields")

    if isinstance(data.get("workloads"), list) and isinstance(config.get("workloads"), list):
        return paper_validation.compact_summary(data)
    if "source_buffer_id" in config and "destination_buffer_id" in config:
        return model_loading.compact_summary(data)
    if "cpu_buffer_id" in config and "gpu_buffer_id" in config:
        if "prefetch" in summary and "offload" in summary:
            return training_offload.compact_summary(data)

    raise ValueError("unsupported daemon-first benchmark JSON shape")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact benchmark summary")
    parser.add_argument("json_path", help="benchmark JSON result path")
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    sys.stdout.write(compact_summary_for_result(data))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
