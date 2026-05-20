from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bandwidth_pool import compact_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact benchmark summary")
    parser.add_argument("json_path", help="benchmark JSON result path")
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    sys.stdout.write(compact_summary(data))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
