#!/usr/bin/env python3
"""Run the ARA evaluation suite and write evals/latest_report.md + latest_results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evals.runner import format_report, run_dataset  # noqa: E402

DATASETS = sorted((Path(__file__).parents[1] / "evals" / "datasets").glob("*.yaml"))


def main() -> int:
    report = run_dataset(DATASETS)
    out_dir = Path(__file__).parents[1] / "evals"
    (out_dir / "latest_report.md").write_text(format_report(report))
    (out_dir / "latest_results.json").write_text(json.dumps(report, indent=1, default=str))
    print(format_report(report))
    m = report["metrics"]
    print(f"Results written to {out_dir/'latest_report.md'}")
    return 0 if m["passed"] == m["total_scenarios"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
