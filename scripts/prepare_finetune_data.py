#!/usr/bin/env python3
"""Prepare fine-tuning datasets from ARA's verified task episodes.

Only high-quality episodes (verified answers, all claims supported, no refusals)
are exported. Usage:

    python scripts/prepare_finetune_data.py --tenant ten_dev [--db sqlite:///./data/ara.db]
    # writes data/ft_planner.jsonl, data/ft_answerer.jsonl + prints a report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from ara.agent.training_data import export_training_data, validate_jsonl, write_jsonl  # noqa: E402
from ara.core.config import get_settings  # noqa: E402
from ara.db import UnitOfWork  # noqa: E402
from ara.db.database import Database  # noqa: E402
from ara.evidence import EvidenceManager  # noqa: E402
from ara.guardrails import InjectionDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="ten_dev")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    settings = get_settings()
    db = Database(args.db or settings.database_url)
    db.connect()
    uow = UnitOfWork(db)
    em = EvidenceManager(InjectionDetector(settings.injection_threshold))

    planners, answerers, report = export_training_data(uow, em, tenant_id=args.tenant)
    out = Path(args.out_dir)
    p1 = out / "ft_planner.jsonl"
    p2 = out / "ft_answerer.jsonl"
    n1 = write_jsonl(str(p1), planners) if planners else 0
    n2 = write_jsonl(str(p2), answerers) if answerers else 0

    print(json.dumps(report.to_dict(), indent=1))
    for path, n in ((p1, n1), (p2, n2)):
        if n:
            v = validate_jsonl(str(path))
            print(f"{path}: {n} examples, {v['lines']} lines, "
                  f"{v['total_chars']} chars, {len(v['errors'])} errors")
            for err in v["errors"][:5]:
                print("  !", err)
        else:
            print(f"{path}: 0 examples (run some tasks first, e.g. POST /chat)")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
