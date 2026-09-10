#!/usr/bin/env python3
"""Submit a fine-tuning job to an OpenAI-compatible endpoint.

SAFETY: dry-run by default. It validates the dataset and prints exactly what
would be executed. Pass --execute (and have OPENAI_API_KEY set) to run for real.

    # 1. build data
    python scripts/prepare_finetune_data.py --tenant ten_dev
    # 2. validate (dry run)
    python scripts/run_finetune.py --dataset data/ft_planner.jsonl \
        --base-model gpt-4.1-mini --suffix ara-planner
    # 3. execute
    python scripts/run_finetune.py --dataset data/ft_planner.jsonl \
        --base-model gpt-4.1-mini --suffix ara-planner --execute --wait

Then set in .env:  OPENAI_MODEL_PLANNER=ft:...:ara-planner:<id>  (printed on success)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from ara.agent.training_data import validate_jsonl  # noqa: E402
from ara.core.errors import LLMError  # noqa: E402

FINE_TUNABLE_HINT = ("gpt-4.1-mini, gpt-4.1, gpt-4o-mini (OpenAI), or any base model your "
                     "OpenAI-compatible endpoint supports for /fine_tuning/jobs")


def submit(base_url: str, api_key: str, dataset_path: str, base_model: str, suffix: str,
           epochs: int, wait: bool, execute: bool) -> int:
    stats = validate_jsonl(dataset_path)
    print(f"dataset: {dataset_path} -> {stats['lines']} lines, "
          f"{stats['total_chars']} chars, {len(stats['errors'])} validation errors")
    if stats["errors"]:
        for err in stats["errors"][:10]:
            print("  !", err)
        return 2
    if stats["lines"] < 50:
        print(f"WARNING: only {stats['lines']} examples; OpenAI recommends >=50 "
              "(a few hundred for stable behavior improvements)")

    if not execute:
        print(json.dumps({
            "dry_run": True,
            "would_upload": dataset_path,
            "would_post": f"{base_url}/fine_tuning/jobs",
            "body": {"training_file": "<file-id>", "model": base_model, "suffix": suffix,
                     "hyperparameters": {"n_epochs": epochs}},
            "note": "pass --execute (with OPENAI_API_KEY set) to run",
        }, indent=1))
        return 0

    import httpx

    headers = {"Authorization": f"Bearer {api_key}"}
    with open(dataset_path, "rb") as fh:
        resp = httpx.post(f"{base_url}/files", headers=headers,
                          files={"file": (Path(dataset_path).name, fh)},
                          data={"purpose": "fine-tune"}, timeout=120)
    if resp.status_code >= 400:
        print(f"upload failed: {resp.status_code} {resp.text[:300]}")
        return 1
    file_id = resp.json()["id"]
    print(f"uploaded file {file_id}")

    resp = httpx.post(f"{base_url}/fine_tuning/jobs", headers=headers,
                      json={"training_file": file_id, "model": base_model, "suffix": suffix,
                            "hyperparameters": {"n_epochs": epochs}}, timeout=60)
    if resp.status_code >= 400:
        print(f"job creation failed: {resp.status_code} {resp.text[:300]}")
        return 1
    job = resp.json()
    print(f"job {job.get('id')} status={job.get('status')}")
    if not wait:
        return 0

    while True:
        time.sleep(20)
        resp = httpx.get(f"{base_url}/fine_tuning/jobs/{job['id']}", headers=headers, timeout=30)
        data = resp.json()
        status = data.get("status")
        print(f"  status: {status}")
        if status in ("succeeded", "failed", "cancelled"):
            if status == "succeeded":
                model = data.get("fine_tuned_model")
                print(json.dumps({"fine_tuned_model": model,
                                  "next": f"set OPENAI_MODEL_PLANNER={model} in .env"}, indent=1))
                return 0
            print(json.dumps(data, indent=1)[:1500])
            return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-model", default="gpt-4.1-mini", help=FINE_TUNABLE_HINT)
    ap.add_argument("--suffix", required=True, help="model name suffix, e.g. ara-planner")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--execute", action="store_true", help="actually run (default: dry-run)")
    ap.add_argument("--wait", action="store_true", help="poll until the job finishes")
    args = ap.parse_args()

    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if args.execute and not api_key:
        print("OPENAI_API_KEY is required with --execute")
        return 1
    try:
        return submit(base_url, api_key, args.dataset, args.base_model, args.suffix,
                      args.epochs, args.wait, args.execute)
    except LLMError as exc:
        print(f"error: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
