"""ID helpers and time utilities (UTC everywhere)."""
from __future__ import annotations

import datetime as dt
import uuid

PREFIXES = {
    "task": "task",
    "run": "run",
    "step": "step",
    "ev": "ev",
    "doc": "doc",
    "page": "pg",
    "msg": "msg",
    "mem": "mem",
    "apr": "apr",
    "call": "call",
    "claim": "claim",
    "plan": "plan",
    "user": "user",
    "conv": "conv",
    "key": "key",
    "tenant": "ten",
    "eval": "eval",
}


def new_id(kind: str) -> str:
    prefix = PREFIXES.get(kind, kind)
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat(timespec="milliseconds")
