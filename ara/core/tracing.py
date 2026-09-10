"""Observability: per-run trace recorder (durable spans) + optional OTel export.

The trace is the audit backbone: every plan change, model call, tool call,
guardrail decision, approval and retry is recorded as a span and persisted with
the task so any run can be reconstructed after the fact.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from ara.core.ids import iso_now, new_id


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    started_at: str
    ended_at: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "attributes": self.attributes,
        }


class Trace:
    """In-memory span tree for one agent run; serialized into the task record."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or new_id("run")
        self.spans: list[Span] = []

    @contextlib.contextmanager
    def span(self, name: str, **attrs: Any):
        parent = self.spans[-1].span_id if self.spans else None
        sp = Span(span_id=new_id("step"), parent_id=parent, name=name, started_at=iso_now(), attributes=dict(attrs))
        self.spans.append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.ended_at = iso_now()

    def event(self, name: str, **attrs: Any) -> None:
        """Zero-duration marker (decision logs, guardrail verdicts, retries...)."""
        parent = self.spans[-1].span_id if self.spans else None
        self.spans.append(
            Span(span_id=new_id("step"), parent_id=parent, name=name, started_at=iso_now(), ended_at=iso_now(), attributes=dict(attrs))
        )

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "spans": [s.to_dict() for s in self.spans]}


class CostMeter:
    """Deterministic token/cost accounting. Never trusts model-reported cost."""

    def __init__(self, price_in_per_1k: float, price_out_per_1k: float):
        self.price_in = price_in_per_1k
        self.price_out = price_out_per_1k
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))

    def estimate_tokens(self, text: str) -> int:
        # ~4 chars/token heuristic, only used when a provider reports no usage.
        return max(1, len(text) // 4)

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * self.price_in + self.output_tokens * self.price_out) / 1000

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
