"""Budgets & loop protection. Enforced deterministically — an LLM can never
extend its own budget, disable checks, or continue past a stop condition."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ara.core.errors import BudgetExceeded
from ara.core.tracing import CostMeter


@dataclass
class BudgetState:
    max_iterations: int
    max_tool_calls: int
    max_llm_calls: int
    max_tokens: int
    max_cost_usd: float
    deadline: float                     # monotonic deadline
    cost: CostMeter
    iterations: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    stop_reason: str | None = None
    events: list[dict] = field(default_factory=list)

    @classmethod
    def from_contract(cls, budget, price_in: float, price_out: float) -> "BudgetState":
        return cls(
            max_iterations=budget.max_iterations,
            max_tool_calls=budget.max_tool_calls,
            max_llm_calls=budget.max_llm_calls,
            max_tokens=budget.max_tokens,
            max_cost_usd=budget.max_cost_usd,
            deadline=time.monotonic() + budget.time_budget_s,
            cost=CostMeter(price_in, price_out),
        )

    # -- called BEFORE each iteration / call --------------------------------
    def check_iteration(self) -> None:
        self.iterations += 1
        if self.iterations > self.max_iterations:
            self.stop_reason = "max_iterations"
            raise BudgetExceeded(f"iteration budget exceeded (>{self.max_iterations})")
        self._check_clock()

    def pre_llm(self) -> None:
        self.llm_calls += 1
        if self.llm_calls > self.max_llm_calls:
            self.stop_reason = "max_llm_calls"
            raise BudgetExceeded(f"LLM call budget exceeded (>{self.max_llm_calls})")

    def pre_tool(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            self.stop_reason = "max_tool_calls"
            raise BudgetExceeded(f"tool call budget exceeded (>{self.max_tool_calls})")

    def after_llm(self, input_tokens: int, output_tokens: int) -> None:
        self.cost.add(input_tokens, output_tokens)
        if self.cost.input_tokens + self.cost.output_tokens > self.max_tokens:
            self.stop_reason = "max_tokens"
            raise BudgetExceeded(f"token budget exceeded (>{self.max_tokens})")
        if self.cost.cost_usd > self.max_cost_usd:
            self.stop_reason = "max_cost"
            raise BudgetExceeded(f"cost budget exceeded (>${self.max_cost_usd:.2f})")

    def _check_clock(self) -> None:
        if time.monotonic() > self.deadline:
            self.stop_reason = "time_budget"
            raise BudgetExceeded(f"time budget exceeded (>{self.deadline:+.0f}s window)")

    @property
    def time_remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations, "tool_calls": self.tool_calls, "llm_calls": self.llm_calls,
            "tokens": self.cost.input_tokens + self.cost.output_tokens, "cost_usd": round(self.cost.cost_usd, 6),
            "stop_reason": self.stop_reason,
        }

    # -- durable suspend/resume -------------------------------------------
    def serialize(self) -> dict:
        return {
            "max_iterations": self.max_iterations, "max_tool_calls": self.max_tool_calls,
            "max_llm_calls": self.max_llm_calls, "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd, "time_remaining_s": round(self.time_remaining_s, 3),
            "iterations": self.iterations, "tool_calls": self.tool_calls, "llm_calls": self.llm_calls,
            "input_tokens": self.cost.input_tokens, "output_tokens": self.cost.output_tokens,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def restore(cls, data: dict, price_in: float, price_out: float) -> "BudgetState":
        b = cls(
            max_iterations=data["max_iterations"], max_tool_calls=data["max_tool_calls"],
            max_llm_calls=data["max_llm_calls"], max_tokens=data["max_tokens"],
            max_cost_usd=data["max_cost_usd"], deadline=time.monotonic() + float(data["time_remaining_s"]),
            cost=CostMeter(price_in, price_out),
        )
        b.iterations = data.get("iterations", 0)
        b.tool_calls = data.get("tool_calls", 0)
        b.llm_calls = data.get("llm_calls", 0)
        b.stop_reason = data.get("stop_reason")
        b.cost.input_tokens = data.get("input_tokens", 0)
        b.cost.output_tokens = data.get("output_tokens", 0)
        return b
