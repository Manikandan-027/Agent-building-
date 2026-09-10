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
