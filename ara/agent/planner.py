"""Planner: the LLM PROPOSES a plan; deterministic validation DISPOSES.

Validation guarantees (all enforced in code, never by prompting):
  - plan is a JSON object with bounded steps (<= MAX_STEPS)
  - every step uses a legal action type
  - tool_call steps reference existing, contract-allowed tools only
  - no cycles in depends_on
  - exactly zero or one final_answer step, always last
  - goal non-empty; garbage model output => safe fallback plan

A deterministic mini-planner handles arithmetic/clock requests without any LLM,
so the system stays useful (and safe) even in scripted mode.
"""
from __future__ import annotations

import json
import re

from ara.agent.llm import LLMProvider, parse_json_object
from ara.agent.state import Plan, PlanStep, TaskContract
from ara.core.errors import LLMError
from ara.core.logging import get_logger
from ara.core.types import StepAction
from ara.tools.registry import ToolRegistry

log = get_logger("ara.planner")

MAX_STEPS = 8
LEGAL_ACTIONS = {a.value for a in StepAction}
ARITH_RE = re.compile(
    r"(?:(?:what\s+is|calculate|compute|evaluate|how\s+much\s+is)\s+)?"
    r"(-?\d+(?:\.\d+)?(?:\s*[-+*/^%]\s*-?\d+(?:\.\d+)?)+)"
)
CLOCK_RE = re.compile(r"(?i)\b(what(?:'s| is) the (current )?time|current (utc )?time|today'?s date)\b")


class Planner:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    # ------------------------------------------------------------------ prompts
    def _planner_prompt(self, contract: TaskContract, catalog: str, memory_context: str,
                        corpus_size: int = 0) -> str:
        return (
            "You are the PLANNER of a grounded research agent.\n"
            f"DOCUMENT CORPUS: {corpus_size} pages are RETRIEVABLE via the retrieve action.\n"
            f"REQUEST: {contract.user_request}\n"
            f"NORMALIZED GOAL: {contract.normalized_goal or '(derive it)'}\n"
            f"CONSTRAINTS: {contract.constraints or 'none'}\n"
            f"FORBIDDEN ACTIONS: {contract.forbidden_actions or 'none'}\n"
            f"ALLOWED TOOLS (only these may be called): "
            f"{contract.allowed_tools or 'all visible tools'}\n"
            f"TOOLS:\n{catalog or '(no tools)'}\n"
            f"MEMORY:\n{memory_context or '(none)'}\n\n"
            "Return ONLY a JSON object:\n"
            '{"goal": str, "rationale": str, "steps": [{"description": str, '
            '"action": "retrieve|tool_call|reason|final_answer", "tool": str|null, '
            '"args": object|null, "query": str|null, "depends_on": [step indices as ints]}]}\n'
            "Rules: <=8 steps; prefer the fewest needed; retrieve BEFORE answering; "
            "end with one final_answer step; NEVER plan actions that are forbidden."
        )

    # ------------------------------------------------------------------ propose
    def propose(self, contract: TaskContract, catalog: str, memory_context: str = "",
                corpus_size: int = 0) -> Plan:
        try:
            resp = self.llm.complete("PLANNER", self._planner_prompt(contract, catalog, memory_context, corpus_size),
                                     json_mode=True, max_tokens=900)
            raw = parse_json_object(resp.text)
        except LLMError as exc:
            log.warning("planner_llm_failed_using_fallback", extra={"fields": {"error": exc.message}})
            raw = None
        plan = self.validate_and_build(raw, contract) if raw else self.fallback_plan(contract)
        return plan

    # ------------------------------------------------------------------ validate
    def validate_and_build(self, raw: dict, contract: TaskContract) -> Plan:
        """Deterministic plan validation. Anything illegal => ValueError => caller falls back."""
        steps_raw = raw.get("steps")
        if not isinstance(steps_raw, list) or not (1 <= len(steps_raw) <= MAX_STEPS):
            raise ValueError("plan must contain 1..8 steps")
        goal = str(raw.get("goal") or contract.normalized_goal or contract.user_request)[:500]
        plan = Plan(goal=goal, rationale=str(raw.get("rationale", ""))[:500])
        index_to_id: dict[int, str] = {}
        final_seen = False
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                raise ValueError(f"step {i} not an object")
            action = str(s.get("action", "")).lower()
            if action not in LEGAL_ACTIONS:
                raise ValueError(f"step {i}: illegal action '{action}'")
            if action == "final_answer":
                if final_seen or i != len(steps_raw) - 1:
                    raise ValueError("final_answer must be the single last step")
                final_seen = True
            tool = s.get("tool")
            if action == "tool_call":
                if not isinstance(tool, str) or not tool:
                    raise ValueError(f"step {i}: tool_call requires tool name")
                if not self.registry.exists(tool):
                    raise ValueError(f"step {i}: unknown tool '{tool}'")
                if contract.allowed_tools and tool not in contract.allowed_tools:
                    raise ValueError(f"step {i}: tool '{tool}' not allowed by contract")
                if tool in contract.forbidden_actions:
                    raise ValueError(f"step {i}: tool '{tool}' is forbidden")
                if not isinstance(s.get("args", {}), dict):
                    raise ValueError(f"step {i}: args must be an object")
            deps = s.get("depends_on") or []
            if not isinstance(deps, list):
                raise ValueError(f"step {i}: depends_on must be a list")
            deps = [d for d in deps if isinstance(d, int) and 0 <= d < i]
            step = PlanStep(
                description=str(s.get("description", action))[:300],
                action=StepAction(action),
                tool=tool if action == "tool_call" else None,
                args=s.get("args") or {},
                query=(s.get("query") or None),
                instruction=(s.get("instruction") or None),
                depends_on=[index_to_id[d] for d in deps],
            )
            index_to_id[i] = step.step_id
            plan.steps.append(step)
        if not final_seen:
            plan.steps.append(PlanStep(description="Produce the final grounded answer",
                                       action=StepAction.FINAL_ANSWER))
        self._assert_acyclic(plan)
        return plan

    @staticmethod
    def _assert_acyclic(plan: Plan) -> None:
        visited, in_stack = set(), set()

        def dfs(step_id: str) -> None:
            if step_id in in_stack:
                raise ValueError("cyclic plan")
            if step_id in visited:
                return
            in_stack.add(step_id)
            step = next((s for s in plan.steps if s.step_id == step_id), None)
            for dep in (step.depends_on if step else []):
                dfs(dep)
            in_stack.discard(step_id)
            visited.add(step_id)

        for s in plan.steps:
            dfs(s.step_id)

    # ------------------------------------------------------------------ fallback
    def fallback_plan(self, contract: TaskContract) -> Plan:
        """Safe minimal plan when the model cannot propose a valid one."""
        plan = Plan(goal=contract.normalized_goal or contract.user_request,
                    rationale="deterministic fallback plan (model plan invalid/absent)")
        plan.steps.append(PlanStep(description="Search documents for relevant evidence",
                                   action=StepAction.RETRIEVE, query=contract.user_request[:300]))
        plan.steps.append(PlanStep(description="Produce the final grounded answer",
                                   action=StepAction.FINAL_ANSWER))
        return plan

    # ------------------------------------------------------------------ deterministic mini-planner
    def plan(self, contract: TaskContract, catalog: str, memory_context: str = "",
             corpus_size: int = 0) -> Plan:
        """Entry point. Deterministic shortcuts first; LLM proposal otherwise."""
        req = contract.user_request
        calc_forbidden = "calculator" in contract.forbidden_actions
        if ARITH_RE.search(req) and not calc_forbidden and \
                (not contract.allowed_tools or "calculator" in contract.allowed_tools):
            expr = ARITH_RE.search(req).group(1).replace("^", "**")
            plan = Plan(goal=f"compute {expr}", rationale="deterministic arithmetic plan")
            plan.steps.append(PlanStep(description=f"Calculate {expr}", action=StepAction.TOOL_CALL,
                                       tool="calculator", args={"expression": expr}))
            plan.steps.append(PlanStep(description="Produce the final grounded answer",
                                       action=StepAction.FINAL_ANSWER))
            return plan
        if CLOCK_RE.search(req):
            plan = Plan(goal="current time", rationale="deterministic clock plan")
            plan.steps.append(PlanStep(description="Get current UTC time", action=StepAction.TOOL_CALL,
                                       tool="clock", args={}))
            plan.steps.append(PlanStep(description="Produce the final grounded answer",
                                       action=StepAction.FINAL_ANSWER))
            return plan
        return self.propose(contract, catalog, memory_context, corpus_size)
