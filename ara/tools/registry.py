"""Tool registry: every tool is a fully-specified, typed contract.

A tool cannot be registered without: input/output schema, permission set, risk
classification, timeout, retry policy, idempotency + side-effect classification,
and audit requirement. Registry is the single source of truth for what the agent
is allowed to see and call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ara.core.errors import NotFoundError, ValidationError
from ara.core.types import RiskLevel


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    handler: Callable[[dict, dict], Any]                 # (args, ctx) -> result
    risk: RiskLevel = RiskLevel.LOW
    permissions: set[str] = field(default_factory=lambda: {"tools:low"})
    timeout_s: float = 10.0
    max_attempts: int = 1
    retryable: bool = False                              # side-effecting tools default to no-retry
    idempotent: bool = False
    side_effects: str = "none"                           # none | local_write | external | destructive
    audit: bool = True
    trusted_output: bool = False                         # tool outputs trusted as evidence? (rare)
    fallback_tool: str | None = None
    source: str = "builtin"                              # builtin | mcp:<server>

    def validate_spec(self) -> None:
        if not self.name or not self.description:
            raise ValidationError("tool needs name and description")
        if not isinstance(self.input_schema, dict) or not isinstance(self.output_schema, dict):
            raise ValidationError("tool schemas must be JSON Schema dicts")
        if self.side_effects == "destructive" and self.risk.rank < RiskLevel.CRITICAL.rank:
            raise ValidationError(f"{self.name}: destructive side effects require CRITICAL risk")
        if self.side_effects == "external" and self.risk.rank < RiskLevel.HIGH.rank:
            raise ValidationError(f"{self.name}: external side effects require HIGH risk")
        if self.side_effects != "none" and self.idempotent is False and self.max_attempts > 1:
            # blind retries of non-idempotent side effects are forbidden
            raise ValidationError(f"{self.name}: non-idempotent side-effecting tools must not auto-retry")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        spec.validate_spec()
        if spec.name in self._tools:
            raise ValidationError(f"tool '{spec.name}' already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise NotFoundError(f"unknown tool '{name}'")
        return self._tools[name]

    def exists(self, name: str) -> bool:
        return name in self._tools

    def visible_to(self, allowed_tools: list[str] | None, permissions: set[str]) -> list[ToolSpec]:
        """The agent only ever sees tools authorized for the principal + contract."""
        out = []
        for spec in self._tools.values():
            if allowed_tools and spec.name not in allowed_tools:
                continue
            if "*" in permissions or (spec.permissions & permissions):
                out.append(spec)
        return out

    def catalog_for_prompt(self, specs: list[ToolSpec]) -> str:
        lines = []
        for s in specs:
            args = ", ".join(f"{k}:{v.get('type', '?')}" for k, v in s.input_schema.get("properties", {}).items())
            lines.append(f"- {s.name}({args}): {s.description} [risk={s.risk.value}]")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)
