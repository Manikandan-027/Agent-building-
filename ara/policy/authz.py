"""Policy layer implementation: authorization, risk classification, approvals."""
from __future__ import annotations

from dataclasses import dataclass, field

from ara.core.errors import AuthorizationError, PolicyViolation
from ara.core.logging import get_logger
from ara.core.types import RiskLevel

log = get_logger("ara.policy")

# Role -> tool permissions. Least privilege by default.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "user": {
        "documents:read", "documents:write", "memory:read", "memory:write",
        "tools:low", "tools:medium", "chat", "tasks:read", "tasks:create", "approvals:decide",
    },
    "admin": {"*"},
}

# Permissions each builtin tool requires.
TOOL_PERMISSIONS: dict[str, set[str]] = {
    "calculator": {"tools:low"},
    "clock": {"tools:low"},
    "text_stats": {"tools:low"},
    "generate_report_draft": {"tools:medium"},
    "send_report": {"tools:medium"},           # HIGH risk via escalation
    "delete_document": {"tools:medium"},       # CRITICAL via escalation
    "web_search": {"tools:low", "documents:read"},
}

# Dynamic risk escalators: deterministic, argument-driven.
ESCALATION_RULES: list[dict] = [
    {"tool": "send_report", "risk": RiskLevel.HIGH, "reason": "external communication"},
    {"tool": "delete_document", "risk": RiskLevel.CRITICAL, "reason": "irreversible deletion"},
    {"tool": "sql_query", "risk": RiskLevel.HIGH, "reason": "database mutation potential"},
]
# Argument-level escalators (applied to any tool)
ARG_ESCALATORS: list[dict] = [
    {"pattern": r"(?i)\b(drop|truncate)\s+table\b", "risk": RiskLevel.CRITICAL, "reason": "destructive SQL"},
    {"pattern": r"(?i)\bdelete\b", "risk": RiskLevel.HIGH, "reason": "delete in arguments"},
]


@dataclass
class Principal:
    """Authenticated actor."""

    user_id: str
    tenant_id: str
    role: str = "user"
    permissions: set[str] = field(default_factory=set)

    @classmethod
    def for_role(cls, user_id: str, tenant_id: str, role: str) -> "Principal":
        return cls(user_id=user_id, tenant_id=tenant_id, role=role,
                   permissions=set(ROLE_PERMISSIONS.get(role, set())))

    def has(self, permission: str) -> bool:
        return "*" in self.permissions or permission in self.permissions


class PolicyEngine:
    """Deterministic authorization + risk decisions. Never consults an LLM."""

    def __init__(self, extra_rules: list[dict] | None = None):
        self.escalation_rules = list(ESCALATION_RULES) + (extra_rules or [])

    def authorize_tool(self, principal: Principal, tool_name: str, base_risk: RiskLevel,
                       spec_permissions: set[str] | None = None) -> RiskLevel:
        """Returns the effective risk after escalation; raises if unauthorized."""
        perms = spec_permissions or TOOL_PERMISSIONS.get(tool_name, {"tools:low"})
        if not any(principal.has(p) for p in perms):
            log.warning("tool_unauthorized", extra={"fields": {"tool": tool_name, "role": principal.role}})
            raise AuthorizationError(f"role '{principal.role}' is not authorized to use tool '{tool_name}'")
        effective = base_risk
        for rule in self.escalation_rules:
            if rule["tool"] == tool_name and rule["risk"].rank > effective.rank:
                effective = rule["risk"]
        return effective

    def classify_args_risk(self, tool_name: str, args: dict) -> tuple[RiskLevel, list[str]]:
        """Argument-driven escalation (works for MCP tools too)."""
        import json as _json

        blob = _json.dumps(args, default=str)
        effective, reasons = RiskLevel.LOW, []
        for rule in self.escalation_rules:
            if rule["tool"] == tool_name:
                effective = max(effective, rule["risk"], key=lambda r: r.rank)
                reasons.append(rule["reason"])
        for rule in ARG_ESCALATORS:
            import re

            if re.search(rule["pattern"], blob):
                if rule["risk"].rank > effective.rank:
                    effective = rule["risk"]
                reasons.append(rule["reason"])
        return effective, reasons

    def check_forbidden_actions(self, contract, tool_name: str) -> None:
        if tool_name in contract.forbidden_actions:
            raise PolicyViolation(f"tool '{tool_name}' is forbidden by the task contract")
        if contract.allowed_tools and tool_name not in contract.allowed_tools:
            raise PolicyViolation(f"tool '{tool_name}' is not in the task allowlist")

    def check_risk_floor(self, contract, effective_risk: RiskLevel) -> None:
        if effective_risk.rank > contract.risk_level.rank and contract.risk_level != RiskLevel.LOW:
            # contract risk acts as a ceiling for autonomous execution
            raise PolicyViolation(
                f"tool risk {effective_risk.value} exceeds contract risk ceiling {contract.risk_level.value}"
            )


class ApprovalRequiredSignal(Exception):
    """Internal control-flow: carries the created approval request."""

    def __init__(self, approval_id: str, tool: str, risk: RiskLevel, reason: str):
        self.approval_id = approval_id
        self.tool = tool
        self.risk = risk
        self.reason = reason
        super().__init__(f"approval required for {tool}: {reason}")
