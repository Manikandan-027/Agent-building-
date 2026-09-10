"""Guardrails: input, tool, output.

Input guardrail   — malformed/unsafe requests, injection in user input, unsupported task types.
Tool guardrail    — thin declarative wrapper enforcing the ToolPipeline gates.
Output guardrail  — hallucinated/unsupported claims, invalid citations, leaked secrets,
                    fabricated tool claims, policy violations. Last line of defense.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ara.core.errors import PolicyViolation, UnsafeContentError
from ara.core.logging import get_logger
from ara.core.types import RiskLevel
from ara.guardrails.injection import InjectionDetector

log = get_logger("ara.guardrails")

_SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{20,}", "openai-style api key"),
    (r"(?i)aws_access_key_id\s*[=:]\s*AKIA[0-9A-Z]{16}", "aws access key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
    (r"(?i)bearer\s+[a-z0-9._-]{25,}", "bearer token"),
    (r"(?i)(api[_-]?key|secret|password)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "embedded credential"),
]

MAX_INPUT_CHARS = 20_000
MAX_INPUT_LINES = 400


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: str = ""
    sanitized: str | None = None
    metadata: dict | None = None


class InputGuardrail:
    def __init__(self, detector: InjectionDetector, forbidden_topics: list[str] | None = None,
                 supported_intents: set[str] | None = None):
        self.detector = detector
        self.forbidden_topics = forbidden_topics or [r"(?i)\b(build|buy)\s+(a\s+)?(bomb|weapon)\b",
                                                     r"(?i)\bchild\s*(porn|abuse)\b",
                                                     r"(?i)\b(cyanide|sarin)\b"]
        self.supported_intents = supported_intents  # None = all intents supported

    def check(self, user_input: str) -> GuardrailVerdict:
        if not user_input or not user_input.strip():
            return GuardrailVerdict(False, "empty request")
        if len(user_input) > MAX_INPUT_CHARS or user_input.count("\n") > MAX_INPUT_LINES:
            sanitized = user_input[:MAX_INPUT_CHARS]
            return GuardrailVerdict(True, "truncated to limits", sanitized=sanitized)
        for pattern in self.forbidden_topics:
            if re.search(pattern, user_input):
                raise UnsafeContentError("request rejected by safety policy")
        scan = self.detector.scan(user_input)
        if scan.flagged:
            # user input containing injection-style payloads is suspicious but the user
            # is the instruction source; we sanitize rather than refuse outright.
            log.warning("input_injection_pattern", extra={"fields": scan.to_dict()})
            return GuardrailVerdict(True, "user input contains injection-like patterns (sanitized)",
                                    sanitized=scan.normalized_text, metadata={"scan": scan.to_dict()})
        if self.supported_intents is not None:
            intent = self._classify_intent(user_input)
            if intent not in self.supported_intents:
                raise PolicyViolation(f"unsupported task type '{intent}'")
        return GuardrailVerdict(True, "ok")

    @staticmethod
    def _classify_intent(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ("summarize", "summary", "tl;dr")):
            return "summarize"
        if any(k in t for k in ("compare", "versus", " vs ", "difference")):
            return "compare"
        if any(k in t for k in ("find", "search", "what", "which", "who", "when", "how much", "how many")):
            return "research"
        return "general"


class OutputGuardrail:
    def __init__(self, detector: InjectionDetector):
        self.detector = detector

    def check(self, answer: str, *, citations_used: list[str], valid_evidence_ids: set[str],
              tool_call_ids: set[str]) -> GuardrailVerdict:
        problems: list[str] = []

        # 1. hallucinated citations
        for cite in citations_used:
            if cite not in valid_evidence_ids:
                problems.append(f"citation '{cite}' does not reference real evidence")

        # 2. fabricated tool execution claims (mentions a call id never executed)
        for call_id in re.findall(r"\bcall_[a-z0-9]+\b", answer):
            if call_id not in tool_call_ids:
                problems.append(f"answer claims tool call '{call_id}' that never executed")

        # 3. secret leakage
        for pattern, label in _SECRET_PATTERNS:
            if re.search(pattern, answer):
                problems.append(f"answer leaks {label}")

        # 4. injection echo (answer repeating untrusted directives as its own)
        scan = self.detector.scan(answer)
        if scan.flagged and scan.score >= 0.8:
            problems.append("answer itself contains injection-grade content")

        if problems:
            log.warning("output_guardrail_block", extra={"fields": {"problems": problems}})
            return GuardrailVerdict(False, "; ".join(problems))
        return GuardrailVerdict(True, "ok")

    def sanitize(self, answer: str) -> str:
        """Redact secrets but keep the answer usable (defense in depth)."""
        out = answer
        for pattern, _label in _SECRET_PATTERNS:
            out = re.sub(pattern, "[REDACTED]", out)
        return out


class ToolGuardrail:
    """Declarative pre-flight mirroring the pipeline (used by planner validation and APIs)."""

    def __init__(self, registry, policy_engine: PolicyEngineLike):
        self.registry = registry
        self.policy = policy_engine

    def preflight(self, principal, tool_name: str, args: dict, contract) -> RiskLevel:
        spec = self.registry.get(tool_name)
        base = self.policy.authorize_tool(principal, tool_name, spec.risk, spec.permissions)
        arg_risk, _ = self.policy.classify_args_risk(tool_name, args)
        effective = max(base, spec.risk, arg_risk, key=lambda r: r.rank)
        self.policy.check_forbidden_actions(contract, tool_name)
        self.policy.check_risk_floor(contract, effective)
        return effective


class PolicyEngineLike:
    """Structural type for ToolGuardrail to avoid import cycles."""
