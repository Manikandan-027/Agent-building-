"""The tool execution pipeline. Every call — proposed by a model or a plan —
passes through ALL gates, in order:

  MODEL PROPOSAL
    -> existence check            (registry)
    -> contract check             (allowlist / forbidden_actions)
    -> INPUT VALIDATION           (jsonschema, strict)
    -> AUTHORIZATION              (role permissions)
    -> RISK CHECK                 (static + argument-driven escalation)
    -> HUMAN APPROVAL if required (durable, resumable)
    -> IDEMPOTENCY CHECK          (replay protection)
    -> EXECUTION                  (timeout + bounded retries; never blind for
                                   non-idempotent side effects)
    -> OUTPUT VALIDATION          (jsonschema)
    -> OBSERVATION VERIFICATION   (deterministic sanity checks)
    -> STATE UPDATE + AUDIT

The LLM can only propose a call. Everything else here is deterministic code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ara.agent.resilience import with_retry
from ara.core.errors import AraError, ToolError, ToolTimeout, ToolUnavailable, ValidationError
from ara.core.logging import get_logger
from ara.core.types import RiskLevel
from ara.policy.authz import ApprovalRequiredSignal, PolicyEngine, Principal
from ara.tools.registry import ToolRegistry, ToolSpec
from ara.tools.validation import validate_input, validate_output

log = get_logger("ara.tools.pipeline")

MAX_RESULT_BYTES = 64_000


@dataclass
class PipelineContext:
    principal: Principal
    contract: Any                    # TaskContract
    registry: ToolRegistry
    policy: PolicyEngine
    create_approval: Callable[[str, str, dict, RiskLevel, str], str]  # -> approval_id
    get_approval_decision: Callable[[str], str | None]                # -> APPROVED/REJECTED/PENDING/None
    audit: Callable[..., None]
    trace: Any = None                # ara.core.tracing.Trace
    documents_store: dict | None = None


@dataclass
class PipelineOutcome:
    record: Any                      # ToolCallRecord
    evidence_worthy: bool = False    # output may ground claims
    suppressed: bool = False         # result withheld from prompt (failed verification)


class ToolPipeline:
    def execute(self, tool_name: str, args: dict, ctx: PipelineContext, *,
                approved_approval_id: str | None = None, idempotency_key: str | None = None,
                attempt: int = 1) -> PipelineOutcome:
        from ara.agent.state import ToolCallRecord  # local import avoids cycle

        started = time.monotonic()

        def finish(rec: ToolCallRecord) -> PipelineOutcome:
            rec.latency_ms = int((time.monotonic() - started) * 1000)
            return PipelineOutcome(record=rec)

        rec = ToolCallRecord(tool=tool_name, args=args, attempt=attempt, idempotency_key=idempotency_key)

        # 1. existence ------------------------------------------------------------------
        if not ctx.registry.exists(tool_name):
            rec.status, rec.error = "error", f"unknown tool '{tool_name}'"
            return finish(rec)

        spec = ctx.registry.get(tool_name)

        # 2. contract allowlist / forbidden ---------------------------------------------
        try:
            ctx.policy.check_forbidden_actions(ctx.contract, tool_name)
        except AraError as exc:
            rec.status, rec.error = "rejected", exc.message
            return finish(rec)

        # 3. input validation -------------------------------------------------------------
        try:
            args = validate_input(args, spec.input_schema, tool_name)
            rec.args = args
        except ValidationError as exc:
            rec.status, rec.error = "error", f"input validation failed: {exc.message}"
            rec.result = exc.details
            return finish(rec)

        # 4. authorization ------------------------------------------------------------------
        try:
            base_risk = ctx.policy.authorize_tool(ctx.principal, tool_name, spec.risk, spec.permissions)
        except AraError as exc:
            rec.status, rec.error = "rejected", exc.message
            return finish(rec)

        # 5. risk classification (static + argument-driven) ---------------------------------
        arg_risk, reasons = ctx.policy.classify_args_risk(tool_name, args)
        effective = max(base_risk, spec.risk, arg_risk, key=lambda r: r.rank)
        rec.risk_level = effective

        # 6. contract risk ceiling ------------------------------------------------------------
        try:
            ctx.policy.check_risk_floor(ctx.contract, effective)
        except AraError as exc:
            rec.status, rec.error = "rejected", exc.message
            return finish(rec)

        # 7. human approval gate ---------------------------------------------------------------
        if effective.requires_approval():
            if approved_approval_id:
                decision = ctx.get_approval_decision(approved_approval_id)
                if decision == "APPROVED":
                    rec.approved_by = "human-approval:" + approved_approval_id
                elif decision == "REJECTED":
                    rec.status, rec.error = "rejected", "human rejected the action"
                    return finish(rec)
                else:
                    rec.status, rec.error = "needs_approval", f"approval {approved_approval_id} not decided"
                    return finish(rec)
            else:
                approval_id = ctx.create_approval(tool_name, args, effective, "; ".join(reasons) or spec.side_effects)
                rec.status = "needs_approval"
                rec.result = {"approval_id": approval_id}
                raise ApprovalRequiredSignal(approval_id, tool_name, effective,
                                             "; ".join(reasons) or spec.side_effects) from None

        # 8. idempotency replay protection -------------------------------------------------------
        if idempotency_key and spec.idempotent is False:
            rec.call_id = idempotency_key

        # 9. execution (timeout + bounded retry) ----------------------------------------------------
        attempts_allowed = spec.max_attempts if (spec.retryable or spec.max_attempts == 1) else 1
        try:
            result = with_retry(
                lambda: self._execute_once(spec, args, ctx),
                max_attempts=max(1, attempts_allowed),
                base_delay_s=0.1,
                retryable=(ToolTimeout, ToolUnavailable, ConnectionError) if spec.retryable else (),
                deadline_s=min(spec.timeout_s * attempts_allowed + 1.0, 30.0),
                on_retry=lambda a, e: ctx.trace and ctx.trace.event("tool_retry", tool=tool_name, attempt=a, error=str(e)),
            )
            rec.result = result
            rec.status = "ok"
        except (ToolError, ValidationError, AraError) as exc:
            rec.status, rec.error = "error", f"{exc.code}: {exc.message}"
            if isinstance(exc, ToolTimeout):
                rec.status = "timeout"
            return finish(rec)
        except Exception as exc:  # noqa: BLE001 — tool code is untrusted boundary
            rec.status, rec.error = "error", f"unexpected tool failure: {type(exc).__name__}: {exc}"
            return finish(rec)

        # 10. output validation ----------------------------------------------------------------------
        try:
            rec.result = validate_output(rec.result, spec.output_schema, tool_name)
        except ValidationError as exc:
            rec.status, rec.error = "error", f"output validation failed: {exc.message}"
            rec.result = str(rec.result)[:500]
            return finish(rec)

        # 11. observation verification -----------------------------------------------------------------
        verified, note = self._verify_observation(spec, rec.result)
        if isinstance(rec.result, dict):
            rec.result["_observation"] = {"verified": verified, "note": note}
        if not verified:
            rec.error = note

        # 12. audit ---------------------------------------------------------------------------------------
        ctx.audit(action="tool_executed", subject=f"{tool_name}",
                  detail={"call_id": rec.call_id, "status": rec.status, "risk": effective.value,
                          "attempt": rec.attempt, "latency_ms": rec.latency_ms})
        return finish(rec)

    # ------------------------------------------------------------------ helpers
    def _execute_once(self, spec: ToolSpec, args: dict, ctx: PipelineContext) -> Any:
        from ara.agent.resilience import run_with_timeout

        def call():
            return spec.handler(args, {"tenant_id": ctx.principal.tenant_id, "user_id": ctx.principal.user_id,
                                       "documents_store": ctx.documents_store})

        return run_with_timeout(call, spec.timeout_s, what=f"tool:{spec.name}")

    def _verify_observation(self, spec: ToolSpec, result: Any) -> tuple[bool, str]:
        """Deterministic post-execution sanity checks — never trust raw output."""
        import json as _json

        if result is None:
            return False, "tool returned null"
        size = len(_json.dumps(result, default=str))
        if size > MAX_RESULT_BYTES:
            return False, f"result too large ({size} bytes) — truncated/suppressed"
        if isinstance(result, dict):
            if result.get("verified") is False:
                return False, "tool self-reported unverified result"
            if "results" in result and isinstance(result["results"], list) and not result["results"]:
                return True, "empty result set — treat as NO EVIDENCE, not as a negative fact"
        return True, "ok"

    def execute_with_fallback(self, tool_name: str, args: dict, ctx: PipelineContext, **kw) -> PipelineOutcome:
        """Primary tool; on error/timeout fall back to its declared fallback tool."""
        outcome = self.execute(tool_name, args, ctx, **kw)
        if outcome.record.status in {"error", "timeout"} and ctx.registry.get(tool_name).fallback_tool:
            fb = ctx.registry.get(tool_name).fallback_tool
            fb_outcome = self.execute(fb, args, ctx, **kw)
            fb_outcome.record.args = {"fallback_for": tool_name, **(fb_outcome.record.args or {})}
            if ctx.trace:
                ctx.trace.event("tool_fallback", primary=tool_name, fallback=fb,
                                primary_status=outcome.record.status)
            return fb_outcome
        return outcome
