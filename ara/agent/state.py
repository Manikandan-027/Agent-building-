"""Explicit agent state + task contract.

The runtime never relies exclusively on raw conversation history: `AgentState`
is the single durable source of truth for a task, persisted as JSON on every
transition so the workflow can suspend (approval) and resume safely.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ara.core.ids import new_id
from ara.core.types import RiskLevel, StepAction, StepStatus, TaskStatus


class ExecutionBudget(BaseModel):
    max_iterations: int = 40
    max_tool_calls: int = 24
    max_llm_calls: int = 16
    max_tokens: int = 120_000
    max_cost_usd: float = 2.0
    time_budget_s: float = 120.0
    max_replans: int = 3


class VerificationRequirements(BaseModel):
    require_citations: bool = True
    min_supporting_evidence: int = 1
    forbid_unverified_numbers: bool = True
    forbid_unverified_dates: bool = True
    refuse_on_unsupported_claims: bool = True


class TaskContract(BaseModel):
    """Every autonomous task carries an explicit, validated contract."""

    task_id: str = Field(default_factory=lambda: new_id("task"))
    user_request: str
    normalized_goal: str = ""
    constraints: list[str] = Field(default_factory=list)
    required_information: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)   # empty = all authorized tools
    forbidden_actions: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    success_conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    execution_budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    verification_requirements: VerificationRequirements = Field(default_factory=VerificationRequirements)


class ToolCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: new_id("call"))
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "timeout", "needs_approval", "rejected"] = "ok"
    result: Any = None
    error: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    attempt: int = 1
    latency_ms: int = 0
    approved_by: str | None = None
    idempotency_key: str | None = None


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    document_id: str
    page: int = 0
    content_type: str = "visual_document_page"  # | text_snippet | tool_output | calculation
    content: str = ""
    relevance_score: float = 0.0
    source_authority: str = "unknown"            # high | medium | low | unknown
    source_trust: str = "untrusted"              # ContentTrust of the origin
    retrieval_timestamp: str = ""
    tenant_id: str = ""
    injection_scan: dict = Field(default_factory=dict)  # {flagged, score, reasons[]}
    meta: dict = Field(default_factory=dict)     # document_version, source, timestamps...

    @property
    def citation(self) -> str:
        return f"[{self.evidence_id} doc={self.document_id} p{self.page}]"


class PlanStep(BaseModel):
    step_id: str = Field(default_factory=lambda: new_id("step"))
    description: str
    action: StepAction
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    query: str | None = None            # for retrieve steps
    instruction: str | None = None      # for reason steps
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result_summary: str | None = None


class Plan(BaseModel):
    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str = ""
    version: int = 1

    def pending(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]


class VerificationReport(BaseModel):
    answer_supported: bool = False
    claims: list[dict] = Field(default_factory=list)   # {text, supported, evidence_ids[], kind}
    unsupported_claims: list[str] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    checks: dict = Field(default_factory=dict)


class AgentState(BaseModel):
    """Durable, explicit state. Serialized to the tasks table on each transition."""

    task: TaskContract
    tenant_id: str = ""
    user_id: str = ""
    conversation_id: str | None = None
    status: TaskStatus = TaskStatus.QUEUED
    plan: Plan | None = None
    current_step_id: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_results: list[ToolCallRecord] = Field(default_factory=list)
    memory_context: list[dict] = Field(default_factory=list)
    approval_request_id: str | None = None
    approval_status: str | None = None
    errors: list[dict] = Field(default_factory=list)
    verification: VerificationReport = Field(default_factory=VerificationReport)
    final_answer: str | None = None
    final_answer_unverified_note: str | None = None
    iterations: int = 0
    replans: int = 0
    scratch: dict[str, Any] = Field(default_factory=dict)

    def add_error(self, source: str, message: str, **details: Any) -> None:
        self.errors.append({"source": source, "message": message, **details})

    def pending_step(self) -> PlanStep | None:
        if not self.plan:
            return None
        for s in self.plan.steps:
            if s.status == StepStatus.PENDING:
                # respect dependencies: only run when all deps are DONE
                deps_ok = all(
                    d in self.completed_steps
                    for d in next((x for x in self.plan.steps if x.step_id == s.step_id), s).depends_on
                )
                if deps_ok:
                    return s
        return None
