"""Regression + unit tests for the core layer (config, ids, state, audit chain)."""
from ara.agent.state import AgentState, Plan, PlanStep, TaskContract
from ara.core.errors import BudgetExceeded, PolicyViolation
from ara.core.ids import new_id, utcnow
from ara.core.types import RiskLevel, StepAction, TaskStatus


def test_ids_are_prefixed_and_unique():
    a, b = new_id("task"), new_id("task")
    assert a.startswith("task_") and b.startswith("task_") and a != b


def test_utcnow_is_timezone_aware():
    assert utcnow().tzinfo is not None


def test_risk_ordering_and_approval_gate():
    assert RiskLevel.LOW.rank < RiskLevel.MEDIUM.rank < RiskLevel.HIGH.rank < RiskLevel.CRITICAL.rank
    assert not RiskLevel.MEDIUM.requires_approval()
    assert RiskLevel.HIGH.requires_approval()
    assert RiskLevel.CRITICAL.requires_approval()


def test_task_contract_defaults_are_safe():
    c = TaskContract(user_request="x")
    assert c.execution_budget.max_tool_calls > 0
    assert c.verification_requirements.require_citations
    assert c.risk_level == RiskLevel.LOW


def test_agent_state_pending_step_respects_dependencies():
    s1 = PlanStep(description="retrieve docs", action=StepAction.RETRIEVE, query="q")
    s2 = PlanStep(description="calc", action=StepAction.TOOL_CALL, tool="calculator", depends_on=[s1.step_id])
    state = AgentState(task=TaskContract(user_request="r"), plan=Plan(steps=[s1, s2]))
    assert state.pending_step().step_id == s1.step_id
    s1.status = "DONE"
    state.completed_steps.append(s1.step_id)
    assert state.pending_step().step_id == s2.step_id
    s2.status = "DONE"
    state.completed_steps.append(s2.step_id)
    assert state.pending_step() is None


def test_terminal_status_flag():
    assert TaskStatus.COMPLETED.is_terminal
    assert not TaskStatus.WAITING_FOR_APPROVAL.is_terminal


def test_error_taxonomy_codes():
    assert BudgetExceeded("x").http_status == 429
    assert PolicyViolation("x").http_status == 403
    assert BudgetExceeded("x").to_dict()["code"] == "budget_exceeded"


def test_audit_log_hash_chain_tamper_evidence(uow):
    uow.audit.append(actor="t", action="a1", detail={"x": 1})
    uow.audit.append(actor="t", action="a2", detail={"x": 2})
    assert uow.audit.verify_chain()
    # tamper with a row -> chain breaks
    uow.db.execute("UPDATE audit_log SET action='forged' WHERE action='a1'")
    assert not uow.audit.verify_chain()


def test_tenant_isolation_in_repositories(uow):
    uow.tasks.create(
        contract={"task_id": "task_1", "user_request": "r", "risk_level": "LOW"},
        state={"task": {"task_id": "task_1"}, "conversation_id": None},
        tenant_id="ten_a", user_id="u1", trace_id="run_1", status="QUEUED",
    )
    assert uow.tasks.get("task_1", "ten_a") is not None
    assert uow.tasks.get("task_1", "ten_b") is None  # cross-tenant access blocked
