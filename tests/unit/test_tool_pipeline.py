"""Tool pipeline tests: validation, authz, risk escalation, approval gate, retries,
idempotency, observation verification, fallback. These encode the core promise:
an LLM proposal can never bypass deterministic gates."""
import pytest

from ara.agent.state import TaskContract
from ara.core.types import RiskLevel
from ara.policy.authz import ApprovalRequiredSignal as Sig, PolicyEngine, Principal
from ara.tools import (
    MockWebCorpus, ToolPipeline, ToolRegistry, PipelineContext, get_sent_log,
    register_builtin_tools,
)
from ara.tools.builtin import ReportStore, delete_document_handler, get_report_store


def make_ctx(registry, *, role="user", contract=None, approvals=None, trace=None, documents_store=None):
    approvals = approvals if approvals is not None else {}
    principal = Principal.for_role("u1", "ten_a", role)
    contract = contract or TaskContract(user_request="r", risk_level=RiskLevel.LOW)

    def create_approval(tool, args, risk, reason):
        aid = f"apr_{len(approvals) + 1}"
        approvals[aid] = {"tool": tool, "status": "PENDING", "args": args, "risk": risk}
        return aid

    def get_decision(aid):
        return approvals.get(aid, {}).get("status")

    audit_calls = []
    return PipelineContext(
        principal=principal, contract=contract, registry=registry, policy=PolicyEngine(),
        create_approval=create_approval, get_approval_decision=get_decision,
        audit=lambda **kw: audit_calls.append(kw), trace=trace, documents_store=documents_store,
    ), approvals, audit_calls


@pytest.fixture(autouse=True)
def _isolated_global_stores():
    """Keep process-global tool stores clean between tests."""
    yield
    get_sent_log().sent.clear()
    get_report_store().drafts.clear()


@pytest.fixture()
def registry():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    return reg


def test_calculator_executes_and_is_verified(registry):
    ctx, _, audit = make_ctx(registry)
    outcome = ToolPipeline().execute("calculator", {"expression": "2 * (3 + 4)"}, ctx)
    assert outcome.record.status == "ok"
    assert outcome.record.result["result"] == 14
    assert outcome.record.result["_observation"]["verified"] is True
    assert audit and audit[0]["action"] == "tool_executed"


def test_calculator_rejects_code_injection(registry):
    ctx, _, _ = make_ctx(registry)
    outcome = ToolPipeline().execute("calculator", {"expression": "__import__('os').system('ls')"}, ctx)
    assert outcome.record.status == "error"
    assert "validation failed" in outcome.record.error or "disallowed" in outcome.record.error


def test_unknown_tool_is_recorded_not_raised(registry):
    ctx, _, _ = make_ctx(registry)
    outcome = ToolPipeline().execute("nonexistent", {}, ctx)
    assert outcome.record.status == "error" and "unknown tool" in outcome.record.error


def test_input_schema_strict_rejects_extra_fields(registry):
    ctx, _, _ = make_ctx(registry)
    outcome = ToolPipeline().execute("clock", {"tz": "hacked"}, ctx)
    assert outcome.record.status == "error" and "validation failed" in outcome.record.error


def test_param_range_enforced(registry):
    ctx, _, _ = make_ctx(registry)
    outcome = ToolPipeline().execute("web_search", {"query": "x", "limit": 999}, ctx)
    assert outcome.record.status == "error"  # limit must be <= 10


def test_contract_allowlist_blocks_tools(registry):
    contract = TaskContract(user_request="r", allowed_tools=["calculator"])
    ctx, _, _ = make_ctx(registry, contract=contract)
    outcome = ToolPipeline().execute("web_search", {"query": "x"}, ctx)
    assert outcome.record.status == "rejected" and "allowlist" in outcome.record.error


def test_contract_forbidden_actions_block_tools(registry):
    contract = TaskContract(user_request="r", forbidden_actions=["delete_document"])
    ctx, _, _ = make_ctx(registry, contract=contract)
    outcome = ToolPipeline().execute("delete_document", {"document_id": "doc_1"}, ctx)
    assert outcome.record.status == "rejected"


def test_authorization_blocks_tool_for_role(registry):
    reg = ToolRegistry()
    from ara.tools.registry import ToolSpec
    reg.register(ToolSpec(name="admin_only", description="d",
                          input_schema={"type": "object", "properties": {}},
                          output_schema={"type": "object", "properties": {}},
                          handler=lambda a, c: {}, permissions={"admin:gravity"}))
    ctx, _, _ = make_ctx(reg, role="user")
    outcome = ToolPipeline().execute("admin_only", {}, ctx)
    assert outcome.record.status == "rejected" and "not authorized" in outcome.record.error


def test_high_risk_tool_requires_durable_approval(registry):
    ctx, approvals, _ = make_ctx(registry)
    store = get_report_store()
    draft = store.create("t", "b", {"tenant_id": "ten_a"})
    with pytest.raises(Sig) as ei:
        ToolPipeline().execute("send_report", {"draft_id": draft["draft_id"], "recipient": "a@b.c"}, ctx)
    sig = ei.value
    assert sig.approval_id in approvals
    assert approvals[sig.approval_id]["status"] == "PENDING"
    assert sig.risk == RiskLevel.HIGH


def test_approved_high_risk_tool_executes_exactly_once(registry):
    ctx, approvals, _ = make_ctx(registry)
    draft = get_report_store().create("t", "b", {"tenant_id": "ten_a"})
    with pytest.raises(Sig) as ei:
        ToolPipeline().execute("send_report", {"draft_id": draft["draft_id"], "recipient": "a@b.c"}, ctx)
    approvals[ei.value.approval_id]["status"] = "APPROVED"
    outcome = ToolPipeline().execute("send_report", {"draft_id": draft["draft_id"], "recipient": "a@b.c"}, ctx,
                                     approved_approval_id=ei.value.approval_id)
    assert outcome.record.status == "ok" and outcome.record.approved_by
    assert len(get_sent_log().sent) == 1  # executed exactly once


def test_rejected_approval_blocks_execution(registry):
    ctx, approvals, _ = make_ctx(registry)
    draft = get_report_store().create("t", "b", {"tenant_id": "ten_a"})
    with pytest.raises(Sig) as ei:
        ToolPipeline().execute("send_report", {"draft_id": draft["draft_id"], "recipient": "a@b.c"}, ctx)
    approvals[ei.value.approval_id]["status"] = "REJECTED"
    outcome = ToolPipeline().execute("send_report", {"draft_id": draft["draft_id"], "recipient": "a@b.c"}, ctx,
                                     approved_approval_id=ei.value.approval_id)
    assert outcome.record.status == "rejected"
    assert len(get_sent_log().sent) == 0  # never executed


def test_argument_driven_risk_escalation(registry):
    ctx, approvals, _ = make_ctx(registry)
    # args contain 'delete' -> HIGH -> approval required even for a low-risk tool
    with pytest.raises(Sig) as ei:
        ToolPipeline().execute("generate_report_draft",
                               {"title": "delete old data", "body": "x"}, ctx)
    assert ei.value.risk == RiskLevel.HIGH


def test_non_idempotent_tool_never_blindly_retries(registry):
    spec = registry.get("send_report")
    assert spec.idempotent is False and spec.max_attempts == 1


def test_destructive_tool_requires_critical_risk():
    from ara.tools.registry import ToolSpec
    with pytest.raises(Exception):
        ToolSpec(name="bad_delete", description="d",
                 input_schema={"type": "object", "properties": {}},
                 output_schema={"type": "object", "properties": {}},
                 handler=lambda a, c: {}, side_effects="destructive", risk=RiskLevel.LOW).validate_spec()


def test_delete_document_destructive_flow(registry):
    store = {"doc_1": {"title": "x"}}
    ctx, approvals, _ = make_ctx(registry, documents_store=store)
    with pytest.raises(Sig) as ei:
        ToolPipeline().execute("delete_document", {"document_id": "doc_1"}, ctx)
    assert ei.value.risk == RiskLevel.CRITICAL
    approvals[ei.value.approval_id]["status"] = "APPROVED"
    outcome = ToolPipeline().execute("delete_document", {"document_id": "doc_1"}, ctx,
                                     approved_approval_id=ei.value.approval_id)
    assert outcome.record.status == "ok" and "doc_1" not in store


def test_missing_document_fails_cleanly_after_approval(registry):
    ctx, approvals, _ = make_ctx(registry, documents_store={})
    with pytest.raises(Sig) as ei:
        ToolPipeline().execute("delete_document", {"document_id": "doc_missing"}, ctx)
    approvals[ei.value.approval_id]["status"] = "APPROVED"
    outcome = ToolPipeline().execute("delete_document", {"document_id": "doc_missing"}, ctx,
                                     approved_approval_id=ei.value.approval_id)
    assert outcome.record.status == "error"


def test_output_schema_violation_is_caught(registry):
    from ara.tools.registry import ToolSpec
    reg = ToolRegistry()
    reg.register(ToolSpec(name="broken", description="d",
                          input_schema={"type": "object", "properties": {}},
                          output_schema={"type": "object", "properties": {"n": {"type": "integer"}},
                                         "required": ["n"]},
                          handler=lambda a, c: {"n": "not an int"}))
    ctx, _, _ = make_ctx(reg)
    outcome = ToolPipeline().execute("broken", {}, ctx)
    assert outcome.record.status == "error" and "output validation failed" in outcome.record.error


def test_web_results_are_marked_untrusted(registry):
    corpus = MockWebCorpus()
    corpus.seed("http://x", "acme revenue was 100 dollars in 2023")
    reg = ToolRegistry()
    register_builtin_tools(reg, web_corpus=corpus)
    ctx, _, _ = make_ctx(reg)
    outcome = ToolPipeline().execute("web_search", {"query": "acme revenue"}, ctx)
    assert outcome.record.status == "ok"
    assert outcome.record.result["disclaimer"] == "untrusted web content"
    assert outcome.evidence_worthy is False  # pipeline never claims trust by default


def test_fallback_tool_used_on_failure():
    reg = ToolRegistry()
    from ara.tools.registry import ToolSpec
    calls = {"flaky": 0}

    def flaky(a, c):
        calls["flaky"] += 1
        raise ConnectionError("down")

    reg.register(ToolSpec(name="flaky", description="d",
                          input_schema={"type": "object", "properties": {}},
                          output_schema={"type": "object", "properties": {}},
                          handler=flaky, retryable=False, max_attempts=1, fallback_tool="stable"))
    reg.register(ToolSpec(name="stable", description="d",
                          input_schema={"type": "object", "properties": {}},
                          output_schema={"type": "object",
                                         "properties": {"source": {"type": "string"}}},
                          handler=lambda a, c: {"source": "stable"}))
    ctx, _, _ = make_ctx(reg)
    outcome = ToolPipeline().execute_with_fallback("flaky", {}, ctx)
    assert outcome.record.result["source"] == "stable"
    assert outcome.record.result["_observation"]["verified"] is True
    assert outcome.record.args.get("fallback_for") == "flaky"
