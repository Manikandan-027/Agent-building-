"""MCP integration tests against a REAL subprocess server (scripts/demo_mcp_server.py):
lifecycle, discovery, governed calls through the pipeline, risky-name approval
gating, output untrusted labeling, timeout handling."""
import json
import sys
from pathlib import Path

import pytest

from ara.agent.state import TaskContract
from ara.core.errors import ToolError
from ara.core.types import RiskLevel
from ara.mcp import McpBridge, McpClient, McpStdioTransport
from ara.policy.authz import PolicyEngine, Principal
from ara.tools import ToolPipeline, ToolRegistry, PipelineContext

SERVER_CMD = [sys.executable, str(Path(__file__).parents[2] / "scripts" / "demo_mcp_server.py")]


@pytest.fixture()
def client():
    c = McpClient(McpStdioTransport(SERVER_CMD), server_name="demo")
    yield c
    c.close()


def test_initialize_and_list_tools(client):
    info = client.initialize()
    assert info["serverInfo"]["name"] == "ara-demo-server"
    tools = client.list_tools()
    names = {t["name"] for t in tools}
    assert {"unit_convert", "fake_web_lookup", "slow_multiply"} <= names


def test_call_tool_roundtrip(client):
    client.initialize()
    result = client.call_tool("unit_convert", {"value": 1, "from_unit": "kg", "to_unit": "lb"})
    assert result["is_error"] is False
    assert "2.20462" in result["content"][0]["text"]


def test_bridge_registers_governed_tools(client):
    registry = ToolRegistry()
    bridge = McpBridge(registry)
    registered = bridge.attach_server(client)
    assert len(registered) == 3
    # risky-sounding tools are classified HIGH
    assert registry.get("mcp_demo_fake_web_lookup").risk == RiskLevel.LOW
    # 'convert'... none risky here; check untrusted output labeling
    spec = registry.get("mcp_demo_fake_web_lookup")
    assert spec.trusted_output is False
    assert spec.source == "mcp:demo"


def test_bridge_tool_call_through_full_pipeline(client):
    registry = ToolRegistry()
    McpBridge(registry).attach_server(client)
    principal = Principal.for_role("u1", "ten_a", "user")
    contract = TaskContract(user_request="convert units", risk_level=RiskLevel.LOW)
    approvals: dict = {}

    ctx = PipelineContext(
        principal=principal, contract=contract, registry=registry, policy=PolicyEngine(),
        create_approval=lambda t, a, r, reason: approvals.setdefault("id", "apr_1"),
        get_approval_decision=lambda aid: approvals.get("status"),
        audit=lambda **kw: None,
    )
    outcome = ToolPipeline().execute("mcp_demo_unit_convert",
                                     {"value": 5, "from_unit": "m", "to_unit": "ft"}, ctx)
    assert outcome.record.status == "ok"
    assert "16.4042" in outcome.record.result["text"]


def test_risky_named_mcp_tool_requires_approval(client):
    """A server tool whose name implies writes (e.g. delete_*) is HIGH risk and
    must suspend for human approval even though the server declared nothing."""
    registry = ToolRegistry()
    bridge = McpBridge(registry, risk_overrides={"fake_web_lookup": RiskLevel.HIGH})
    bridge.attach_server(client)
    assert registry.get("mcp_demo_fake_web_lookup").risk == RiskLevel.HIGH
    assert registry.get("mcp_demo_fake_web_lookup").side_effects == "external"

    principal = Principal.for_role("u1", "ten_a", "user")
    contract = TaskContract(user_request="lookup", risk_level=RiskLevel.HIGH)
    approvals: dict = {}

    def create_approval(tool, args, risk, reason):
        approvals["id"] = "apr_9"
        return approvals["id"]

    ctx = PipelineContext(
        principal=principal, contract=contract, registry=registry, policy=PolicyEngine(),
        create_approval=create_approval, get_approval_decision=lambda aid: approvals.get("status"),
        audit=lambda **kw: None,
    )
    from ara.policy.authz import ApprovalRequiredSignal

    with pytest.raises(ApprovalRequiredSignal):
        ToolPipeline().execute("mcp_demo_fake_web_lookup", {"key": "acme_ceo"}, ctx)


def test_mcp_tool_error_surfaces_cleanly(client):
    client.initialize()
    result = client.call_tool("unit_convert", {"value": 1, "from_unit": "kg", "to_unit": "m"})
    assert result["is_error"] is True


def test_mcp_bad_arguments_rejected_by_schema_validation(client):
    registry = ToolRegistry()
    McpBridge(registry).attach_server(client)
    principal = Principal.for_role("u1", "ten_a", "user")
    contract = TaskContract(user_request="x", risk_level=RiskLevel.LOW)
    ctx = PipelineContext(
        principal=principal, contract=contract, registry=registry, policy=PolicyEngine(),
        create_approval=lambda *a: "apr_x", get_approval_decision=lambda aid: None,
        audit=lambda **kw: None,
    )
    outcome = ToolPipeline().execute("mcp_demo_unit_convert",
                                     {"value": "not-a-number", "from_unit": "m", "to_unit": "ft"}, ctx)
    assert outcome.record.status == "error"
    assert "validation failed" in outcome.record.error


def test_mcp_timeout_on_slow_tool(client):
    client_slow = McpClient(McpStdioTransport(SERVER_CMD), server_name="demo", timeout_s=0.3)
    try:
        client_slow.initialize()
        from ara.core.errors import ToolTimeout

        with pytest.raises(ToolTimeout):
            client_slow.call_tool("slow_multiply", {"a": 2, "b": 3, "delay_s": 5})
    finally:
        client_slow.close()


def test_mcp_output_flows_into_evidence_as_untrusted(uow, settings):
    """MCP tool text must be treatable as untrusted evidence (grounding with scan)."""
    registry = ToolRegistry()
    client = McpClient(McpStdioTransport(SERVER_CMD), server_name="demo")
    try:
        McpBridge(registry).attach_server(client)
        principal = Principal.for_role("u1", "ten_a", "user")
        contract = TaskContract(user_request="lookup", risk_level=RiskLevel.LOW)
        ctx = PipelineContext(
            principal=principal, contract=contract, registry=registry, policy=PolicyEngine(),
            create_approval=lambda *a: "apr_x", get_approval_decision=lambda aid: None,
            audit=lambda **kw: None,
        )
        outcome = ToolPipeline().execute("mcp_demo_fake_web_lookup", {"key": "acme_ceo"}, ctx)
        assert outcome.record.status == "ok"
        from ara.evidence import EvidenceManager
        from ara.guardrails import InjectionDetector

        em = EvidenceManager(InjectionDetector(0.55))
        ev = em.make(content=outcome.record.result["text"], document_id="mcp:demo",
                     content_type="tool_output", trust=__import__("ara.core.types", fromlist=["ContentTrust"]).ContentTrust.TOOL_OUTPUT,
                     source="tool_output")
        assert ev["source_trust"] == "tool_output"
        assert "Jane Doe" in ev["content"]
    finally:
        client.close()
