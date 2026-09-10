"""MCP -> ARA tool bridge.

Exposes MCP server tools through the same governed ToolRegistry pipeline as
builtin tools: JSON-schema input validation, permissions, risk classification,
approval gating, output validation and audit all apply UNIFORMLY.

Risk policy for MCP tools (configurable):
  default LOW for read-only-looking tools; tools whose name matches
  write/delete/send/execute patterns are classified HIGH (approval-gated).
  All MCP tool outputs are UNTRUSTED content.
"""
from __future__ import annotations

import re

from ara.core.errors import ToolError
from ara.core.logging import get_logger
from ara.core.types import RiskLevel
from ara.mcp.client import McpClient
from ara.policy.authz import PolicyEngine
from ara.tools.registry import ToolRegistry, ToolSpec

log = get_logger("ara.mcp.bridge")

RISKY_NAME_RE = re.compile(r"(?i)(write|create|update|delete|send|execute|run|deploy|drop|insert|post|put)")


def classify_mcp_tool_risk(tool: dict, overrides: dict[str, RiskLevel] | None = None) -> RiskLevel:
    name = tool.get("name", "")
    if overrides and name in overrides:
        return overrides[name]
    if RISKY_NAME_RE.search(name):
        return RiskLevel.HIGH
    return RiskLevel.LOW


def jsonschema_from_mcp(tool: dict) -> dict:
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    if schema.get("type") != "object":  # tolerate sloppy servers
        schema = {"type": "object", "properties": {}}
    return schema


def text_from_mcp_content(content: list) -> str:
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


class McpBridge:
    def __init__(self, registry: ToolRegistry, policy: PolicyEngine | None = None,
                 risk_overrides: dict[str, RiskLevel] | None = None):
        self.registry = registry
        self.policy = policy or PolicyEngine()
        self.risk_overrides = risk_overrides or {}
        self.clients: dict[str, McpClient] = {}

    def attach_server(self, client: McpClient, *, permissions: set[str] | None = None,
                      timeout_s: float = 20.0, max_attempts: int = 1) -> list[str]:
        """Discover tools on an MCP server and register them. Returns tool names."""
        tools = client.list_tools()
        registered: list[str] = []
        for tool in tools:
            name = f"mcp_{client.server_name}_{tool.get('name', 'unnamed')}"
            if self.registry.exists(name):
                log.warning("mcp_tool_name_collision_skipped", extra={"fields": {"tool": name}})
                continue
            risk = classify_mcp_tool_risk(tool, self.risk_overrides)
            spec = ToolSpec(
                name=name,
                description=f"[MCP:{client.server_name}] {tool.get('description', '')}"[:500],
                input_schema=jsonschema_from_mcp(tool),
                output_schema={"type": "object", "properties": {
                    "is_error": {"type": "boolean"},
                    "text": {"type": "string"},
                    "structured": {"type": ["object", "array", "string", "number", "boolean", "null"]},
                }, "required": ["is_error", "text"]},
                handler=self._make_handler(client, tool.get("name", "")),
                risk=risk,
                permissions=permissions or ({"tools:low"} if risk == RiskLevel.LOW else {"tools:medium"}),
                timeout_s=timeout_s,
                max_attempts=max_attempts,
                retryable=False,               # remote side effects: never blind-retry
                idempotent=False,              # unknown: assume not idempotent
                side_effects="external" if risk.rank >= RiskLevel.HIGH.rank else "none",
                audit=True,
                trusted_output=False,          # MCP output is untrusted content
                source=f"mcp:{client.server_name}",
            )
            self.registry.register(spec)
            registered.append(name)
        self.clients[client.server_name] = client
        log.info("mcp_server_attached", extra={"fields": {"server": client.server_name,
                                                          "tools": registered}})
        return registered

    @staticmethod
    def _make_handler(client: McpClient, mcp_tool_name: str):
        def handler(args: dict, ctx: dict) -> dict:
            result = client.call_tool(mcp_tool_name, args)
            if result.get("is_error"):
                raise ToolError(f"MCP tool '{mcp_tool_name}' reported an error: "
                                f"{text_from_mcp_content(result.get('content', []))[:200]}")
            text = text_from_mcp_content(result.get("content", []))
            return {"is_error": False, "text": text[:8000], "structured": result.get("structured")}
        return handler

    def detach_server(self, server_name: str) -> None:
        client = self.clients.pop(server_name, None)
        if not client:
            return
        for name in [n for n, s in self.registry._tools.items()
                     if s.source == f"mcp:{server_name}"]:
            del self.registry._tools[name]
        client.close()
