"""MCP (Model Context Protocol) client — JSON-RPC 2.0.

Implements the MCP lifecycle over two transports:
  - stdio  : newline-delimited JSON-RPC to a subprocess server (the standard MCP stdio transport)
  - http   : POST JSON-RPC to an HTTP endpoint (streamable-HTTP-style servers)

Only the subset ARA needs is implemented: initialize, tools/list, tools/call,
ping. Protocol: https://spec.modelcontextprotocol.io (2024-11-05+).

Security: tool descriptions and outputs from MCP servers are UNTRUSTED. The
bridge labels them accordingly; prompts wrap them in untrusted blocks; output
content is injection-scanned like any other external content.
"""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

from ara.agent.resilience import run_with_timeout, with_retry
from ara.core.errors import ToolError, ToolTimeout, ToolUnavailable
from ara.core.ids import iso_now
from ara.core.logging import get_logger

log = get_logger("ara.mcp")

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "ara-agent", "version": "0.1.0"}


class McpError(ToolError):
    code = "mcp_error"


class McpStdioTransport:
    """Newline-delimited JSON-RPC over a subprocess's stdin/stdout."""

    def __init__(self, command: list[str], cwd: str | None = None, env: dict | None = None):
        self.command = command
        self.cwd = cwd
        self.env = env
        self.proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._next_id = 1

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.cwd, env=self.env, text=True, bufsize=1,
        )

    def _send(self, payload: dict) -> None:
        if not self.proc or self.proc.poll() is not None:
            raise ToolUnavailable(f"MCP server process not running ({self.command[0] if self.command else '?'})")
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise ToolUnavailable(f"MCP server pipe broken: {exc}") from exc

    def _recv(self, want_id: int, timeout_s: float) -> dict:
        """Read frames until the wanted response arrives, honoring the deadline via
        selector-based reads (a blocking readline() would ignore the timeout)."""
        import selectors
        import time

        assert self.proc and self.proc.stdout
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToolTimeout(f"MCP response timeout after {timeout_s}s")
                events = sel.select(timeout=remaining)
                if not events:
                    raise ToolTimeout(f"MCP response timeout after {timeout_s}s")
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        raise ToolUnavailable("MCP server exited unexpectedly")
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("mcp_bad_frame", extra={"fields": {"line": line[:120]}})
                    continue
                if msg.get("id") == want_id:
                    return msg
                # notifications/other responses ignored (single-request-at-a-time)
        finally:
            sel.unregister(self.proc.stdout)
            sel.close()

    def request(self, method: str, params: dict | None = None, timeout_s: float = 20.0) -> dict:
        with self._lock:
            self.start()
            rid = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            msg = self._recv(rid, timeout_s)
            if "error" in msg:
                err = msg["error"]
                raise McpError(f"MCP error {err.get('code')}: {err.get('message')}")
            return msg.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class McpHttpTransport:
    def __init__(self, url: str, headers: dict | None = None):
        self.url = url
        self.headers = headers or {}
        self._next_id = 1

    def start(self) -> None:
        return None

    def request(self, method: str, params: dict | None = None, timeout_s: float = 20.0) -> dict:
        import httpx

        def call() -> dict:
            self._next_id += 1
            body = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
            try:
                resp = httpx.post(self.url, json=body, headers={**self.headers,
                                 "Accept": "application/json"}, timeout=timeout_s)
            except httpx.HTTPError as exc:
                raise ToolUnavailable(f"MCP HTTP endpoint unreachable: {exc}") from exc
            if resp.status_code >= 400:
                raise McpError(f"MCP HTTP error {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            if "error" in data:
                raise McpError(f"MCP error {data['error'].get('code')}: {data['error'].get('message')}")
            return data.get("result", {})

        return with_retry(call, max_attempts=2, base_delay_s=0.2, retryable=(ToolUnavailable,))

    def notify(self, method: str, params: dict | None = None) -> None:
        return None

    def close(self) -> None:
        return None


class McpClient:
    """High-level MCP session: initialize -> tools/list -> tools/call."""

    def __init__(self, transport, *, server_name: str, timeout_s: float = 20.0):
        self.transport = transport
        self.server_name = server_name
        self.timeout_s = timeout_s
        self.initialized = False
        self.server_info: dict = {}

    def initialize(self) -> dict:
        result = run_with_timeout(
            lambda: self.transport.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": CLIENT_INFO,
            }, timeout_s=self.timeout_s),
            timeout_s=self.timeout_s + 2, what=f"mcp:{self.server_name}:initialize")
        self.transport.notify("notifications/initialized")
        self.initialized = True
        self.server_info = result.get("serverInfo", {})
        log.info("mcp_initialized", extra={"fields": {"server": self.server_name,
                                                      "serverInfo": self.server_info}})
        return result

    def list_tools(self) -> list[dict]:
        if not self.initialized:
            self.initialize()
        result = self.transport.request("tools/list", {}, timeout_s=self.timeout_s)
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> dict:
        if not self.initialized:
            self.initialize()
        result = self.transport.request("tools/call", {"name": name, "arguments": arguments},
                                        timeout_s=self.timeout_s)
        # MCP result shape: {content: [{type: text,...}], isError: bool}
        return {
            "is_error": bool(result.get("isError", False)),
            "content": result.get("content", []),
            "structured": result.get("structuredContent"),
            "called_at": iso_now(),
        }

    def ping(self) -> bool:
        try:
            self.transport.request("ping", {}, timeout_s=5)
            return True
        except ToolError:
            return False

    def close(self) -> None:
        self.transport.close()
