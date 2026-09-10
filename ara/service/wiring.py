"""Application wiring: builds the full agent stack from Settings (composition root).

Everything is assembled here once — repositories, vector store, colpali client,
guardrails, policy engine, tool registry (builtin + MCP), runtime — and shared by
the API, the worker, and evals.
"""
from __future__ import annotations

import json
import os

from ara.agent.llm import make_provider
from ara.agent.runtime import AgentRuntime
from ara.core.config import Settings
from ara.core.logging import get_logger, setup_logging
from ara.db import UnitOfWork
from ara.db.database import Database
from ara.evidence import EvidenceManager, VerificationEngine
from ara.guardrails import InjectionDetector, OutputGuardrail
from ara.memory import MemoryManager
from ara.policy.authz import PolicyEngine
from ara.retrieval import IngestionPipeline, RetrievalEngine, make_colpali, make_vector_store
from ara.tools import ToolRegistry, get_mock_web_corpus, register_builtin_tools

log = get_logger("ara.wiring")


class AppContext:
    """Shared stack container."""

    def __init__(self, settings: Settings):
        self.settings = settings
        setup_logging(settings.log_level)

        self.db = Database(settings.database_url)
        self.db.connect()
        self.uow = UnitOfWork(self.db)

        self.detector = InjectionDetector(settings.injection_threshold)
        self.evidence_manager = EvidenceManager(self.detector)
        self.colpali = make_colpali(settings)
        self.vector_store = make_vector_store(settings)
        self.retrieval = RetrievalEngine(self.colpali, self.vector_store, self.evidence_manager,
                                         self.detector, top_k=settings.retrieval_top_k,
                                         min_score=settings.retrieval_min_score)
        self.verifier = VerificationEngine(self.evidence_manager)
        self.output_guardrail = OutputGuardrail(self.detector)
        self.policy = PolicyEngine()
        self.memory = MemoryManager(self.uow)

        self.registry = ToolRegistry()
        register_builtin_tools(self.registry, web_corpus=get_mock_web_corpus())
        self._attach_mcp_servers()

        self.llm = make_provider(settings)
        self.runtime = AgentRuntime(uow=self.uow, llm=self.llm, registry=self.registry,
                                    retrieval=self.retrieval, evidence_manager=self.evidence_manager,
                                    verifier=self.verifier, output_guardrail=self.output_guardrail,
                                    policy_engine=self.policy, memory=self.memory, settings=settings)
        self.ingestion = IngestionPipeline(self.uow, self.colpali, self.vector_store)

    def _attach_mcp_servers(self) -> None:
        """MCP servers configured via ARA_MCP_SERVERS (JSON list of
        {"name": str, "command": [..]} for stdio or {"name","url"} for HTTP)."""
        raw = os.environ.get("ARA_MCP_SERVERS", "").strip()
        if not raw:
            return
        try:
            servers = json.loads(raw)
        except json.JSONDecodeError:
            log.error("mcp_config_invalid_json")
            return
        from ara.mcp import McpBridge, McpClient, McpHttpTransport, McpStdioTransport

        bridge = McpBridge(self.registry, self.policy)
        for spec in servers[:8]:
            try:
                if spec.get("url"):
                    transport = McpHttpTransport(spec["url"], headers=spec.get("headers"))
                else:
                    transport = McpStdioTransport(spec["command"])
                client = McpClient(transport, server_name=spec["name"],
                                   timeout_s=float(spec.get("timeout_s", 20)))
                bridge.attach_server(client)
            except Exception as exc:  # noqa: BLE001 — one bad server must not kill startup
                log.error("mcp_attach_failed", extra={"fields": {"server": spec.get("name"),
                                                                 "error": str(exc)[:200]}})

    def close(self) -> None:
        for name in list(getattr(self, "registry")._tools):
            pass
        self.db.close()
