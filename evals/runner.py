"""Evaluation framework.

Each scenario is executed through the REAL runtime (real guardrails, real
verification, real tool pipeline) against a fresh in-memory-backed database with
a deterministic scripted LLM. Metrics are computed from observed behavior, never
asserted by the scenario author.

Tracked metrics:
  task_success, refusal_precision, citation_coverage, unsupported_claim_rate,
  retrieval_hit_rate, guardrail_activations, approval_compliance, latency, cost.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ara.agent.llm import ScriptedBehavior, ScriptedProvider
from ara.agent.runtime import AgentRuntime
from ara.agent.state import TaskContract
from ara.core.config import Settings
from ara.db import UnitOfWork
from ara.db.database import Database
from ara.evidence import EvidenceManager, VerificationEngine
from ara.guardrails import InjectionDetector, OutputGuardrail
from ara.memory import MemoryManager
from ara.policy.authz import PolicyEngine, Principal
from ara.retrieval import InProcessMockColPali, IngestionPipeline, RetrievalEngine
from ara.retrieval.vectors import InMemoryMultiVectorStore
from ara.tools import ToolRegistry, get_mock_web_corpus, get_report_store, get_sent_log, register_builtin_tools
from ara.tools.registry import ToolSpec


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    passed: bool
    checks: dict = field(default_factory=dict)
    status: str = ""
    answer: str = ""
    latency_ms: int = 0
    tool_calls: int = 0
    claims_total: int = 0
    claims_supported: int = 0
    citations: int = 0
    guardrail_activations: int = 0
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def build_stack(seed_web: dict | None = None, extra_tools: list[ToolSpec] | None = None,
                behaviors: list[ScriptedBehavior] | None = None):
    tmp = tempfile.mkdtemp()
    settings = Settings(env="test", database_url=f"sqlite:///./{tmp}/eval.db",
                        dev_api_key="eval", injection_threshold=0.55)
    db = Database(settings.database_url)
    db.connect()
    uow = UnitOfWork(db)
    detector = InjectionDetector(0.55)
    em = EvidenceManager(detector)
    colpali = InProcessMockColPali()
    store = InMemoryMultiVectorStore()
    registry = ToolRegistry()
    web = get_mock_web_corpus()
    if seed_web:
        for url, content in seed_web.items():
            web.seed(url, content)
    register_builtin_tools(registry, web_corpus=web)
    for spec in extra_tools or []:
        registry.register(spec)
    retrieval = RetrievalEngine(colpali, store, em, detector, top_k=5, min_score=0.1)
    runtime = AgentRuntime(uow=uow, llm=ScriptedProvider(behaviors=behaviors), registry=registry,
                           retrieval=retrieval, evidence_manager=em,
                           verifier=VerificationEngine(em), output_guardrail=OutputGuardrail(detector),
                           policy_engine=PolicyEngine(), memory=MemoryManager(uow), settings=settings)
    return {"runtime": runtime, "uow": uow, "ingest": IngestionPipeline(uow, colpali, store),
            "registry": registry, "settings": settings, "db": db, "em": em}


# ---------------------------------------------------------------- fault-injection tool factories
def failing_handler(args, ctx):
    raise ConnectionError("simulated permanent failure")


def flaky_handler_factory(fail_times: int = 2):
    state = {"n": 0}

    def handler(args, ctx):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise ConnectionError("simulated transient failure")
        return {"value": "recovered"}
    return handler


def slow_handler(args, ctx):
    import time

    time.sleep(float(args.get("delay_s", 10)))
    return {"done": True}


def malformed_handler(args, ctx):
    return {"unexpected_shape": True}


HANDLER_FACTORIES = {
    "failing": lambda spec=None: failing_handler,
    "slow": lambda spec=None: slow_handler,
    "malformed": lambda spec=None: malformed_handler,
    "flaky": lambda spec=None: flaky_handler_factory(2),
}


def load_scenarios(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text())
    return data["scenarios"]


def run_scenario(sc: dict) -> ScenarioResult:
    res = ScenarioResult(scenario_id=sc["id"], category=sc["category"], passed=False)
    behaviors = []
    for b in sc.get("scripted_llm", []):
        behaviors.append(ScriptedBehavior(match=lambda s, u, _b=b: _b["match"] in s, response=b["response"]))
    stack = build_stack(seed_web=sc.get("seed_web"), extra_tools=None, behaviors=behaviors or None)
    # eval-specific fault-injection tools (handlers resolved from HANDLER_FACTORIES)
    from ara.core.types import RiskLevel as _RL

    for tool in sc.get("register_tools", []):
        t = dict(tool)
        handler_key = t.pop("handler", "failing")
        t["handler"] = HANDLER_FACTORIES.get(handler_key, HANDLER_FACTORIES["failing"])(t)
        if isinstance(t.get("risk"), str):
            t["risk"] = _RL(t["risk"])
        if isinstance(t.get("permissions"), list):
            t["permissions"] = set(t["permissions"])
        stack["registry"].register(ToolSpec(**t))
    try:
        for doc in sc.get("corpus", []):
            stack["ingest"].ingest(tenant_id="ten_eval", filename=doc["filename"],
                                   content=doc["content"].encode())
        contract = TaskContract(
            user_request=sc["request"],
            risk_level=sc.get("risk_level", "LOW"),
            allowed_tools=sc.get("allowed_tools", []),
            forbidden_actions=sc.get("forbidden_actions", []),
        )
        for k, v in (sc.get("budget") or {}).items():
            setattr(contract.execution_budget, k, v)
        principal = Principal.for_role("u_eval", "ten_eval", sc.get("role", "user"))
        start = time.monotonic()
        result = stack["runtime"].execute(contract, principal)
        res.latency_ms = int((time.monotonic() - start) * 1000)
        res.status = result.status.value
        res.answer = result.answer or ""
        res.tool_calls = len(result.state.tool_results)
        claims = result.state.verification.claims
        res.claims_total = len(claims)
        res.claims_supported = sum(1 for c in claims if c["supported"])
        cited = {c for v in claims for c in v["evidence_ids"]}
        res.citations = len(cited)
        res.guardrail_activations = len([e for e in result.state.errors if e.get("source") in
                                         ("guardrail", "output_guardrail")]) + \
            (1 if result.state.verification.refused else 0)
        budget = result.state.scratch.get("budget") or {}
        res.cost_usd = float(budget.get("cost_usd", 0.0))

        # ------------------------------------------------ deterministic checks
        exp = sc.get("expected", {})
        checks = {}
        checks["status_ok"] = result.status.value in (exp.get("statuses") or ["COMPLETED"])
        for phrase in exp.get("must_contain", []):
            checks[f"contains:{phrase[:32]}"] = phrase.lower() in res.answer.lower()
        if exp.get("must_contain_any"):
            checks["contains_any"] = any(p.lower() in res.answer.lower() for p in exp["must_contain_any"])
        for phrase in exp.get("must_not_contain", []):
            checks[f"excludes:{phrase[:32]}"] = phrase.lower() not in res.answer.lower()
        if "refuse" in exp:
            checks["refusal"] = result.state.verification.refused == exp["refuse"]
        if "min_citations" in exp:
            checks["min_citations"] = res.citations >= exp["min_citations"]
        if "approval_expected" in exp:
            checks["approval_flow"] = (result.status.value == "WAITING_FOR_APPROVAL") == exp["approval_expected"]
            if exp["approval_expected"]:
                approval = stack["uow"].approvals.pending_for_task(result.task_id)
                checks["approval_recorded"] = approval is not None
        if "injection_flagged" in exp:
            flagged = any(e.injection_scan["flagged"] for e in result.state.evidence)
            checks["injection_flagged"] = flagged == exp["injection_flagged"]
        if exp.get("no_tool_named"):
            used = [r.tool for r in result.state.tool_results]
            checks["tool_not_used"] = exp["no_tool_named"] not in used
        if "min_tool_calls" in exp:
            checks["min_tool_calls"] = res.tool_calls >= exp["min_tool_calls"]
        res.checks = checks
        res.passed = all(checks.values())
        return res
    finally:
        stack["db"].close()
        get_sent_log().sent.clear()
        get_report_store().drafts.clear()


def run_dataset(paths: list[str | Path]) -> dict:
    results: list[ScenarioResult] = []
    for path in paths:
        for sc in load_scenarios(path):
            results.append(run_scenario(sc))
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    refused_expected = [r for r in results if "refusal" in r.checks]
    refusal_expected_ids = {r.scenario_id for r in results if "refusal" in r.checks}
    judged = [r for r in results if r.scenario_id not in refusal_expected_ids]
    claims_total = sum(r.claims_total for r in judged)
    claims_supported = sum(r.claims_supported for r in judged)
    metrics = {
        "total_scenarios": total,
        "passed": passed,
        "task_success_rate": round(passed / total, 3) if total else 0.0,
        "refusal_precision": round(
            sum(1 for r in refused_expected if r.checks.get("refusal")) / len(refused_expected), 3)
        if refused_expected else None,
        "unsupported_claim_rate": round((claims_total - claims_supported) / claims_total, 3)
        if claims_total else 0.0,
        "citation_coverage": round(
            sum(1 for r in results if r.citations > 0 or r.checks.get("refusal")) / total, 3) if total else 0.0,
        "guardrail_activations": sum(r.guardrail_activations for r in results),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / total) if total else 0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "per_category": {},
    }
    for r in results:
        cat = metrics["per_category"].setdefault(r.category, {"total": 0, "passed": 0})
        cat["total"] += 1
        cat["passed"] += 1 if r.passed else 0
    return {"metrics": metrics, "results": [vars(r) for r in results]}


def format_report(report: dict) -> str:
    m = report["metrics"]
    lines = ["# ARA Evaluation Report", "",
             f"- Scenarios: {m['passed']}/{m['total_scenarios']} passed (success rate {m['task_success_rate']:.0%})",
             f"- Refusal precision: {m['refusal_precision']}",
             f"- Unsupported claim rate: {m['unsupported_claim_rate']:.1%}",
             f"- Citation coverage: {m['citation_coverage']:.0%}",
             f"- Guardrail activations: {m['guardrail_activations']}",
             f"- Avg latency: {m['avg_latency_ms']} ms", "", "| category | passed |",
             "|---|---|"]
    for cat, c in m["per_category"].items():
        lines.append(f"| {cat} | {c['passed']}/{c['total']} |")
    lines += ["", "## Scenario detail", "", "| scenario | category | result | failed checks |",
              "|---|---|---|---|"]
    for r in report["results"]:
        failed = [k for k, ok in r["checks"].items() if not ok]
        lines.append(f"| {r['scenario_id']} | {r['category']} | {'✅' if r['passed'] else '❌ ' + r['status']} | {', '.join(failed)} |")
    return "\n".join(lines) + "\n"
