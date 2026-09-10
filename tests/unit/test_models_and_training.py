"""Model routing + fine-tuning data pipeline tests."""
import json

import pytest

from ara.agent.llm import OpenAICompatibleProvider, route_models
from ara.agent.runtime import build_answerer_prompt
from ara.agent.training_data import (
    ExportReport,
    export_training_data,
    validate_jsonl,
    write_jsonl,
    _episode_is_high_quality,
)
from ara.evidence import EvidenceManager
from ara.guardrails import InjectionDetector


# ---------------------------------------------------------------- routing
def test_routing_planner_gets_strong_model():
    models = route_models("PLANNER", default="gpt-4.1-mini", planner_model="gpt-4.1",
                          fast_model="gpt-4.1-mini", fallbacks=[])
    assert models[0] == "gpt-4.1" and models[1] == "gpt-4.1-mini"


def test_routing_extractive_roles_get_fast_model():
    for tag in ("REASONER", "ANSWERER", "CLAIMER"):
        models = route_models(tag, default="gpt-4.1-mini", planner_model="gpt-4.1",
                              fast_model="mistral-small", fallbacks=["gemini-flash"])
        assert models[0] == "mistral-small"
        assert models == ["mistral-small", "gpt-4.1-mini", "gemini-flash"]


def test_routing_dedupes_and_preserves_order():
    models = route_models("ANSWERER", default="m1", planner_model="m2", fast_model="m1",
                          fallbacks=["m1", "m3"])
    assert models == ["m1", "m3"]


def test_routing_no_split_configured_uses_default():
    models = route_models("PLANNER", default="m1", planner_model="", fast_model="", fallbacks=["m2"])
    assert models == ["m1", "m2"]


def test_provider_sends_routed_model(monkeypatch):
    sent = []

    class FakeResp:
        status_code = 200
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def fake_post(url, **kwargs):
        sent.append((url, kwargs["json"]["model"]))
        return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OpenAICompatibleProvider("http://fake/v1", "sk", "gpt-4.1-mini",
                                 planner_model="gpt-4.1", fast_model="mistral-small")
    p.complete("PLANNER", "REQUEST: x")
    p.complete("ANSWERER", "REQUEST: y")
    assert [m for _, m in sent] == ["gpt-4.1", "mistral-small"]


# ---------------------------------------------------------------- training data
def _seed_completed_task(uow, runtime_env, *, refused=False):
    """Run one real research task through the runtime to create a verifiable episode."""
    from ara.agent.state import TaskContract
    from ara.policy.authz import Principal

    corpus = ("Acme Corporation FY2023 annual revenue was 50.2 million dollars, an increase "
              "of 12 percent over FY2022.")
    runtime_env["ingest"].ingest(tenant_id="ten_ft", filename="r.txt", content=corpus.encode())
    result = runtime_env["runtime"].execute(
        TaskContract(user_request="What was Acme FY2023 revenue?"),
        Principal.for_role("u1", "ten_ft", "user"))
    return result


@pytest.fixture()
def ft_env(uow, settings):
    """Minimal runtime stack on the test DB (mirrors e2e fixture)."""
    from ara.agent.llm import ScriptedProvider
    from ara.agent.runtime import AgentRuntime
    from ara.evidence import EvidenceManager as EM, VerificationEngine
    from ara.guardrails import OutputGuardrail
    from ara.memory import MemoryManager
    from ara.policy.authz import PolicyEngine
    from ara.retrieval import InProcessMockColPali, IngestionPipeline, RetrievalEngine
    from ara.retrieval.vectors import InMemoryMultiVectorStore
    from ara.tools import ToolRegistry, register_builtin_tools

    detector = InjectionDetector(0.55)
    em = EM(detector)
    colpali = InProcessMockColPali()
    store = InMemoryMultiVectorStore()
    registry = ToolRegistry()
    register_builtin_tools(registry)
    retrieval = RetrievalEngine(colpali, store, em, detector, top_k=5, min_score=0.1)
    runtime = AgentRuntime(uow=uow, llm=ScriptedProvider(), registry=registry, retrieval=retrieval,
                           evidence_manager=em, verifier=VerificationEngine(em),
                           output_guardrail=OutputGuardrail(detector), policy_engine=PolicyEngine(),
                           memory=MemoryManager(uow), settings=settings)
    return {"runtime": runtime, "uow": uow, "ingest": IngestionPipeline(uow, colpali, store),
            "em": em}


def test_export_requires_verified_episodes(ft_env):
    _seed_completed_task(ft_env["uow"], ft_env)
    planners, answerers, report = export_training_data(ft_env["uow"], ft_env["em"],
                                                       tenant_id="ten_ft")
    assert report.qualified_tasks >= 1
    assert planners and answerers, "a verified episode must produce both example types"
    ex = planners[0]
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    plan_obj = json.loads(ex["messages"][2]["content"])
    assert any(s["action"] == "retrieve" for s in plan_obj["steps"])
    ans = json.loads(answerers[0]["messages"][2]["content"])
    assert "50.2" in ans["answer"] and ans["citations"]
    assert "NOT VERIFIABLE" not in ans["answer"]


def test_export_rejects_refused_episodes(ft_env):
    from ara.agent.state import TaskContract
    from ara.policy.authz import Principal

    # question with no matching corpus -> refusal -> must be EXCLUDED
    ft_env["ingest"].ingest(tenant_id="ten_ft", filename="x.txt",
                            content=b"Totally unrelated text about gardening.")
    ft_env["runtime"].execute(TaskContract(user_request="What was Acme FY2023 revenue?"),
                              Principal.for_role("u1", "ten_ft", "user"))
    planners, answerers, report = export_training_data(ft_env["uow"], ft_env["em"],
                                                       tenant_id="ten_ft")
    combined = planners + answerers
    assert combined == [] or report.rejected.get("answer_not_supported", 0) >= 1
    for ex in combined:
        assert "cannot verify" not in ex["messages"][2]["content"].lower()


def test_jsonl_roundtrip_and_validation(tmp_path):
    examples = [{"messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": json.dumps({"answer": "a [ev_x].", "citations": ["ev_x"]})}]}]
    path = tmp_path / "ft.jsonl"
    assert write_jsonl(str(path), examples) == 1
    stats = validate_jsonl(str(path))
    assert stats["lines"] == 1 and not stats["errors"]


def test_jsonl_validation_catches_bad_shapes(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": ""}]}\nnot json\n')
    stats = validate_jsonl(str(path))
    assert len(stats["errors"]) >= 2  # role order + invalid JSON line


def test_high_quality_gate_logic():
    def row(status="COMPLETED", refused=False, supported=True, claims=None, note=None):
        return {"status": status,
                "state_json": json.dumps({"verification": {
                    "refused": refused, "answer_supported": supported,
                    "claims": claims or [{"supported": True, "evidence_ids": ["ev_1"]}]},
                    "final_answer_unverified_note": note})}

    ok, why = _episode_is_high_quality(row())
    assert ok and why == "ok"
    assert not _episode_is_high_quality(row(status="FAILED"))[0]
    assert not _episode_is_high_quality(row(refused=True))[0]
    assert not _episode_is_high_quality(row(supported=False))[0]
    assert not _episode_is_high_quality(row(claims=[{"supported": False, "evidence_ids": []}]))[0]
    assert not _episode_is_high_quality(row(note="guardrail block"))[0]
