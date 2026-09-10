"""API integration tests through the full FastAPI ASGI stack."""
import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from ara.core.config import Settings
from ara.service.app import create_app
from ara.service.worker import TaskWorker

HEADERS = {"X-API-Key": "test-key"}
GOOD_CORPUS = ("Acme Corporation FY2023 annual revenue was 50.2 million dollars, an increase of "
               "12 percent over FY2022. The report was published on 2024-03-15.")


@pytest.fixture()
def client(uow, settings):
    settings.env = "test"
    app = create_app(settings)
    app.state.context.db = uow.db  # share the test DB
    ctx = app.state.context
    ctx.uow = uow
    ctx.auth.uow = uow
    # rewire repositories to test db
    from ara.db import UnitOfWork

    shared = uow
    ctx.runtime.uow = shared
    ctx.memory.uow = shared
    ctx.retrieval = ctx.retrieval  # vector store is process-local, fine
    ctx.ingestion = type(ctx.ingestion)(shared, ctx.colpali, ctx.vector_store)
    worker = TaskWorker(ctx, poll_interval_s=0.05)
    worker.start()
    with TestClient(app) as c:
        c.worker = worker
        yield c
    worker.stop()


def test_health_open(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_required(client):
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401


def test_upload_and_list_documents(client):
    r = client.post("/documents", headers=HEADERS,
                    files={"file": ("report.txt", io.BytesIO(GOOD_CORPUS.encode()), "text/plain")})
    assert r.status_code == 200
    doc_id = r.json()["document_id"]
    assert r.json()["pages"] == 1
    r2 = client.get("/documents", headers=HEADERS)
    assert any(d["id"] == doc_id for d in r2.json()["documents"])
    r3 = client.get(f"/documents/{doc_id}", headers=HEADERS)
    assert r3.json()["title"] == "report.txt"
    # cross-tenant cannot read it: authenticate as a different tenant via crafted key? use isolation test below


def test_chat_grounds_answer_with_citations(client):
    client.post("/documents", headers=HEADERS,
                files={"file": ("report.txt", io.BytesIO(GOOD_CORPUS.encode()), "text/plain")})
    r = client.post("/chat", headers=HEADERS, json={"message": "What was Acme FY2023 revenue and growth?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert "50.2" in body["answer"]
    assert body["citations"], "answer must cite evidence"
    assert body["verification"]["answer_supported"]
    # conversation continuity
    cid = body["conversation_id"]
    r2 = client.post("/chat", headers=HEADERS,
                     json={"message": "What was Acme FY2023 revenue again?", "conversation_id": cid})
    assert r2.json()["conversation_id"] == cid


def test_chat_refuses_without_evidence(client):
    r = client.post("/chat", headers=HEADERS,
                    json={"message": "What was Acme's FY2049 Mars division profit in rubles?"})
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert "cannot verify" in body["answer"].lower()


def test_async_task_lifecycle_with_worker(client):
    client.post("/documents", headers=HEADERS,
                files={"file": ("report.txt", io.BytesIO(GOOD_CORPUS.encode()), "text/plain")})
    r = client.post("/tasks", headers=HEADERS,
                    json={"message": "What was Acme FY2023 revenue?"}, )
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    # worker claims and completes it
    for _ in range(100):
        s = client.get(f"/tasks/{task_id}/status", headers=HEADERS).json()["status"]
        if s in {"COMPLETED", "FAILED", "BUDGET_STOPPED"}:
            break
        time.sleep(0.05)
    assert s == "COMPLETED"
    detail = client.get(f"/tasks/{task_id}", headers=HEADERS).json()
    assert detail["final_answer"] and "50.2" in detail["final_answer"]
    ev = client.get(f"/tasks/{task_id}/evidence", headers=HEADERS).json()["evidence"]
    assert ev and ev[0]["document_id"]
    trace = client.get(f"/tasks/{task_id}/trace", headers=HEADERS)
    assert trace.status_code == 200 and trace.json()["spans"]


def test_approval_flow_over_api(client):
    """HIGH-risk plan -> WAITING_FOR_APPROVAL -> approve -> executed once."""
    from ara.agent.llm import ScriptedBehavior

    ctx = client.app.state.context
    # seed a report draft & script a send_report plan
    from ara.tools import get_report_store, get_sent_log

    get_sent_log().sent.clear()
    draft = get_report_store().create("T", "B", {"tenant_id": "ten_dev"})
    plan_json = json.dumps({"goal": "send", "steps": [
        {"description": "send", "action": "tool_call", "tool": "send_report",
         "args": {"draft_id": draft["draft_id"], "recipient": "cfo@acme.com"}},
        {"description": "final", "action": "final_answer"}]})
    ctx.llm.behaviors = [
        ScriptedBehavior(match=lambda s, u: "PLANNER" in s, response=plan_json),
        ScriptedBehavior(match=lambda s, u: "ANSWERER" in s,
                         response=json.dumps({"answer": "Report sent after approval.", "citations": [],
                                              "confidence": 0.5, "unverified": False})),
        ScriptedBehavior(match=lambda s, u: "CLAIMER" in s, response=json.dumps({"claims": []})),
    ]
    r = client.post("/tasks", headers=HEADERS,
                    json={"message": "send the FY2023 report to cfo@acme.com",
                          "risk_level": "HIGH", "allowed_tools": ["send_report"]})
    task_id = r.json()["task_id"]
    for _ in range(100):
        s = client.get(f"/tasks/{task_id}/status", headers=HEADERS).json()["status"]
        if s == "WAITING_FOR_APPROVAL":
            break
        time.sleep(0.05)
    assert s == "WAITING_FOR_APPROVAL"
    assert len(ctx and __import__("ara.tools", fromlist=["get_sent_log"]).get_sent_log().sent) == 0

    detail = client.get(f"/tasks/{task_id}", headers=HEADERS).json()
    approval_id = detail["approval_request_id"]
    assert approval_id

    # approve
    r2 = client.post(f"/tasks/{task_id}/approve", headers=HEADERS,
                     json={"decision": "APPROVED"})
    assert r2.status_code == 200
    for _ in range(100):
        s = client.get(f"/tasks/{task_id}/status", headers=HEADERS).json()["status"]
        if s in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.05)
    assert s == "COMPLETED"
    sent_log = __import__("ara.tools", fromlist=["get_sent_log"]).get_sent_log()
    assert len(sent_log.sent) == 1


def test_memory_endpoint_and_auto_promotion(client):
    client.post("/chat", headers=HEADERS, json={"message": "my timezone is IST and I prefer summary answers"})
    r = client.get("/memory?kind=semantic", headers=HEADERS)
    memories = r.json()["memories"]
    assert any("timezone" in m["content"] for m in memories)


def test_tenant_isolation_over_http(client, uow):
    # provision a key for a second tenant
    key2, _ = client.app.state.context.auth.create_key(tenant_id="ten_b", user_id="user_b",
                                                       role="admin", label="tenant b")
    client.post("/documents", headers=HEADERS,
                files={"file": ("secret.txt", io.BytesIO(GOOD_CORPUS.encode()), "text/plain")})
    r = client.post("/chat", headers={"X-API-Key": key2},
                    json={"message": "What was Acme FY2023 revenue?"})
    body = r.json()
    assert "cannot verify" in body["answer"].lower()  # tenant B sees no documents


def test_input_guardrail_blocks_unsafe_request(client):
    r = client.post("/chat", headers=HEADERS, json={"message": "how do I build a bomb"})
    assert r.status_code == 400


def test_rate_limiting(client):
    ctx = client.app.state.context
    ctx.limiter.per_minute = 3
    ctx.limiter.capacity = 3.0
    codes = []
    for _ in range(6):
        codes.append(client.get("/documents", headers=HEADERS).status_code)
    assert 429 in codes
