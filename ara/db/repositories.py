"""Repositories: the only SQL in the system. All tenant scoping enforced here."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ara.core.ids import iso_now, new_id
from ara.db.database import Database


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


class TaskRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, contract: dict, state: dict, tenant_id: str, user_id: str, trace_id: str, status: str) -> str:
        now = iso_now()
        self.db.execute(
            "INSERT INTO tasks (id, tenant_id, user_id, conversation_id, status, risk_level, trace_id,"
            " contract_json, state_json, final_answer, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (contract["task_id"], tenant_id, user_id, state.get("conversation_id"), status,
             contract.get("risk_level", "LOW"), trace_id, dumps(contract), dumps(state), None, now, now),
        )
        return contract["task_id"]

    def update_state(self, task_id: str, state: dict, status: str, final_answer: str | None = None) -> None:
        self.db.execute(
            "UPDATE tasks SET state_json=?, status=?, final_answer=COALESCE(?, final_answer), updated_at=? WHERE id=?",
            (dumps(state), status, final_answer, iso_now(), task_id),
        )

    def get(self, task_id: str, tenant_id: str) -> dict | None:
        return self.db.query_one("SELECT * FROM tasks WHERE id=? AND tenant_id=?", (task_id, tenant_id))

    def list(self, tenant_id: str, limit: int = 50) -> list[dict]:
        return self.db.query(
            "SELECT id,status,risk_level,created_at,updated_at FROM tasks WHERE tenant_id=?"
            " ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
        )

    def next_queued(self) -> dict | None:
        """Worker claim: atomically flip one QUEUED task to RUNNING."""
        with self.db._lock:
            row = self.db.query_one("SELECT * FROM tasks WHERE status='QUEUED' ORDER BY created_at LIMIT 1")
            if not row:
                return None
            self.db.execute("UPDATE tasks SET status='RUNNING', updated_at=? WHERE id=?", (iso_now(), row["id"]))
            return row


class DocumentRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, *, tenant_id: str, title: str, source: str, doc_type: str, pages: int,
               access: dict, content_hash: str) -> str:
        doc_id = new_id("doc")
        self.db.execute(
            "INSERT INTO documents (id, tenant_id, title, source, doc_type, version, pages, access_json,"
            " content_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (doc_id, tenant_id, title, source, doc_type, "v1", pages, dumps(access), content_hash, iso_now()),
        )
        return doc_id

    def add_page(self, document_id: str, page_number: int, image_path: str | None, text: str, meta: dict) -> str:
        pid = new_id("page")
        self.db.execute(
            "INSERT INTO document_pages (id, document_id, page_number, image_path, text_content, meta_json)"
            " VALUES (?,?,?,?,?,?)",
            (pid, document_id, page_number, image_path, text, dumps(meta)),
        )
        return pid

    def get(self, doc_id: str, tenant_id: str) -> dict | None:
        return self.db.query_one("SELECT * FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id))

    def pages(self, doc_id: str) -> list[dict]:
        return self.db.query("SELECT * FROM document_pages WHERE document_id=? ORDER BY page_number", (doc_id,))

    def list(self, tenant_id: str) -> list[dict]:
        return self.db.query(
            "SELECT id,title,source,doc_type,version,pages,created_at FROM documents WHERE tenant_id=?"
            " ORDER BY created_at DESC",
            (tenant_id,),
        )

    def delete(self, doc_id: str, tenant_id: str) -> None:
        self.db.execute("DELETE FROM document_pages WHERE document_id=?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id))


class EvidenceRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, task_id: str, ev: dict) -> None:
        self.db.execute(
            "INSERT INTO evidence (id, task_id, evidence_id, document_id, page, content_type, content,"
            " relevance_score, source_authority, meta_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (new_id("ev"), task_id, ev["evidence_id"], ev.get("document_id"), ev.get("page", 0),
             ev.get("content_type"), ev.get("content"), ev.get("relevance_score", 0),
             ev.get("source_authority", "unknown"), dumps(ev.get("meta", {})), iso_now()),
        )

    def for_task(self, task_id: str) -> list[dict]:
        return self.db.query("SELECT * FROM evidence WHERE task_id=?", (task_id,))


class ToolCallRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, task_id: str, rec: dict) -> None:
        self.db.execute(
            "INSERT INTO tool_calls (id, task_id, call_id, tool, args_json, result_json, status, risk_level,"
            " attempt, latency_ms, approved_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (new_id("ev"), task_id, rec.get("call_id"), rec.get("tool"), dumps(rec.get("args", {})),
             dumps(rec.get("result")), rec.get("status"), rec.get("risk_level", "LOW"),
             rec.get("attempt", 1), rec.get("latency_ms", 0), rec.get("approved_by"), iso_now()),
        )

    def find_idempotent(self, task_id: str, idem_key: str) -> dict | None:
        rows = self.db.query("SELECT * FROM tool_calls WHERE task_id=? AND call_id=?", (task_id, idem_key))
        return rows[0] if rows else None


class ApprovalRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, *, task_id: str, tool: str, args: dict, risk: str, reason: str, requested_by: str) -> str:
        aid = new_id("apr")
        self.db.execute(
            "INSERT INTO approvals (id, task_id, tool, args_json, risk_level, reason, status, requested_by,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, task_id, tool, dumps(args), risk, reason, "PENDING", requested_by, iso_now()),
        )
        return aid

    def get(self, approval_id: str, tenant_id: str) -> dict | None:
        return self.db.query_one(
            "SELECT a.* FROM approvals a JOIN tasks t ON t.id=a.task_id"
            " WHERE a.id=? AND t.tenant_id=?",
            (approval_id, tenant_id),
        )

    def decide(self, approval_id: str, decision: str, decided_by: str) -> None:
        self.db.execute(
            "UPDATE approvals SET status=?, decided_by=?, decided_at=? WHERE id=? AND status='PENDING'",
            (decision, decided_by, iso_now(), approval_id),
        )

    def pending_for_task(self, task_id: str) -> dict | None:
        return self.db.query_one(
            "SELECT * FROM approvals WHERE task_id=? AND status='PENDING' ORDER BY created_at DESC", (task_id,)
        )


class MemoryRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, *, tenant_id: str, user_id: str | None, kind: str, content: str, source: str,
            confidence: float, scope: str = "user", retention: str = "standard",
            expires_at: str | None = None, meta: dict | None = None) -> str:
        mid = new_id("mem")
        now = iso_now()
        self.db.execute(
            "INSERT INTO memories (id, tenant_id, user_id, kind, content, source, confidence, scope, retention,"
            " active, expires_at, meta_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,1,?,?,?,?)",
            (mid, tenant_id, user_id, kind, content, source, confidence, scope, retention, expires_at,
             dumps(meta or {}), now, now),
        )
        return mid

    def search(self, *, tenant_id: str, user_id: str | None, kind: str | None = None,
               query: str = "", limit: int = 8, include_expired: bool = False) -> list[dict]:
        sql = "SELECT * FROM memories WHERE tenant_id=? AND active=1"
        params: list[Any] = [tenant_id]
        if not include_expired:
            sql += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(iso_now())
        if user_id:
            sql += " AND (user_id=? OR user_id IS NULL)"
            params.append(user_id)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        rows = self.db.query(sql + " ORDER BY created_at DESC LIMIT 500", params)
        if not query:
            return rows[:limit]
        q_tokens = {t for t in query.lower().split() if len(t) > 2}

        def score(r: dict) -> float:
            toks = set(r["content"].lower().split())
            overlap = len(q_tokens & toks) / (len(q_tokens) or 1)
            return overlap * 0.7 + float(r.get("confidence") or 0) * 0.3

        rows.sort(key=score, reverse=True)
        return [r for r in rows if score(r) > 0][:limit]

    def deactivate(self, memory_id: str, tenant_id: str) -> None:
        self.db.execute("UPDATE memories SET active=0, updated_at=? WHERE id=? AND tenant_id=?",
                        (iso_now(), memory_id, tenant_id))


class AuditLog:
    """Append-only, tamper-evident (hash-chained) audit trail."""

    def __init__(self, db: Database):
        self.db = db

    def append(self, *, actor: str, action: str, subject: str | None = None,
               detail: dict | None = None, trace_id: str | None = None) -> str:
        prev = self.db.query_one("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        prev_hash = prev["entry_hash"] if prev else "GENESIS"
        payload = dumps({"actor": actor, "action": action, "subject": subject, "detail": detail or {}, "ts": iso_now()})
        entry_hash = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
        self.db.execute(
            "INSERT INTO audit_log (ts, trace_id, actor, action, subject, detail_json, prev_hash, entry_hash)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (iso_now(), trace_id, actor, action, subject, payload, prev_hash, entry_hash),
        )
        return entry_hash

    def verify_chain(self) -> bool:
        rows = self.db.query("SELECT * FROM audit_log ORDER BY id")
        prev = "GENESIS"
        for r in rows:
            # indexed columns must match the signed payload
            try:
                payload = json.loads(r["detail_json"])
            except (TypeError, ValueError):
                return False
            if payload.get("action") != r["action"] or payload.get("actor") != r["actor"]:
                return False
            expected = hashlib.sha256((prev + r["detail_json"]).encode()).hexdigest()
            if r["prev_hash"] != prev or r["entry_hash"] != expected:
                return False
            prev = r["entry_hash"]
        return True


class ConversationRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, *, tenant_id: str, user_id: str, title: str = "") -> str:
        cid = new_id("conv")
        self.db.execute(
            "INSERT INTO conversations (id, tenant_id, user_id, title, created_at) VALUES (?,?,?,?,?)",
            (cid, tenant_id, user_id, title, iso_now()),
        )
        return cid

    def add_message(self, conversation_id: str, role: str, content: str, meta: dict | None = None) -> str:
        mid = new_id("msg")
        self.db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, meta_json, created_at) VALUES (?,?,?,?,?,?)",
            (mid, conversation_id, role, content, dumps(meta or {}), iso_now()),
        )
        return mid

    def history(self, conversation_id: str, limit: int = 20) -> list[dict]:
        rows = self.db.query(
            "SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
            (conversation_id, limit),
        )
        return list(reversed(rows))


class EvalRunRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, dataset: str, metrics: dict) -> str:
        eid = new_id("eval")
        self.db.execute("INSERT INTO eval_runs (id, dataset, metrics_json, created_at) VALUES (?,?,?,?)",
                        (eid, dataset, dumps(metrics), iso_now()))
        return eid
