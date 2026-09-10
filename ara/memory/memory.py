"""Memory architecture — four explicit categories with policies.

Working memory        : transient per-task scratch state (lives in AgentState).
Conversational memory : recent messages of the current conversation (bounded window).
Semantic memory       : durable user/application facts — stored ONLY when they pass
                        deterministic promotion rules (explicit user attribute
                        statements, high confidence, non-transient), deduplicated.
Episodic memory       : completed task episodes with outcome + evidence refs, used
                        for cross-task continuity ("last time you asked...").

Every record carries: source, timestamp, confidence, scope, retention, expiry.
Nothing is promoted to long-term memory automatically without passing policy.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from ara.core.ids import iso_now, utcnow
from ara.core.logging import get_logger
from ara.db import UnitOfWork

log = get_logger("ara.memory")

# Deterministic promotion rules for semantic memory.
PROMOTABLE_PATTERNS = [
    r"(?i)\bmy (name|email|phone|role|team|department|manager|timezone|currency|language)\s+is\b",
    r"(?i)\bi (work|am) (at|in|for)\b",
    r"(?i)\bprefer(s|ence|r)?\b.{0,40}\b(metric|imperial|currency|usd|eur|inr|gbp|json|table|bullet|summary|detail)\b",
    r"(?i)\b(?:our|my) (company|organization|tenant) is\b",
]
NON_PROMOTABLE = [
    r"(?i)\b(today|now|currently|temporarily|just for this|one[- ]time)\b",
    r"(?i)\b(password|secret|api[- ]?key|credential|token|otp)\b",   # never store secrets
]
DEFAULT_RETENTION_DAYS = {"standard": 180, "sensitive": 30, "durable": 730}


@dataclass
class MemoryRecord:
    memory_id: str
    kind: str                 # conversational | semantic | episodic
    content: str
    source: str               # user | agent | system | task_episode
    confidence: float
    scope: str                # user | tenant | task
    retention: str
    created_at: str
    expires_at: str | None
    meta: dict


class MemoryManager:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    # ------------------------------------------------------------ conversational
    def conversational_context(self, conversation_id: str | None, limit: int = 10) -> list[dict]:
        if not conversation_id:
            return []
        return self.uow.conversations.history(conversation_id, limit=limit)

    # ------------------------------------------------------------ semantic
    def should_promote_to_semantic(self, user_message: str) -> tuple[bool, str]:
        """Deterministic promotion decision. Never consults an LLM. Secrets NEVER."""
        import re

        for pattern in NON_PROMOTABLE:
            if re.search(pattern, user_message):
                return False, "non-promotable content (transient or sensitive)"
        for pattern in PROMOTABLE_PATTERNS:
            if re.search(pattern, user_message):
                return True, "matches durable user-attribute pattern"
        return False, "no durable attribute pattern"

    def remember_user_fact(self, *, tenant_id: str, user_id: str, content: str, source: str = "user",
                           confidence: float = 0.9, retention: str = "standard", meta: dict | None = None) -> str | None:
        ok, _ = self.should_promote_to_semantic(content)
        if not ok:
            return None
        norm = " ".join(content.lower().split())
        existing = self.uow.memories.search(tenant_id=tenant_id, user_id=user_id, kind="semantic",
                                            query=content, limit=5)
        for row in existing:
            row_norm = " ".join(row["content"].lower().split())
            same_fact = row_norm == norm
            same_slot = self._attribute_slot(row_norm) is not None and \
                self._attribute_slot(row_norm) == self._attribute_slot(norm)
            if same_fact or same_slot:
                self.uow.memories.deactivate(row["id"], tenant_id)  # replace stale slot
                break
        expires = self._expiry(retention)
        mid = self.uow.memories.add(tenant_id=tenant_id, user_id=user_id, kind="semantic", content=content,
                                    source=source, confidence=confidence, scope="user",
                                    retention=retention, expires_at=expires, meta=meta)
        log.info("semantic_memory_stored", extra={"fields": {"memory_id": mid}})
        return mid

    @staticmethod
    def _attribute_slot(content: str) -> str | None:
        """'my manager is Bob' -> 'my manager'; used to replace updated attributes."""
        if " is " in f" {content} ":
            head = content.split(" is ")[0].strip()
            if 0 < len(head.split()) <= 5:
                return head
        return None

    # ------------------------------------------------------------ episodic
    def record_episode(self, *, tenant_id: str, user_id: str, task_id: str, goal: str,
                       outcome: str, summary: str, evidence_ids: list[str] | None = None,
                       confidence: float = 0.85) -> str:
        return self.uow.memories.add(
            tenant_id=tenant_id, user_id=user_id, kind="episodic",
            content=f"Task {task_id} ({goal}) -> {outcome}: {summary}"[:1200],
            source="task_episode", confidence=confidence, scope="user", retention="durable",
            expires_at=self._expiry("durable"),
            meta={"task_id": task_id, "outcome": outcome, "evidence_ids": evidence_ids or []})

    # ------------------------------------------------------------ recall
    def recall(self, *, tenant_id: str, user_id: str | None, query: str,
               kinds: tuple[str, ...] = ("semantic", "episodic"), limit: int = 6) -> list[dict]:
        import json as _json

        out: list[dict] = []
        for kind in kinds:
            for row in self.uow.memories.search(tenant_id=tenant_id, user_id=user_id, kind=kind,
                                                query=query, limit=limit):
                try:
                    row["meta"] = _json.loads(row.pop("meta_json", "{}") or "{}")
                except ValueError:
                    row["meta"] = {}
                out.append(row)
        return out

    def context_for_task(self, *, tenant_id: str, user_id: str, query: str,
                         conversation_id: str | None) -> dict[str, Any]:
        conv = self.conversational_context(conversation_id)
        recalled = self.recall(tenant_id=tenant_id, user_id=user_id, query=query)
        return {"recent_messages": conv, "recalled_memories": [
            {"id": r["id"], "kind": r["kind"], "content": r["content"], "confidence": r["confidence"],
             "created_at": r["created_at"], "source": r["source"]} for r in recalled]}

    # ------------------------------------------------------------ retention
    @staticmethod
    def _expiry(retention: str) -> str | None:
        days = DEFAULT_RETENTION_DAYS.get(retention)
        if not days:
            return None
        return (utcnow() + dt.timedelta(days=days)).isoformat(timespec="seconds")

    def purge_expired(self) -> int:
        rows = self.uow.db.query("SELECT id, tenant_id FROM memories WHERE active=1 AND expires_at IS NOT NULL"
                                 " AND expires_at <= ?", (iso_now(),))
        for r in rows:
            self.uow.memories.deactivate(r["id"], r["tenant_id"])
        return len(rows)
