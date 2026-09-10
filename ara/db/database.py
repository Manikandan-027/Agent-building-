"""Durable storage: database adapter + repositories.

Design:
- One thin `Database` adapter over sqlite3 (dev/CI) or psycopg (production).
- DDL is portable across both dialects; JSON is stored as TEXT.
- Repositories are the ONLY code that touches SQL. Domain layers never do.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Iterable

from ara.core.config import get_settings
from ara.core.errors import AraError

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL DEFAULT 'user', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE, label TEXT, revoked INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL,
  title TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, meta_json TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, title TEXT NOT NULL,
  source TEXT, doc_type TEXT, version TEXT NOT NULL DEFAULT 'v1',
  pages INTEGER NOT NULL DEFAULT 0, access_json TEXT NOT NULL DEFAULT '{}',
  content_hash TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_pages (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL, page_number INTEGER NOT NULL,
  image_path TEXT, text_content TEXT, meta_json TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL,
  conversation_id TEXT, status TEXT NOT NULL, risk_level TEXT NOT NULL,
  trace_id TEXT, contract_json TEXT NOT NULL, state_json TEXT NOT NULL,
  final_answer TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
  document_id TEXT, page INTEGER DEFAULT 0, content_type TEXT,
  content TEXT, relevance_score REAL, source_authority TEXT, meta_json TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, call_id TEXT NOT NULL,
  tool TEXT NOT NULL, args_json TEXT, result_json TEXT, status TEXT,
  risk_level TEXT, attempt INTEGER DEFAULT 1, latency_ms INTEGER DEFAULT 0,
  approved_by TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, tool TEXT NOT NULL,
  args_json TEXT, risk_level TEXT NOT NULL, reason TEXT, status TEXT NOT NULL,
  requested_by TEXT, decided_by TEXT, decided_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT,
  kind TEXT NOT NULL, content TEXT NOT NULL, source TEXT,
  confidence REAL DEFAULT 0.5, scope TEXT DEFAULT 'user',
  retention TEXT DEFAULT 'standard', active INTEGER NOT NULL DEFAULT 1,
  expires_at TEXT, meta_json TEXT DEFAULT '{}', created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY {PK}, ts TEXT NOT NULL, trace_id TEXT, actor TEXT,
  action TEXT NOT NULL, subject TEXT, detail_json TEXT, prev_hash TEXT, entry_hash TEXT);
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY, dataset TEXT NOT NULL, metrics_json TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_task ON evidence(task_id);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(tenant_id, user_id, kind);
"""


class Database:
    """Thread-safe database adapter. `backend` is 'sqlite' or 'postgres'."""

    def __init__(self, url: str | None = None):
        self.url = url or get_settings().database_url
        self.backend = "postgres" if self.url.startswith(("postgres", "postgresql")) else "sqlite"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        if self.backend == "sqlite":
            import pathlib

            path = self.url.split("sqlite:///", 1)[-1]
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        else:
            try:
                import psycopg  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise AraError("psycopg not installed; pip install '.[prod]' for Postgres support") from exc
            self._psycopg = psycopg

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        with self._lock:
            if self.backend == "sqlite":
                assert self._conn is not None
                self._conn.executescript(SCHEMA.replace("{PK}", ""))
                self._conn.commit()
            else:
                self._pg_exec_script()

    def _pg_exec_script(self) -> None:  # pragma: no cover - requires PG
        with self._psycopg.connect(self.url) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.replace("{PK}", "BIGSERIAL"))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- primitives --------------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            if self.backend == "sqlite":
                assert self._conn is not None
                self._conn.execute(sql, tuple(params))
                self._conn.commit()
            else:  # pragma: no cover - requires PG
                with self._psycopg.connect(self.url) as conn, conn.cursor() as cur:
                    cur.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            if self.backend == "sqlite":
                assert self._conn is not None
                rows = self._conn.execute(sql, tuple(params)).fetchall()
                return [dict(r) for r in rows]
            with self._psycopg.connect(self.url) as conn, conn.cursor() as cur:  # pragma: no cover
                cur.execute(sql, tuple(params))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None
