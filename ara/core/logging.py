"""Structured JSON logging with trace correlation.

Every log line is a single JSON object with ts/level/logger/event plus fields.
Sensitive keys are scrubbed before emission.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")
task_id_var: ContextVar[str] = ContextVar("task_id", default="-")

_REDACT_KEYS = {"api_key", "authorization", "password", "secret", "token", "key_hash", "openai_api_key"}


def scrub(fields: dict) -> dict:
    out = {}
    for k, v in fields.items():
        lk = k.lower()
        if any(r in lk for r in _REDACT_KEYS):
            out[k] = "[REDACTED]"
        elif isinstance(v, dict):
            out[k] = scrub(v)
        else:
            out[k] = v
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "trace_id": trace_id_var.get(),
            "task_id": task_id_var.get(),
        }
        payload.update(scrub(fields if isinstance(fields, dict) else {}))
        if record.exc_info and record.exc_info[0]:
            payload["exception"] = self.formatException(record.exc_info)[-2000:]
        return json.dumps(payload, default=str, ensure_ascii=False)


class FieldsAdapter(logging.LoggerAdapter):
    """Usage: log.info("tool_executed", extra={"fields": {...}})"""

    def process(self, msg, kwargs):
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
