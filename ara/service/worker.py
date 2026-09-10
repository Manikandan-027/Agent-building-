"""Background worker: claims QUEUED tasks and executes them.

Runs in-process (API dev mode) or standalone (`python -m ara.service.worker`) in
docker-compose, where it can scale independently of the API.
"""
from __future__ import annotations

import json
import threading
import time

from ara.core.errors import AraError
from ara.core.logging import get_logger
from ara.core.tracing import Trace
from ara.core.types import TaskStatus
from ara.policy.authz import Principal

log = get_logger("ara.worker")


class TaskWorker:
    def __init__(self, ctx, poll_interval_s: float = 0.5):
        self.ctx = ctx
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="ara-worker", daemon=True)
        self._thread.start()
        log.info("worker_started")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                row = self.ctx.uow.tasks.next_queued()
                if not row:
                    self._stop.wait(self.poll_interval_s)
                    continue
                self._execute(row)
            except Exception as exc:  # noqa: BLE001 — worker must survive any task
                log.error("worker_loop_error", extra={"fields": {"error": str(exc)[:200]}})
                self._stop.wait(1.0)

    def _execute(self, row: dict) -> None:
        from ara.agent.state import AgentState, TaskContract

        task_id = row["id"]
        try:
            contract = TaskContract(**json.loads(row["contract_json"]))
            saved_state = json.loads(row["state_json"])
            state = AgentState.model_validate(saved_state) if saved_state.get("plan") else None
            principal = Principal.for_role(row["user_id"], row["tenant_id"],
                                           self._role_of(row["user_id"]))
            trace = Trace(trace_id=row["trace_id"])
            log.info("worker_executing_task", extra={"fields": {"task_id": task_id}})
            if state and state.status == TaskStatus.RUNNING:
                result = self.ctx.runtime._run_loop(state, principal, trace)
            else:
                result = self.ctx.runtime.execute(contract, principal, trace=trace)
            log.info("worker_task_finished",
                     extra={"fields": {"task_id": task_id, "status": result.status.value}})
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)[:300]
            log.error("worker_task_failed", extra={"fields": {"task_id": task_id, "error": detail}})
            try:
                saved = json.loads(row["state_json"])
                saved.setdefault("errors", []).append({"source": "worker", "message": detail})
                status = TaskStatus.FAILED.value if not isinstance(exc, AraError) else saved.get("status",
                                                                                               TaskStatus.FAILED.value)
                self.ctx.uow.tasks.update_state(task_id, saved, TaskStatus.FAILED.value,
                                                f"Task failed: {detail}")
            except Exception:  # noqa: BLE001
                pass

    def _role_of(self, user_id: str) -> str:
        row = self.ctx.uow.db.query_one("SELECT role FROM users WHERE id=?", (user_id,))
        return (row or {}).get("role", "user")


def main() -> None:  # standalone worker entrypoint (docker-compose)
    from ara.service.wiring import AppContext
    from ara.core.config import get_settings

    ctx = AppContext(get_settings())
    worker = TaskWorker(ctx)
    worker.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        worker.stop()
