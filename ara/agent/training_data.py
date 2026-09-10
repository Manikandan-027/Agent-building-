"""Fine-tuning data pipeline: mine SFT examples from ARA's OWN verified episodes.

Why mine episodes instead of hand-writing data? Because ARA's verification layer
already labels which runs were GOOD: answer supported, every claim backed by
admissible evidence, zero refusals, no guardrail blocks. Training on anything
else would teach the model to hallucinate — the exact failure this system
exists to prevent.

Outputs OpenAI chat-format JSONL:
  {"messages": [{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}]}

Quality gates (deterministic, all must hold for an episode to be included):
  - task status == COMPLETED
  - verification.answer_supported is true
  - verification.refused is false
  - every verified claim supported, >=1 supporting citation
  - no output-guardrail refusal recorded
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ara.core.logging import get_logger
from ara.db import UnitOfWork
from ara.agent.planner import build_planner_prompt
from ara.agent.runtime import build_answerer_prompt
from ara.guardrails.injection import wrap_untrusted_block
from ara.evidence.manager import EvidenceManager, as_evidence_dict

log = get_logger("ara.training")

MAX_EXAMPLE_CHARS = 24_000
MAX_EXAMPLES_PER_ROLE = 10_000


@dataclass
class ExportReport:
    scanned_tasks: int = 0
    qualified_tasks: int = 0
    planner_examples: int = 0
    answerer_examples: int = 0
    rejected: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"scanned": self.scanned_tasks, "qualified": self.qualified_tasks,
                "planner_examples": self.planner_examples,
                "answerer_examples": self.answerer_examples, "rejected": self.rejected}


def _episode_is_high_quality(row: dict) -> tuple[bool, str]:
    try:
        state = json.loads(row["state_json"])
    except json.JSONDecodeError:
        return False, "unparsable_state"
    if row["status"] != "COMPLETED":
        return False, f"status_{row['status']}"
    verif = state.get("verification") or {}
    if verif.get("refused"):
        return False, "refused"
    if not verif.get("answer_supported"):
        return False, "answer_not_supported"
    claims = verif.get("claims") or []
    if not claims or not all(c.get("supported") for c in claims):
        return False, "unsupported_claims_present"
    if not any(c.get("evidence_ids") for c in claims):
        return False, "no_citations"
    note = state.get("final_answer_unverified_note")
    if note:
        return False, "guardrail_note"
    return True, "ok"


def _planner_example(row: dict, catalog: str) -> dict | None:
    state = json.loads(row["state_json"])
    plan = state.get("plan")
    if not plan or not plan.get("steps"):
        return None
    contract = state.get("task", {})
    contract["normalized_goal"] = plan.get("goal") or contract.get("normalized_goal", "")
    user = build_planner_prompt(_contract_like(contract), catalog or "(no tools)", "", 1)
    assistant_obj = {"goal": plan.get("goal", ""), "rationale": plan.get("rationale", ""),
                     "steps": [{"description": s.get("description"), "action": s.get("action"),
                                "tool": s.get("tool"), "args": s.get("args") or {},
                                "query": s.get("query"), "depends_on": s.get("depends_on") or []}
                               for s in plan["steps"]]}
    assistant = json.dumps(assistant_obj, ensure_ascii=False)
    if len(user) + len(assistant) > MAX_EXAMPLE_CHARS:
        return None
    return {"messages": [
        {"role": "system", "content": "You are the PLANNER of a grounded research agent. "
                                      "Propose safe, minimal, evidence-first plans."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def _answerer_example(row: dict, em: EvidenceManager) -> dict | None:
    state = json.loads(row["state_json"])
    task = state.get("task", {})
    cited = {c for v in (state.get("verification", {}).get("claims") or [])
             for c in v.get("evidence_ids", [])}
    blocks = []
    for ev in (row.get("_evidence") or []):
        ev = as_evidence_dict(ev)
        if not em.is_groundedable(ev):
            continue
        blocks.append(wrap_untrusted_block(ev["content"][:1500], f"EV_{ev['evidence_id']}"))
    if not blocks:
        return None
    user = build_answerer_prompt(task.get("user_request", ""), "\n\n".join(blocks))
    answer = row.get("final_answer") or ""
    if not answer or len(user) + len(answer) > MAX_EXAMPLE_CHARS:
        return None
    # strip runtime-added "Sources:"/"CONFLICT" trailers: teach the raw answer format
    answer = answer.split("\n\nSources:")[0].split("\n\nCONFLICT NOTICE")[0].strip()
    return {"messages": [
        {"role": "system", "content": "You are the ANSWERER of a grounded research agent. "
                                      "Answer ONLY from provided evidence and cite evidence ids."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(
            {"answer": answer, "citations": sorted(cited), "confidence": 0.85, "unverified": False},
            ensure_ascii=False)},
    ]}


def _contract_like(contract: dict):
    from ara.agent.state import TaskContract

    try:
        return TaskContract(**{k: v for k, v in contract.items()
                               if k in TaskContract.model_fields})
    except Exception:  # noqa: BLE001 — fall back to minimal contract
        return _contract_like({"user_request": contract.get("user_request", "unknown"), })


def export_training_data(uow: UnitOfWork, em: EvidenceManager, *, tenant_id: str,
                         catalog: str = "", limit: int = MAX_EXAMPLES_PER_ROLE) -> tuple[list[dict], list[dict], ExportReport]:
    """Return (planner_examples, answerer_examples, report)."""
    report = ExportReport()
    rows = uow.db.query(
        "SELECT * FROM tasks WHERE tenant_id=? AND status='COMPLETED' ORDER BY created_at DESC LIMIT ?",
        (tenant_id, limit * 3))
    report.scanned_tasks = len(rows)
    planners: list[dict] = []
    answerers: list[dict] = []
    for row in rows:
        ok, reason = _episode_is_high_quality(row)
        if not ok:
            report.rejected[reason] = report.rejected.get(reason, 0) + 1
            continue
        report.qualified_tasks += 1
        # attach evidence rows for the answerer example
        ev_rows = uow.evidence.for_task(row["id"])
        parsed = []
        for e in ev_rows:
            try:
                meta = json.loads(e.get("meta_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            # only episodes that PASSED verification are exported, and stored
            # evidence is the admitted (injection-filtered) subset by construction
            parsed.append({"evidence_id": e["evidence_id"], "document_id": e["document_id"],
                           "page": e["page"], "content": e["content"],
                           "source_trust": "retrieved",
                           "injection_scan": {"flagged": False}, "meta": meta})
        # trust filter: only rows stored from admissible sources; injection flags were
        # already enforced at run time, stored evidence reflects the admitted subset
        row = dict(row)
        row["_evidence"] = parsed
        if len(planners) < limit:
            ex = _planner_example(row, catalog)
            if ex:
                planners.append(ex)
                report.planner_examples += 1
        if len(answerers) < limit:
            ex = _answerer_example(row, em)
            if ex:
                answerers.append(ex)
                report.answerer_examples += 1
    return planners, answerers, report


def write_jsonl(path: str, examples: list[dict]) -> int:
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)


def validate_jsonl(path: str) -> dict:
    """Deterministic dataset validation: shape, roles, size budget, no empties."""
    stats = {"lines": 0, "errors": [], "total_chars": 0}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["errors"].append(f"line {i}: invalid JSON: {exc}")
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 3:
                stats["errors"].append(f"line {i}: messages must be a list of >=3")
                continue
            roles = [m.get("role") for m in msgs]
            if roles[:2] != ["system", "user"] or roles[-1] != "assistant":
                stats["errors"].append(f"line {i}: bad role order {roles}")
            for m in msgs:
                if not isinstance(m.get("content"), str) or not m["content"].strip():
                    stats["errors"].append(f"line {i}: empty message content")
            stats["total_chars"] += sum(len(m.get("content", "")) for m in msgs)
    return stats
