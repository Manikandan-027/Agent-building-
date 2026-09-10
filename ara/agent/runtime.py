"""AgentRuntime: the deterministic orchestrator.

CONTROL FLOW (LLM proposes; runtime decides):

  normalize goal -> recall memory -> propose+validate plan
  -> while steps remain AND budgets hold:
       retrieve   -> RetrievalEngine (scoped, injection-scanned) -> Evidence[]
       tool_call  -> ToolPipeline (validate/authz/risk/approval/execute/verify)
                     [approval needed -> persist & suspend WAITING_FOR_APPROVAL]
       reason     -> bounded LLM observation (model text is NEVER evidence)
       final      -> evidence-grounded draft
                     -> claim verification (numbers/dates/citations/conflicts)
                     -> output guardrail (fabricated citations/tools, secrets)
                     -> refuse, or finalize with citations + conflict notices
  -> episodic memory, audit, trace persistence

Re-planning is bounded (max_replans) and triggered when evidence invalidates the
plan. Every transition is persisted: crash-safe, approval-resumable.
"""
from __future__ import annotations

import json
import re

from ara.agent.budgets import BudgetState
from ara.agent.llm import LLMProvider, parse_json_object
from ara.agent.planner import Planner
from ara.agent.state import AgentState, Evidence, Plan, PlanStep, TaskContract, VerificationReport
from ara.core.errors import NotFoundError
from ara.core.logging import get_logger
from ara.core.tracing import Trace
from ara.core.types import ContentTrust, StepAction, StepStatus, TaskStatus
from ara.db import UnitOfWork
from ara.evidence.manager import EvidenceManager
from ara.evidence.verification import VerificationEngine
from ara.guardrails.guardrails import OutputGuardrail
from ara.guardrails.injection import wrap_untrusted_block
from ara.memory.memory import MemoryManager
from ara.policy.authz import ApprovalRequiredSignal, PolicyEngine, Principal
from ara.retrieval.engine import RetrievalEngine
from ara.tools.pipeline import PipelineContext, ToolPipeline
from ara.tools.registry import ToolRegistry

log = get_logger("ara.runtime")


class TaskResult:
    def __init__(self, *, task_id: str, status: TaskStatus, answer: str | None,
                 state: AgentState, trace: Trace, error: str | None = None):
        self.task_id = task_id
        self.status = status
        self.answer = answer
        self.state = state
        self.trace = trace
        self.error = error

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "status": self.status.value, "answer": self.answer,
            "error": self.error,
            "evidence_ids": [e["evidence_id"] for e in self.state.evidence],
            "verification": self.state.verification.model_dump(),
            "budget": self.state.scratch.get("budget"),
            "trace_id": self.trace.trace_id,
        }


class AgentRuntime:
    def __init__(self, *, uow: UnitOfWork, llm: LLMProvider, registry: ToolRegistry,
                 retrieval: RetrievalEngine, evidence_manager: EvidenceManager,
                 verifier: VerificationEngine, output_guardrail: OutputGuardrail,
                 policy_engine: PolicyEngine, memory: MemoryManager, settings):
        self.uow = uow
        self.llm = llm
        self.registry = registry
        self.retrieval = retrieval
        self.em = evidence_manager
        self.verifier = verifier
        self.output_guardrail = output_guardrail
        self.policy = policy_engine
        self.memory = memory
        self.settings = settings
        self.planner = Planner(llm, registry)
        self.pipeline = ToolPipeline()

    # ================================================================ entry points
    def execute(self, contract: TaskContract, principal: Principal, *,
                conversation_id: str | None = None, trace: Trace | None = None) -> TaskResult:
        trace = trace or Trace()
        state = AgentState(task=contract, tenant_id=principal.tenant_id, user_id=principal.user_id,
                           conversation_id=conversation_id, status=TaskStatus.PLANNING)
        contract.normalized_goal = contract.normalized_goal or contract.user_request[:300]

        # semantic promotion attempt: policy decides (deterministic, never stores secrets)
        self.memory.remember_user_fact(tenant_id=principal.tenant_id, user_id=principal.user_id,
                                       content=contract.user_request)
        mem_ctx = self.memory.context_for_task(tenant_id=principal.tenant_id, user_id=principal.user_id,
                                               query=contract.user_request, conversation_id=conversation_id)
        state.memory_context = mem_ctx.get("recalled_memories", [])

        with trace.span("plan"):
            catalog = self._tool_catalog(contract, principal)
            plan = self.planner.plan(contract, catalog,
                                     json.dumps(mem_ctx.get("recalled_memories", [])[:3], default=str),
                                     corpus_size=self._corpus_size())
            state.plan = plan
            state.status = TaskStatus.RUNNING
        trace.event("plan_validated", steps=len(plan.steps), goal=plan.goal[:120])

        self._persist(state, principal, trace)
        return self._run_loop(state, principal, trace)

    def resume(self, task_id: str, principal: Principal, *, approval_id: str,
               decision: str, trace: Trace | None = None) -> TaskResult:
        """Resume a WAITING_FOR_APPROVAL task after a human decision (durable state)."""
        trace = trace or Trace()
        row = self.uow.tasks.get(task_id, principal.tenant_id)
        if not row:
            raise NotFoundError(f"task {task_id} not found")
        state = AgentState.model_validate_json(row["state_json"])
        with trace.span("resume", approval_id=approval_id, decision=decision):
            self.uow.approvals.decide(approval_id, decision, principal.user_id)
            self.uow.audit.append(actor=principal.user_id, action=f"approval_{decision.lower()}",
                                  subject=approval_id, detail={"task_id": task_id}, trace_id=trace.trace_id)
            state.approval_status = decision
            step = next((s for s in (state.plan.steps if state.plan else [])
                         if s.status == StepStatus.WAITING_APPROVAL), None)
            if step is None:
                state.add_error("resume", "no step waiting for approval")
            elif decision == "APPROVED":
                step.status = StepStatus.PENDING
                state.scratch["approved_approval_id"] = approval_id
            else:
                step.status = StepStatus.FAILED
                step.result_summary = "human rejected the proposed action"
                state.completed_steps.append(step.step_id)
        state.status = TaskStatus.RUNNING
        self._persist(state, principal, trace)
        return self._run_loop(state, principal, trace)

    # ================================================================ main loop
    def _run_loop(self, state: AgentState, principal: Principal, trace: Trace) -> TaskResult:
        saved = state.scratch.get("_budgets")
        budget = (BudgetState.restore(saved, self.settings.price_input_per_1k, self.settings.price_output_per_1k)
                  if saved else
                  BudgetState.from_contract(state.task.execution_budget,
                                            self.settings.price_input_per_1k, self.settings.price_output_per_1k))

        def persist() -> None:
            state.scratch["_budgets"] = budget.serialize()
            self._persist(state, principal, trace)

        while True:
            try:
                budget.check_iteration()
            except Exception as exc:
                return self._budget_stop(state, principal, trace, str(exc), budget)

            step = state.pending_step()
            if step is None:
                break

            state.current_step_id = step.step_id
            step.status = StepStatus.RUNNING
            try:
                stop = self._dispatch(step, state, principal, trace, budget)
            except ApprovalRequiredSignal as sig:
                # durable suspension — resume via runtime.resume() after human decision
                step.status = StepStatus.WAITING_APPROVAL
                state.status = TaskStatus.WAITING_FOR_APPROVAL
                state.approval_request_id = sig.approval_id
                state.approval_status = "PENDING"
                persist()
                return TaskResult(task_id=state.task.task_id, status=TaskStatus.WAITING_FOR_APPROVAL,
                                  answer=None, state=state, trace=trace,
                                  error=f"waiting for approval {sig.approval_id}: {sig.reason}")
            except Exception as exc:  # noqa: BLE001 — a step failure must never crash the run
                state.add_error("runtime", f"{type(exc).__name__}: {exc}", step=step.step_id)
                trace.event("step_failed", step=step.step_id, error=str(exc)[:200])
                step.status = StepStatus.FAILED
                state.completed_steps.append(step.step_id)
                continue
            persist()
            if stop == "stop":
                break

        try:
            return self._finalize(state, principal, trace, budget)
        except Exception as exc:
            from ara.core.errors import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                return self._budget_stop(state, principal, trace, str(exc), budget)
            raise

    # ================================================================ dispatch
    def _dispatch(self, step: PlanStep, state: AgentState, principal: Principal,
                  trace: Trace, budget: BudgetState) -> str | None:
        if step.action == StepAction.RETRIEVE:
            with trace.span("retrieve", query=(step.query or "")[:120]):
                result = self.retrieval.retrieve(query=step.query or state.task.user_request,
                                                 tenant_id=principal.tenant_id,
                                                 role=principal.role, user_id=principal.user_id)
                admitted = 0
                existing = {e.evidence_id for e in state.evidence}
                for ev in result.evidence:
                    if ev["evidence_id"] not in existing:
                        state.evidence.append(Evidence(**ev))
                        self.uow.evidence.add(state.task.task_id, ev)
                        admitted += 1
                trace.event("retrieved", admitted=admitted, confidence=result.confidence)
                step.result_summary = f"retrieved {admitted} evidence (confidence {result.confidence})"
                step.status = StepStatus.DONE
                state.completed_steps.append(step.step_id)
            return None

        if step.action == StepAction.TOOL_CALL:
            budget.pre_tool()
            ctx = self._pipeline_context(state, principal, trace)
            approved_id = state.scratch.pop("approved_approval_id", None)
            with trace.span("tool_call", tool=step.tool):
                outcome = self.pipeline.execute_with_fallback(
                    step.tool, dict(step.args or {}), ctx, approved_approval_id=approved_id)
                rec = outcome.record
                state.tool_results.append(rec)
                self.uow.tool_calls.add(state.task.task_id, rec.model_dump())
                if rec.status == "ok":
                    if rec.tool == "calculator":
                        ev = self.em.make(
                            content=json.dumps({"calculation": (step.args or {}).get("expression"),
                                                "result": rec.result.get("result"),
                                                "result_str": rec.result.get("result_str")}),
                            document_id="calculator", page=0, content_type="calculation",
                            trust=ContentTrust.APP_DATA, source="calculation", relevance_score=1.0,
                            tenant_id=principal.tenant_id)
                        ev["evidence_id"] = rec.call_id
                        state.evidence.append(Evidence(**ev))
                        self.uow.evidence.add(state.task.task_id, ev)
                    step.result_summary = f"{rec.tool} ok"
                    step.status = StepStatus.DONE
                    state.completed_steps.append(step.step_id)
                elif rec.status in {"error", "timeout"}:
                    step.result_summary = f"{rec.tool} failed: {rec.error}"
                    step.status = StepStatus.FAILED
                    state.completed_steps.append(step.step_id)
                    trace.event("tool_failed", tool=rec.tool, error=(rec.error or "")[:200])
                    self._maybe_replan(state, principal, trace, reason=f"tool {rec.tool} failed")
                # 'rejected' records are terminal for the step: contract/policy refused it
                else:
                    step.result_summary = f"{rec.tool} rejected: {rec.error}"
                    step.status = StepStatus.FAILED
                    state.completed_steps.append(step.step_id)
            return None

        if step.action == StepAction.REASON:
            budget.pre_llm()
            with trace.span("reason"):
                observation = self._reason_step(step, state)
                state.scratch.setdefault("observations", []).append(observation[:1200])
                step.result_summary = observation[:200]
                step.status = StepStatus.DONE
                state.completed_steps.append(step.step_id)
            return None

        if step.action == StepAction.WAIT_APPROVAL:
            step.status = StepStatus.DONE
            state.completed_steps.append(step.step_id)
            return None

        if step.action == StepAction.FINAL_ANSWER:
            step.status = StepStatus.DONE
            state.completed_steps.append(step.step_id)
            return "stop"
        return None

    # ================================================================ answering
    def _reason_step(self, step: PlanStep, state: AgentState) -> str:
        prompt = (
            f"REQUEST: {state.task.user_request}\n"
            f"STEP INSTRUCTION: {step.instruction or step.description}\n"
            "EVIDENCE:\n" + self._render_evidence(state) +
            "\nReason briefly over the evidence. Do not assert unsupported facts."
        )
        resp = self.llm.complete("REASONER", prompt, max_tokens=400)
        return resp.text.strip()

    def _draft_answer(self, state: AgentState) -> tuple[str, list[str]]:
        evidence_blocks = "\n\n".join(
            wrap_untrusted_block(e.content[:1500], f"EV_{e.evidence_id}")
            for e in state.evidence if self.em.is_groundedable(e))
        prompt = build_answerer_prompt(state.task.user_request, evidence_blocks,
                                       state.scratch.get("observations", [])[-3:])
        resp = self.llm.complete("ANSWERER", prompt, json_mode=True, max_tokens=900)
        try:
            data = parse_json_object(resp.text)
            answer = str(data.get("answer", "")).strip()
            cites = [str(c) for c in data.get("citations", [])][:12]
        except Exception:
            answer, cites = resp.text.strip(), []
        return (answer or "I could not produce an answer from the available evidence."), cites

    def _finalize(self, state: AgentState, principal: Principal, trace: Trace,
                  budget: BudgetState) -> TaskResult:
        with trace.span("finalize"):
            budget.pre_llm()
            try:
                answer, proposed_citations = self._draft_answer(state)
            except Exception as exc:
                state.add_error("draft", str(exc))
                answer, proposed_citations = "I could not produce an answer due to an internal error.", []

            # claim-level verification: LLM proposes claim->evidence map, runtime validates
            llm_claim_map = None
            try:
                cite_hint = f"The draft was produced citing these evidence ids: {proposed_citations}. " \
                    if proposed_citations else ""
                claim_prompt = ("CLAIMER\nExtract claims from the draft, each with the evidence ids "
                                f"cited for it.\n{cite_hint}DRAFT: {answer}")
                llm_claim_map = parse_json_object(self.llm.complete("CLAIMER", claim_prompt,
                                                                    json_mode=True, max_tokens=500).text)
            except Exception:
                llm_claim_map = None
            calc_results = [r.result for r in state.tool_results
                            if r.tool == "calculator" and r.status == "ok" and isinstance(r.result, dict)]
            report = self.verifier.verify(answer, [e.model_dump() for e in state.evidence],
                                          state.task.verification_requirements,
                                          calculator_results=calc_results, llm_proposal=llm_claim_map,
                                          default_citations=proposed_citations)
            state.verification = VerificationReport(**report)

            # strip unsupported claims unless refusing outright
            if not report["refused"] and report["unsupported_claims"]:
                bad = {u.strip().lower()[:60] for u in report["unsupported_claims"]}
                kept = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer)
                        if s.strip() and not any(s.strip().lower().startswith(b) or b.startswith(s.strip().lower()[:60])
                                                 for b in bad)]
                answer = " ".join(kept) or ("The retrieved evidence was insufficient to verify an answer, "
                                            "so I am not providing one.")

            # conflict resolution: drop sentences citing the losing side of a
            # resolved conflict (authority > version > recency); if UNRESOLVED,
            # drop BOTH sides — never silently prefer a convenient value.
            losers: set[str] = set()
            for conflict in report["conflicts"]:
                pref = conflict["resolution"].get("prefer")
                sides = {conflict["evidence_a"], conflict["evidence_b"]}
                losers |= (sides - {pref}) if pref else sides
            if losers:
                kept = []
                for sent in re.split(r"(?<=[.!?])\s+", answer):
                    cited_here = set(re.findall(r"\b(?:ev|call)_[a-z0-9_]+\b", sent))
                    if cited_here & losers:
                        continue
                    kept.append(sent)
                answer = " ".join(kept) or (
                    "The sources disagree on the key values and the conflict could not be "
                    "resolved deterministically, so I am not asserting either figure.")

            # output guardrail: fabricated citations, fabricated tool claims, secret leaks
            valid_ids = {e.evidence_id for e in state.evidence}
            tool_ids = {r.call_id for r in state.tool_results}
            cited_ids = [c for v in report["claims"] for c in v["evidence_ids"]]
            guard = self.output_guardrail.check(answer, citations_used=cited_ids,
                                                valid_evidence_ids=valid_ids, tool_call_ids=tool_ids)
            if not guard.allowed:
                answer = self.output_guardrail.sanitize(answer)
                if "leaks" in guard.reason or "injection" in guard.reason:
                    answer = (f"The generated answer was blocked by the output guardrail "
                              f"({guard.reason}) and has been withheld.")
                    state.verification.refused = True
                    state.verification.refusal_reason = guard.reason
                    report["refused"] = True

            # final assembly: provenance + conflict notices + refusal
            final = answer
            sources = [e.citation for e in state.evidence if e.evidence_id in set(cited_ids)]
            if sources and not report["refused"]:
                final += "\n\nSources: " + "; ".join(dict.fromkeys(sources))
            if report["conflicts"]:
                cl = report["conflicts"][0]
                pref = cl["resolution"].get("prefer")
                final += (f"\n\nCONFLICT NOTICE: sources {cl['evidence_a']} and {cl['evidence_b']} "
                          f"disagree (resolution: {cl['resolution']['strategy']}"
                          + (f", preferring {pref}" if pref else " — unresolved, verify manually") + ").")
            if report["refused"]:
                final = (f"I cannot verify this from the available evidence. "
                         f"Reason: {report['refusal_reason']}")

            state.final_answer = final
            if report["refused"]:
                state.final_answer_unverified_note = report["refusal_reason"]
            state.status = (TaskStatus.FAILED
                            if (state.task.failure_conditions and report["refused"])
                            else TaskStatus.COMPLETED)

            self.memory.record_episode(
                tenant_id=principal.tenant_id, user_id=principal.user_id, task_id=state.task.task_id,
                goal=state.task.user_request[:200], outcome=state.status.value,
                summary=(state.final_answer or "")[:400],
                evidence_ids=cited_ids[:10], confidence=0.9 if not report["refused"] else 0.5)

            state.scratch["budget"] = budget.to_dict()
            state.scratch["trace"] = trace.to_dict()
            self._persist(state, principal, trace)
            self.uow.audit.append(actor="runtime", action="task_finalized", subject=state.task.task_id,
                                  detail={"status": state.status.value, "refused": report["refused"]},
                                  trace_id=trace.trace_id)
            return TaskResult(task_id=state.task.task_id, status=state.status, answer=final,
                              state=state, trace=trace)

    def _budget_stop(self, state: AgentState, principal: Principal, trace: Trace, reason: str,
                     budget: BudgetState | None = None) -> TaskResult:
        state.status = TaskStatus.BUDGET_STOPPED
        state.final_answer = (f"I stopped safely before completing this task: {reason}. "
                              "No unverified answer is provided. Retry with a higher budget or a "
                              "narrower request.")
        state.add_error("budget", reason)
        state.scratch["budget"] = budget.serialize() if budget else {"stop_reason": reason}
        state.scratch["trace"] = trace.to_dict()
        self._persist(state, principal, trace)
        self.uow.audit.append(actor="runtime", action="task_budget_stopped", subject=state.task.task_id,
                              detail={"reason": reason}, trace_id=trace.trace_id)
        return TaskResult(task_id=state.task.task_id, status=TaskStatus.BUDGET_STOPPED,
                          answer=state.final_answer, state=state, trace=trace, error=reason)

    # ================================================================ replan
    def _maybe_replan(self, state: AgentState, principal: Principal, trace: Trace, reason: str) -> None:
        if state.replans >= state.task.execution_budget.max_replans:
            trace.event("replan_skipped", reason="max replans reached", trigger=reason[:100])
            return
        state.replans += 1
        with trace.span("replan", version=state.replans, trigger=reason[:100]):
            contract = state.task
            catalog = self._tool_catalog(contract, principal)
            observations = json.dumps(state.scratch.get("observations", [])[-3:], default=str)
            try:
                plan = self.planner.propose(contract, catalog, f"prior observations: {observations}",
                                        corpus_size=self._corpus_size())
            except Exception as exc:  # noqa: BLE001
                trace.event("replan_failed", error=str(exc)[:150])
                return
            plan.steps = [s for s in plan.steps if s.action != StepAction.FINAL_ANSWER][-6:]
            plan.steps.append(PlanStep(description="Produce the final grounded answer",
                                       action=StepAction.FINAL_ANSWER))
            old = state.plan
            state.plan = Plan(goal=(old.goal if old else contract.user_request),
                              rationale=f"replan {state.replans}: {reason}",
                              version=(old.version if old else 1) + 1)
            state.plan.steps = plan.steps
        trace.event("replanned", version=state.plan.version)

    # ================================================================ plumbing
    def _corpus_size(self) -> int:
        try:
            return self.retrieval.store.count()
        except Exception:  # noqa: BLE001
            return 0

    def _tool_catalog(self, contract: TaskContract, principal: Principal) -> str:
        perms = {"tools:low", "tools:medium"} | ({"*"} if principal.role == "admin" else set())
        return self.registry.catalog_for_prompt(self.registry.visible_to(contract.allowed_tools or None, perms))

    def _pipeline_context(self, state: AgentState, principal: Principal, trace: Trace) -> PipelineContext:
        def create_approval(tool: str, args: dict, risk, reason: str) -> str:
            aid = self.uow.approvals.create(task_id=state.task.task_id, tool=tool, args=args,
                                            risk=risk.value, reason=reason, requested_by=principal.user_id)
            self.uow.audit.append(actor=principal.user_id, action="approval_requested", subject=aid,
                                  detail={"tool": tool, "risk": risk.value, "reason": reason},
                                  trace_id=trace.trace_id)
            return aid

        def get_decision(aid: str) -> str | None:
            row = self.uow.approvals.get(aid, principal.tenant_id)
            return row["status"] if row else None

        return PipelineContext(
            principal=principal, contract=state.task, registry=self.registry, policy=self.policy,
            create_approval=create_approval, get_approval_decision=get_decision,
            audit=lambda **kw: self.uow.audit.append(actor=principal.user_id, trace_id=trace.trace_id, **kw),
            trace=trace,
        )

    def _render_evidence(self, state: AgentState) -> str:
        blocks = [f"[{e.evidence_id}] ({e.document_id} p{e.page}) {e.content[:600]}"
                  for e in state.evidence if self.em.is_groundedable(e)]
        return "\n".join(blocks) or "(none)"

    def _persist(self, state: AgentState, principal: Principal, trace: Trace) -> None:
        payload = json.loads(state.model_dump_json())
        row = self.uow.tasks.get(state.task.task_id, principal.tenant_id)
        if not row:
            self.uow.tasks.create(contract=state.task.model_dump(), state=payload,
                                  tenant_id=principal.tenant_id, user_id=principal.user_id,
                                  trace_id=trace.trace_id, status=state.status.value)
        else:
            self.uow.tasks.update_state(state.task.task_id, payload, state.status.value,
                                        state.final_answer)


def build_answerer_prompt(user_request: str, evidence_blocks: str,
                          observations: list | None = None) -> str:
    """Single source of truth for the answerer prompt (inference + fine-tuning)."""
    return (
        f"ANSWERER\nREQUEST: {user_request}\n\n"
        "EVIDENCE (cite evidence ids exactly as given; if insufficient, answer NOT VERIFIABLE):\n"
        f"{evidence_blocks or '(no admissible evidence)'}\n\n"
        + (f"OBSERVATIONS:\n{json.dumps(observations or [])}\n" if observations else "")
        + '\nReturn ONLY JSON: {"answer": str, "citations": [evidence ids], '
          '"confidence": float, "unverified": bool}'
    )
