# ARA Architecture

## Layers

```
┌────────────────────────────────────────────────────────────────────┐
│ API layer (FastAPI)  auth · rate limit · REST · static UI          │
├────────────────────────────────────────────────────────────────────┤
│ Agent runtime (deterministic orchestrator)                         │
│   planner → state machine → budgets → recovery → finalize          │
├──────────────┬───────────────┬──────────────┬──────────────────────┤
│ Retrieval    │ Tools         │ Guardrails   │ Memory               │
│ ColPali      │ registry      │ injection    │ working/conv/        │
│ multi-vector │ pipeline      │ input/tool/  │ semantic/episodic    │
│ Qdrant/mem   │ MCP bridge    │ output       │                      │
├──────────────┴───────────────┴──────────────┴──────────────────────┤
│ Evidence manager · Verification engine (claims/numbers/dates)      │
├────────────────────────────────────────────────────────────────────┤
│ Policy engine (authz · risk · approvals) — deterministic           │
├────────────────────────────────────────────────────────────────────┤
│ Persistence (SQLite/Postgres) · Audit hash-chain · Traces          │
└────────────────────────────────────────────────────────────────────┘
```

## Trust boundaries (source-of-truth policy)

| Content | Trust | May ground claims? |
|---|---|---|
| System instructions / app policy | trusted (code + env) | n/a |
| User message | instruction source | no (it's a request, not evidence) |
| Application data (e.g. calculator results) | trusted app data | yes |
| Retrieved documents | **untrusted** | yes, via citation + verification |
| Web / MCP outputs | **untrusted** | yes, via citation + verification |
| Model-generated text | **never evidence** | no |

All untrusted content enters prompts only through
`wrap_untrusted_block()` (delimited, marked DATA-not-instructions).

## Control flow

1. **Contract** — every task carries a `TaskContract` (goal, constraints,
   allow/forbid lists, risk ceiling, budgets, verification requirements).
2. **Planning** — the LLM *proposes* a JSON plan; `Planner.validate_and_build`
   enforces legality deterministically (≤8 steps, known/allowed tools only,
   acyclic, single trailing `final_answer`). Deterministic shortcuts handle
   arithmetic / percent-change / clock without any model call. Invalid plans ⇒
   safe fallback (retrieve → answer).
3. **Execution loop** (`AgentRuntime._run_loop`):
   - `retrieve` → scoped visual retrieval → injection-scanned `Evidence`
   - `tool_call` → 10-stage pipeline (below)
   - `reason` → bounded LLM observation (stored as *observation*, never evidence)
   - `final_answer` → grounded draft → claim verification → output guardrail →
     citation assembly → conflict notices → refusal if unverifiable
   - Approval gate: `WAITING_FOR_APPROVAL` persists the full state; `resume()`
     continues exactly where it stopped, after the human decision.
4. **Verification** (`ara/evidence/verification.py`): per-claim checks —
   citation existence, admissibility (trust + injection), token support,
   **numbers must appear in cited evidence or verified calculations**,
   dates likewise. Unsupported claims are stripped; nothing supported ⇒ refusal.
5. **Conflicts**: numeric conflicts across sources resolved
   authority → version → recency; unresolved ⇒ both sides dropped + `CONFLICT NOTICE`.

## Tool pipeline (every call, model-proposed or planned)

```
proposal → existence → contract allowlist/forbidden → INPUT SCHEMA
→ AUTHORIZATION (role permissions) → RISK (static + argument-driven escalation)
→ APPROVAL if HIGH/CRITICAL (durable) → IDEMPOTENCY → EXECUTE (timeout+bounded
retry; non-idempotent side effects never blind-retried) → OUTPUT SCHEMA
→ OBSERVATION VERIFICATION → state update → AUDIT
```

Risk is escalated by argument content too (e.g. `delete` in args ⇒ HIGH),
independently of what the model asked for.

## Prompt-injection defense

- Normalization (NFKC, homoglyph folding, zero-width removal) defeats obfuscation.
- Weighted pattern families (override/role-hijack/tool-lure/secret-lure/exfil/encoding)
  produce a bounded score with a configurable threshold.
- **Sentence-level containment**: flagged pages keep their clean sentences as
  grounding content; attack sentences are quarantined (counted in evidence meta).
  Fully-hostile pages are inadmissible entirely.
- Output guardrail independently blocks fabricated citations, fabricated tool
  claims, secret leakage, and injection-grade answers.

## State & durability

`AgentState` (contract, plan, steps, evidence, tool results, verification,
errors, budgets) is persisted as JSON on **every transition**. Workers claim
`QUEUED` tasks atomically; approvals and budget states survive restarts.

## Observability

Per-run `Trace` (spans for plan/retrieve/tool/finalize + decision events) is
persisted with the task and exposed at `GET /tasks/{id}/trace` (admin). Structured
JSON logs carry `trace_id`/`task_id` with secret scrubbing. The audit log is a
SHA-256 hash chain with a `verify_chain()` integrity check. Cost is metered
deterministically from token counts × configured prices.

## Multimodal RAG

```
upload → page extraction (pypdf) → page images (pypdfium2/PIL)
  → colpali-service /embed/pages → multi-vector per page
  → vector store (multi-vector + tenant/permission metadata)
query → /embed/queries → MaxSim search → scope re-assertion
  → Evidence (scan + provenance) → reasoning model (only relevant, admissible)
```

The mock embedder is a hashing multi-vector lexical model with identical API
shape; real ColQwen2 activates via `COLPALI_MODE=real`.
