# ARA Security Model

## Authentication & authorization
- API keys: stored **only as SHA-256 hashes**; provisioning shows plaintext once.
- Roles: `user` < `admin` with least-privilege permission sets; tools declare the
  permissions they require; MCP tools inherit governed permissions.
- **Tenant isolation**: every document, task, evidence, memory and approval row is
  tenant-scoped and filtered in the repository layer; retrieval re-asserts scope
  on every hit (defense in depth). Cross-tenant reads are tested.
- Deletion via API requires admin *and* routes through an approval flow.

## LLM authority boundary
- The model can only *propose* tools/args/plans. Schema validation, allowlists,
  forbidden lists, risk ceilings, and approval gates are enforced in deterministic
  code. There is no code path where model text executes anything.
- The LLM never sees API keys: the provider holds its own credential internally;
  prompts, tool args, logs and state never contain secrets (log scrubber +
  output guardrail as backstop).

## Risk classification & approvals
- Static per-tool risk (LOW/MEDIUM/HIGH/CRITICAL) + **argument-driven escalation**
  (e.g. destructive SQL patterns ⇒ CRITICAL, `delete` in args ⇒ HIGH).
- Contract risk ceiling: a task may forbid higher-risk actions outright.
- HIGH/CRITICAL ⇒ durable approval request; execution only after explicit human
  decision; rejection fails the step safely; approved actions execute exactly once.

## Prompt-injection defense
- All external content (PDFs, web, MCP outputs, memory) is untrusted data.
- Detection: unicode/homoglyph normalization + weighted pattern families
  (instruction override, role hijack, tool lure, secret lure, exfiltration,
  encoded payloads), density-scored.
- Containment: structural (`wrap_untrusted_block`), plus **sentence-level
  quarantine** — attack sentences are stripped from grounding content while clean
  sentences remain usable; fully-hostile pages are inadmissible.
- The output guardrail blocks answers that echo injection-grade directives.

## Injection-resistance of the pipeline
- Tool inputs: strict JSON Schema (no unknown fields, bounded ranges).
- Tool outputs: schema-validated, size-capped, observation-verified (null, oversize,
  self-flagged unverified, and empty-result semantics handled deterministically).
- MCP: risky *names* escalate risk regardless of server claims; outputs are
  untrusted by default; remote calls are never blind-retried.

## Data integrity & audit
- Append-only audit log with SHA-256 hash chain + `verify_chain()` tamper check.
- Every tool call, approval request/decision, document ingestion, task transition
  is audited with actor, subject and trace id.

## Rate limiting & availability
- Per-key token-bucket rate limiting on all authenticated routes.
- Bounded budgets (iterations/tool calls/LLM calls/tokens/cost/wall clock) with
  safe-stop semantics: the agent returns an explanation, never a partial
  unverified answer.

## Known limitations
- Rate limiting is in-process; the Redis backend is wired by config but the
  distributed limiter is a documented follow-up.
- Dev-key mode must never run in production (`ARA_ENV=production` refuses it by
  convention; configure `ARA_API_KEY_SHA256`).
- No per-request CSRF story (pure token-header API, no cookies by design).
