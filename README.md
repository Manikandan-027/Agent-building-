# ARA — Autonomous Multimodal Research and Decision Agent

Production-grade agentic system: planning, ColPali-family **visual document retrieval**,
MCP tool calling, explicit evidence tracking, claim verification, prompt-injection
defense, policy-gated autonomous execution with **human approvals**, bounded budgets,
full observability, and an automated evaluation suite.

> **Design principle:** the LLM only *proposes*. Deterministic code — validation,
> policy, verification, budgets — *decides*. `NO EVIDENCE → NO FACTUAL CLAIM`.

```
USER REQUEST → PLAN → RETRIEVE EVIDENCE → EXECUTE TOOLS → VERIFY OBSERVATIONS
→ CHECK POLICIES → GENERATE ANSWER → VERIFY ANSWER → RETURN
```

---

## Quickstart

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # 138 tests
python scripts/run_evals.py         # 17-scenario evaluation suite
uvicorn ara.service.app:create_app --factory --host 0.0.0.0 --port 8000
# open http://localhost:8000  (UI; default dev key: ara-dev-key-change-me)
```

No LLM key is required to run: without `OPENAI_BASE_URL`/`OPENAI_API_KEY` the
system uses a **deterministic scripted provider** (extractive, evidence-only,
cannot fabricate). Add a key to `.env` (see `.env.example`) for full LLM planning
and answering through any OpenAI-compatible endpoint.

### Docker (production topology)

```bash
docker compose up -d --build
# backend :8000 · worker ×2 · colpali-service :8100 · postgres · redis · qdrant
```

GPU ColPali (real visual embeddings): see `colpali_service.Dockerfile` and
`requirements-colpali.txt`; the service **fails closed** if the model stack is
absent — it never silently degrades.

---

## What it does

| Capability | Where |
|---|---|
| Plan / execute / re-plan multi-step tasks (bounded) | `ara/agent/runtime.py`, `ara/agent/planner.py` |
| Tool registry + 10-stage governed execution pipeline | `ara/tools/` |
| MCP client (stdio+HTTP) & governed bridge | `ara/mcp/`, `scripts/demo_mcp_server.py` |
| ColPali-style multi-vector visual retrieval (MaxSim) | `ara/retrieval/`, `colpali_service/` |
| In-memory + Qdrant vector stores | `ara/retrieval/vectors.py` |
| Evidence model, claim verification, contradiction handling | `ara/evidence/` |
| Prompt-injection detection & sentence-level containment | `ara/guardrails/` |
| Auth, roles, tenant isolation, risk escalation, approvals | `ara/policy/`, `ara/service/auth.py` |
| Working/conversational/semantic/episodic memory | `ara/memory/` |
| Budgets & loop protection (iterations/tools/LLM calls/tokens/cost/time) | `ara/agent/budgets.py` |
| Retries, timeouts, fallbacks, idempotency | `ara/agent/resilience.py` |
| Structured JSON logs, traces, cost metering, audit hash-chain | `ara/core/`, `ara/db/` |
| REST API + UI | `ara/service/` |
| Evaluation suite (9 categories) | `evals/`, `scripts/run_evals.py` |

## API

```
POST /chat                      sync chat (executes the full loop)
POST /documents                 upload & ingest (PDF/image/text)
GET  /documents · /documents/{id}
POST /tasks                     queue async task (worker executes)
GET  /tasks · /tasks/{id} · /tasks/{id}/status · /tasks/{id}/evidence
GET  /tasks/{id}/trace          execution trace (admin)
POST /tasks/{id}/approve        APPROVED|REJECTED (resumes WAITING_FOR_APPROVAL)
POST /tasks/{id}/cancel
GET  /memory?kind=semantic
GET  /health · /ready
```

Auth: `X-API-Key` header (keys stored as SHA-256 hashes; dev key auto-provisioned
outside production).

## Storage backends (pluggable)

Dev/CI runs entirely on SQLite + in-process vector store + deterministic mock
embeddings. Production activates Postgres, Redis and Qdrant purely via env config
(`ARA_DATABASE_URL`, `ARA_REDIS_URL`, `ARA_QDRANT_URL`) — no code changes.

## Honest limitations

- **Mock ColPali mode is lexical.** It preserves the exact multi-vector API shape
  of ColPali/ColQwen2 (per-page patch vectors, MaxSim) so the real model drops in
  without code changes, but it does not understand pixels, layout, charts or
  scans. Real visual understanding requires `COLPALI_MODE=real` on a GPU host.
  Page *images* are rendered and shipped to the service either way.
- The scripted/heuristic LLM answers **extractively from evidence only**. It is
  deliberately conservative; its plan repertoire is narrow (retrieve→answer,
  arithmetic, percent-change, clock). Real open-ended planning needs a real model.
- Conflict resolution uses authority → version → recency; genuinely tied conflicts
  are dropped from the answer and **explicitly reported**, never silently chosen.
- Unsupported claims are stripped and refusals are issued instead of guesses —
  answers can be *less complete* than an unconstrained LLM. That is the point.
- Measured behavior, including failure modes, is published in
  [`EVALUATION.md`](EVALUATION.md). We do not claim "hallucination-free".

## Model selection & fine-tuning

Default model: **`gpt-4.1-mini`** (paid, ~$0.01–0.03/task; best JSON/tool
reliability per dollar). **$0 and ALREADY WIRED:** Groq free tier with task routing —
`gpt-oss-120b` plans, `qwen3.8-27b` answers (verified live against this repo's
prompts; ~150–250 tasks/day free). Alternatives: Gemini 2.5 Flash free tier,
OpenRouter, or a private local Ollama. See MODEL_SELECTION.md.
The runtime routes by task: the PLANNER can use a stronger model
(`OPENAI_MODEL_PLANNER`) while extractive roles (REASONER/ANSWERER/CLAIMER) use a
cheap fast one (`OPENAI_MODEL_FAST`) — routing is enforced in code.

Fine-tuning is **not required** for v1; when you have ≥300 verified episodes,
mine them with `scripts/prepare_finetune_data.py` (only provably-good runs pass
the quality gate) and train via `scripts/run_finetune.py` (API SFT, dry-run by
default) or the LoRA path in `training/lora/`. Details: [`MODEL_SELECTION.md`](MODEL_SELECTION.md).

## Repository

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layer-by-layer design and control
flow, [`SECURITY.md`](SECURITY.md) for the threat model, and
[`EVALUATION.md`](EVALUATION.md) for measured results.
