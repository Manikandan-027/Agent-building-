# Running ARA on your system — step-by-step

Follow these steps in order. Every command is copy-paste ready. Time: ~5 minutes.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| **Python 3.11 or newer** | check with `python --version` (3.12/3.13 work) |
| **pip** | ships with Python; if ancient: `python -m pip install --upgrade pip` |
| **Git** | to clone the repo |
| Nothing else | No GPU, no Docker, no database server needed for local dev — SQLite + in-memory + mock visual retrieval are the defaults. Postgres/Redis/Qdrant are optional (section 7). |

---

## 1. Clone the repo

```bash
git clone https://github.com/Manikandan-027/Agent-building-.git
cd Agent-building-
```

## 2. Create a virtual environment (strongly recommended — avoids version conflicts)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(if PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry — or use `.\.venv\Scripts\activate.bat` from cmd)*

**Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`.

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For running the test suite / linter as well:
```bash
pip install -e ".[dev]"
```

## 4. Configure your LLM key (the ONLY file you must edit)

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and fill the Groq block (free, no credit card — get a key at
<https://console.groq.com> → *API Keys*):

```ini
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...your key...
OPENAI_MODEL_PLANNER=openai/gpt-oss-120b
OPENAI_MODEL_FAST=qwen/qwen3.8-27b
OPENAI_FALLBACK_MODELS=qwen/qwen3.8-27b,openai/gpt-oss-20b
```

> `.env` is git-ignored — your key never leaves your machine via git.
> **No key at all?** Leave those lines empty and the agent runs in deterministic
> scripted mode (great for CI/demo, no live LLM).

Everything else in `.env` can stay as-is. Two worth knowing:
- `ARA_DEV_API_KEY=ara-dev-key-change-me` — the API key clients must send.
- `COLPALI_MODE=mock` — visual retrieval simulated locally (set `real` only if you have a GPU and the ColPali service running).

## 5. Start the server

```bash
uvicorn ara.service.app:create_app --factory --host 0.0.0.0 --port 8000
```
(or `make run` for auto-reload during development)

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
{"event": "llm_provider_openai_compatible", "model": "openai/gpt-oss-120b", ...}
```

## 6. Verify it works

**Health check** — open <http://localhost:8000/health> in a browser, or:
```bash
curl http://localhost:8000/health
```

**Interactive API docs** — open <http://localhost:8000/docs> (try every endpoint from the browser).

**Ask the agent something** (from the repo root):
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: ara-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 1287 * 46?"}'
```

Expected shape of the response:
```json
{ "task_id": "task_...", "answer": "59202", "verification": { "refused": false, ... } }
```

Give it evidence first and ask about it:
```bash
printf 'Acme FY2023 revenue was 50.2 million dollars.' > note.txt
curl -X POST http://localhost:8000/documents -H "X-API-Key: ara-dev-key-change-me" -F file=@note.txt
curl -X POST http://localhost:8000/chat -H "X-API-Key: ara-dev-key-change-me" \
  -H "Content-Type: application/json" -d '{"message": "What was Acme FY2023 revenue?"}'
```

Windows PowerShell equivalent of the last call:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post `
  -Headers @{ "X-API-Key" = "ara-dev-key-change-me" } `
  -ContentType "application/json" `
  -Body '{"message": "What was Acme FY2023 revenue?"}'
```

**No evidence for a factual question?** The agent *refuses* on purpose
(`"refused": true`) — that's the no-hallucination policy, not a bug.

## 7. Run the tests and evals

```bash
python -m pytest tests/ -q          # 158 tests, ~15 s
python scripts/run_evals.py         # scripted eval suite
```

## 8. Optional: production backends & real visual retrieval

Only needed if you want Postgres/Redis/Qdrant instead of the dev defaults:

```bash
docker compose up -d        # starts postgres, redis, qdrant, worker, colpali-service
```
Then in `.env` set `DATABASE_URL=postgresql://...`, `REDIS_URL=redis://localhost:6379/0`,
`QDRANT_URL=http://localhost:6333` and restart uvicorn.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'ara'` | Run uvicorn **from the repo root** (the folder containing `ara/`), with the venv active. |
| `No module named 'fastapi'` | venv not active, or `pip install -r requirements.txt` not run in this venv. Re-do steps 2–3. |
| `python: command not found` (macOS/Linux) | Use `python3`. |
| PowerShell "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then reactivate venv. |
| `Address already in use` / port busy | Use another port: `--port 8001` (and your URLs). |
| `401`/`403` from the API | Send header `X-API-Key: ara-dev-key-change-me` — it must match `ARA_DEV_API_KEY` in `.env`. |
| `429` / rate-limit from Groq | Free tier: ~30 requests/min, ~1,000/day. The agent retries with backoff and falls back automatically; if it still fails, wait a bit or add your own second free provider. |
| Answers say *"I cannot verify this from the available evidence"* | Working as designed — upload documents via `POST /documents` first, or ask about things present in them. |
| `pip install` fails building a package | `python -m pip install --upgrade pip` and retry; all runtime deps ship prebuilt wheels for 3.11–3.13. |
| LLM provider log says `scripted` instead of `openai_compatible` | `OPENAI_BASE_URL` **and** `OPENAI_API_KEY` must both be non-empty in `.env` (unprefixed, or with `ARA_` prefix — both accepted). |

## Endpoint map

| Method & path | Purpose |
|---|---|
| `GET /health`, `GET /ready` | liveness / readiness |
| `POST /chat` | one-shot ask (creates task, runs full pipeline) |
| `POST /tasks` / `GET /tasks...` | async tasks + status, evidence, trace |
| `POST /tasks/{id}/approve` | durable approval for HIGH-risk tool calls |
| `POST /tasks/{id}/cancel` | cancel a running task |
| `POST /documents` / `GET`, `DELETE /documents...` | ingest & manage evidence documents |
| `GET /memory` | inspect agent memory (episodic/semantic) |
| `GET /docs` | Swagger UI for all of the above |
