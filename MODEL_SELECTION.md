# Model Selection & Fine-Tuning Policy

## Is the default model free?

**No.** `gpt-4.1-mini` is pay-as-you-go (~$0.40 per 1M input / $1.60 per 1M output
tokens). A typical ARA task makes 4–6 small calls (~10–20K tokens total), so real
costs are roughly **$0.01–0.03 per task** — a few dollars per month for personal
use, but never zero. For $0, use one of the options below.

## 100% free configuration — WIRED & VERIFIED with Groq (2026-09)

This project ships with the Groq free tier configured and empirically verified
(probed live against this repo's actual prompts):

| Role | Model | Measured behavior |
|---|---|---|
| PLANNER (`OPENAI_MODEL_PLANNER`) | **`openai/gpt-oss-120b`** | strict-JSON plan, correct retrieve→(reason)→final_answer structure, ~1.1s |
| REASONER/ANSWERER/CLAIMER (`OPENAI_MODEL_FAST`) | **`qwen/qwen3.8-27b`** | extractive answers with correct citation ids, ~0.3s, reliable JSON |
| fallbacks | `openai/gpt-oss-20b` (answerer-capable; JSON-mode flaky as planner) | auto-failover on errors/429 |

Rejected in probing: `qwen/qwen3.6-27b` (incomplete plans + immediate 429s),
`groq/compound*` (self-agentic — would bypass ARA's governed tool pipeline),
`gpt-oss-safeguard-20b`/prompt-guard (safety classifiers, not generators),
whisper/orpheus (audio).

Free-tier reality: ~1,000 requests/day on gpt-oss-120b (≈4–6 calls per task ⇒
≈150–250 tasks/day), 30 RPM. On 429 the provider retries with backoff down the
fallback chain, then the task fails safely with a budget-stop style explanation.

**Previous generic recommendation (verified 2026-09):**

Best free pick for THIS project: **Google Gemini 2.5 Flash (AI Studio free tier)** —
ARA sends *large evidence prompts* (1M TPM matters) and needs strict JSON; Gemini
Flash is the strongest free model on both, with ~15 req/min and ~1,000–1,500
req/day (≈200–300 tasks/day), no credit card.

| Provider (free tier) | Model | Free limits | Why / caveat |
|---|---|---|---|
| **Google AI Studio** ← primary | `gemini-2.5-flash` | ~15 RPM · ~1,000–1,500 req/day · 1M TPM | best free JSON/tool quality; **free-tier data may train Google models** — keep sensitive docs off it |
| **Groq** ← fallback | `llama-3.3-70b-versatile` or `gpt-oss-120b` | 30 RPM · ~1,000 req/day · 5–20K TPM | blazing fast; low TPM = split large evidence blocks |
| **OpenRouter** ← last resort | `deepseek-chat:free` etc. | 50 req/day (1,000/day after any $10 topup) | 14+ free models, one key |
| **Ollama (local)** ← unlimited & private | `qwen3:8b` | none — your hardware (≈8–16GB RAM) | truly $0 forever, nothing leaves your machine; slower |

**Privacy note:** on every free API tier your prompts (including retrieved
document content) may be used for provider training. For confidential corpora use
the local Ollama option or a paid endpoint.

Chain them for resilience — on 429/5xx the runtime automatically falls through
(see `OPENAI_FALLBACK_MODELS` below):

```bash
# .env — $0 setup
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
OPENAI_API_KEY=...            # aistudio.google.com -> Get API key
OPENAI_MODEL=gemini-2.5-flash
OPENAI_FALLBACK_MODELS=llama-3.3-70b-versatile,deepseek-chat:free
# (fallbacks only work if the SAME base URL serves them; for cross-provider
#  chains run a local gateway or keep one provider per environment)
```

Local/private alternative ($0 forever):

```bash
ollama serve && ollama pull qwen3:8b
# .env:
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3:8b
```

And remember: with no key at all, ARA runs its built-in deterministic scripted
provider — $0, no network, fully tested (that is what CI uses).

## Paid recommendation (2026, for this architecture)

ARA makes **several small LLM calls per task** (plan → optional reason → answer →
claims), each with strict JSON and citation requirements. The bottleneck is
*format reliability and cost at volume*, not raw intelligence — planning quality
is the only place where a stronger model pays for itself.

| Role | Primary pick | Why | Budget alternative |
|---|---|---|---|
| PLANNER (default `OPENAI_MODEL_PLANNER`) | **`gpt-4.1-mini`** | best JSON/tool reliability-per-dollar (~$0.40/$1.60 per 1M), fine-tunable via API | `gemini-2.5-flash` or `deepseek-v4-flash` |
| REASONER / ANSWERER / CLAIMER (default `OPENAI_MODEL_FAST`) | **`gpt-4.1-mini`** | extraction + citation discipline; cheap at 4–6 calls/task | `mistral-small-3.2`, `llama-4-scout` |
| Heavy planning escalation (optional) | `gpt-4.1` / `claude-sonnet-4.5` class | only if plan-validation failures dominate your eval run | `qwen3-72b` via API |
| Self-hosted / fine-tunable open weights | **`Qwen/Qwen3-8B-Instruct`** (Apache-2.0) | strong tool-use/JSON, LoRA-friendly, tiny serving footprint | Qwen3-14B, Gemma-4-9B |

Defaults ship in `.env.example` as `OPENAI_MODEL=gpt-4.1-mini`. Routing is
**code, not prompting**: the runtime's task tag selects the model
(`ara/agent/llm.py::route_models`) — models cannot influence their own routing.

```bash
# minimal setup
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
# optional split (planner escalation / cheap extraction)
OPENAI_MODEL_PLANNER=gpt-4.1-mini
OPENAI_MODEL_FAST=gpt-4.1-mini
# resilience
OPENAI_FALLBACK_MODELS=gpt-4o-mini,gemini-2.5-flash
```

Any OpenAI-compatible gateway works (OpenRouter, Azure, Bedrock proxies, Groq,
vLLM). The provider retries down the fallback chain on 429/5xx/transport errors.

## Fine-tuning: decision policy

**Not needed for v1.** The system's reliability comes from deterministic
validation/verification — a stronger instruction-following model changes
latency/cost, not the safety envelope. Fine-tune only when ALL of:

1. ≥300 **verified** episodes exported (`scripts/prepare_finetune_data.py` —
   the export refuses low-quality episodes by design),
2. the eval suite shows a measured gap in plan validity or answer formatting
   that prompting iterations did not close,
3. you can gate the switch: fine-tuned model must beat base on the same dataset.

## Two supported paths

| Path | Data | Script | Runs on |
|---|---|---|---|
| API SFT (recommended first) | `data/ft_planner.jsonl` / `data/ft_answerer.jsonl` | `scripts/run_finetune.py` (dry-run default) | anywhere; OpenAI-compatible `/fine_tuning/jobs` |
| Open-weights LoRA | same JSONL | `training/lora/train_lora.py` (refuses CPU) | GPU box → vLLM/TGI → `OPENAI_MODEL_PLANNER` |

Both consume the same train/serve prompt templates (single source of truth:
`build_planner_prompt` / `build_answerer_prompt`), eliminating format skew
between inference and training.
