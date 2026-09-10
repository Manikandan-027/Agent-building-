# Model Selection & Fine-Tuning Policy

## Recommendation (2026, for this architecture)

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
