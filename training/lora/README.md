# LoRA SFT for open-weight models (Qwen3 / Llama / Gemma)

Path for self-hosted fine-tuning of an **open-weight** base model using the SAME
JSONL datasets produced by `scripts/prepare_finetune_data.py`.

Recommended base: **Qwen/Qwen3-8B-Instruct** (Apache-2.0, strong tool-use/JSON,
LoRA-friendly). Alternatives: Qwen3-14B (better, slower), Gemma-4-9B, Llama-4-Scout-17B.

## Requirements (GPU box, NOT the API/CI image)

```bash
pip install "torch>=2.3" "transformers>=4.44" "peft>=0.11" "trl>=0.9" datasets accelerate
export CUDA_VISIBLE_DEVICES=0
```

## Train

```bash
python training/lora/train_lora.py \
  --dataset data/ft_planner.jsonl \
  --base-model Qwen/Qwen3-8B-Instruct \
  --output data/ara-planner-lora
```

## Serve + wire into ARA

Serve an OpenAI-compatible endpoint from the merged model (vLLM/TGI/llama.cpp
server), then set in `.env`:

```
OPENAI_MODEL_PLANNER=<served-model-name>
```

## When to fine-tune (honest guidance)

Do NOT fine-tune to fix behavior that prompting+guardrails already handle — ARA's
reliability comes from deterministic validation, not model obedience. Fine-tune
when you have BOTH of:
1. **>=300 verified episodes** mined by `prepare_finetune_data.py` (quality gate
   built in), and
2. a measured gap in the eval suite (`scripts/run_evals.py`) attributable to
   planning/extraction style — e.g. plan validity failures or citation-format
   misses that persist across prompt iterations.

Gate: the fine-tuned model must beat the base model on the SAME eval dataset
before you switch `OPENAI_MODEL_*`. Keep the base model as fallback via
`OPENAI_FALLBACK_MODELS`.
