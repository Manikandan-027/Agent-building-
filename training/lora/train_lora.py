#!/usr/bin/env python3
"""LoRA SFT on ARA's exported chat-format JSONL (GPU REQUIRED — this script is a
no-op without CUDA by design; it refuses to run a degraded CPU training run).

  python training/lora/train_lora.py --dataset data/ft_planner.jsonl \
      --base-model Qwen/Qwen3-8B-Instruct --output data/ara-planner-lora
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--output", default="data/ara-lora-out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not installed — run on a GPU box per training/lora/README.md")
        return 1
    if not torch.cuda.is_available():
        print("REFUSING: no CUDA device. Fine-tuning on CPU would be uselessly slow; "
              "use the OpenAI-compatible API path (scripts/run_finetune.py) instead.")
        return 1

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if args.max_examples and len(rows) >= args.max_examples:
                break
    if not rows:
        print("dataset is empty")
        return 1
    print(f"training on {len(rows)} examples from {args.dataset}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="auto", device_map="auto")

    def to_text(example):
        return {"text": tok.apply_chat_template(example["messages"], tokenize=False,
                                                add_generation_prompt=False)}

    ds = Dataset.from_list(rows).map(to_text)

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        bf16=True,
        dataset_text_field="text",
        report_to=[],
    )
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved LoRA adapter to {args.output}; merge + serve per training/lora/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
