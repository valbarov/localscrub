"""LoRA fine-tune recipe for stage 2 (ROADMAP M2) — see docs/finetune.md.

Trains a small open-weights model to be the stage-2 extractor, on purely
synthetic data. Reproduce:

    uv run localscrub sft -n 2000 --seed 7 -o data/corpora/sft-2000.jsonl
    uv run --group train python scripts/finetune_lora.py \
        --data data/corpora/sft-2000.jsonl --out models/stage2-qwen3-1.7b-lora

Defaults target a 16 GB consumer GPU: Qwen3-1.7B (Apache-2.0), bf16, plain
LoRA (no quantization), r=16, ~30 minutes. Raw prompt→completion format —
the same string `localscrub.stage2.extraction_prompt` builds at inference,
no chat template — so serving needs no template plumbing either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="SFT JSONL from `localscrub sft`")
    parser.add_argument("--out", type=Path, required=True, help="adapter output directory")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests")
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )
    print(f"{len(dataset)} training examples from {args.data}")

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            max_length=args.max_length,
            completion_only_loss=True,
            logging_steps=10,
            save_strategy="no",
            seed=args.seed,
            report_to=[],
            model_init_kwargs={"dtype": "bfloat16"},
        ),
    )
    trainer.train()
    trainer.save_model(str(args.out))
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
