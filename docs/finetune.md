# LoRA fine-tune recipe: a small specialist for stage 2

Trains a small open-weights model to be the stage-2 extractor, on purely
synthetic data — no real PHI anywhere in the loop, including prompts.
Target hardware: one 16 GB consumer GPU. Results in
[results.md](results.md).

## Why this shape

- **Training data is free and exact.** The corpus generator emits unlimited
  notes with perfect gold spans, so SFT pairs cost nothing and contain no
  annotation noise.
- **Train on the serving distribution.** Each example's prompt is built by
  `localscrub.stage2.extraction_prompt` after actually running stage 1 on
  the note — escalation hints included — so the tuned model sees exactly
  what inference sends. Completions are the extractor's reply format and
  parse with the same salvage-tolerant parser.
- **Raw prompt→completion, no chat template.** Serving needs no template
  plumbing, and the format discipline the model learns is part of what the
  before/after numbers measure.
- **Seeds are disjoint** (train seed 7, eval seed 42), but the template
  corpus has only 7 note skeletons, so template "after" numbers are
  optimistic. The MTSamples benchmark — prose the fine-tune never saw — is
  the honest generalization test. (MTSamples is a well-known public corpus,
  so the *base* model may have seen the transcripts in pretraining;
  "unseen" means unseen by the fine-tune.)

## Recipe

```bash
# 1. training data: 2000 examples, seed disjoint from eval seeds
uv run localscrub sft -n 2000 --seed 7 -o data/corpora/sft-2000.jsonl

# 2. train (Qwen3-1.7B-Base, Apache-2.0; bf16 LoRA r=16, ~30 min on 16 GB)
uv run --group train python scripts/finetune_lora.py \
    --data data/corpora/sft-2000.jsonl \
    --out models/stage2-qwen3-1.7b-lora

# 3. authentic-prose benchmark (public MTSamples CSV, synthetic injections)
uv run localscrub mtsamples --fetch -n 100 --seed 42 \
    -o data/corpora/mtsamples-100.jsonl

# 4. before/after as stage 2, standard harness
uv run --group train python scripts/eval_lora.py \
    --template-notes 20 --mts-notes 15 \
    --mtsamples data/corpora/mtsamples-100.jsonl    # before (base) — fewer notes:
                                                    # the untuned model takes ~30 s each
uv run --group train python scripts/eval_lora.py \
    --adapter models/stage2-qwen3-1.7b-lora \
    --mtsamples data/corpora/mtsamples-100.jsonl    # after (50 template + 30 MTS notes)
```

Hyperparameters (all overridable via flags): LoRA r=16, α=32, dropout 0.05,
all-linear targets; lr 1e-4 cosine, warmup 3%, 2 epochs, effective batch 16
(2 × 8 accumulation), max length 2048, bf16, completion-only loss, seed 7.

Notes:

- The `train` dependency group (peft, trl, datasets, accelerate) is not part
  of any install extra; `uv sync` installs CPU torch — swap in the CUDA
  build for your platform, e.g.
  `uv pip install --reinstall "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130`
  (Blackwell GPUs need the cu13x builds).
- Adapters land in `models/` (gitignored — weights never enter the repo).
- Serving via Ollama: convert the adapter with llama.cpp's
  `convert_lora_to_gguf.py` and reference it from a `Modelfile` `ADAPTER`
  line; the HF-based `scripts/eval_lora.py` client avoids that step for
  evaluation.
