"""Before/after evaluation for the stage-2 LoRA recipe — see docs/finetune.md.

Runs a Hugging Face model (optionally with a LoRA adapter) as the stage-2
extractor and scores it with the standard eval harness:

    uv run --group train python scripts/eval_lora.py \
        --adapter models/stage2-qwen3-1.7b-lora \
        --template-notes 50 --mtsamples data/corpora/mtsamples-100.jsonl --mts-notes 30

Omit --adapter for the "before" run (base model). The client mimics
OllamaClient.generate but ignores `schema` — no constrained decoding — so
the tuned model's format discipline is part of what is being measured; the
salvage parser applies either way, exactly as in production.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


class HfLocalClient:
    """Greedy local generation with the OllamaClient.generate signature.

    ``max_new_tokens`` overrides the extractor's ``num_predict``: an untuned
    base model rambles to whatever cap it gets, and the cap is the dominant
    cost of the "before" run.
    """

    def __init__(
        self, model_name: str, adapter: Path | None = None, max_new_tokens: int = 1024
    ) -> None:
        self.max_new_tokens = max_new_tokens
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda"
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()

    def generate(self, prompt, *, schema=None, seed=None, temperature=None,
                 think=False, num_ctx=None, num_predict=None):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=min(num_predict or self.max_new_tokens, self.max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


class TolerantExtractor:
    """Counts unparseable replies as zero-extraction notes instead of dying.

    An untuned base model will sometimes emit non-JSON; for a benchmark that
    is a legitimate (bad) outcome to score, not a crash.
    """

    def __init__(self, inner):
        self.inner = inner
        self.failures = 0
        self.done = 0

    def extract(self, text, escalations=()):
        import sys

        from localscrub.llm import OllamaError

        started = time.perf_counter()
        try:
            return self.inner.extract(text, escalations)
        except OllamaError:
            self.failures += 1
            return ()
        finally:
            self.done += 1
            print(
                f"  note {self.done} done in {time.perf_counter() - started:.0f}s "
                f"({self.failures} unparseable so far)",
                file=sys.stderr,
                flush=True,
            )


def report(name, result, seconds, count):
    print(f"=== {name} — {seconds:.0f}s, {seconds / count:.1f} s/note ===")
    print(result.format_table())
    print(f"redaction recall {result.redaction_recall:.3f}")
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--template-notes", type=int, default=50)
    parser.add_argument("--template-seed", type=int, default=42)
    parser.add_argument("--mtsamples", type=Path, default=None, help="benchmark JSONL")
    parser.add_argument("--mts-notes", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    from localscrub.corpus import generate_corpus, read_jsonl
    from localscrub.evaluation import evaluate
    from localscrub.stage2 import LlmExtractor

    label = f"{args.model}" + (f" + {args.adapter}" if args.adapter else " (base)")
    stage2 = TolerantExtractor(
        LlmExtractor(HfLocalClient(args.model, args.adapter, args.max_new_tokens))
    )

    notes = generate_corpus(args.template_notes, seed=args.template_seed)
    start = time.perf_counter()
    result = evaluate(notes, stage2=stage2)
    report(f"template corpus, stage 1 + {label}", result,
           time.perf_counter() - start, len(notes))
    print(f"unparseable replies: {stage2.failures}/{len(notes)}\n", flush=True)

    if args.mtsamples is not None:
        stage2.failures = 0
        notes = read_jsonl(args.mtsamples)[: args.mts_notes]
        start = time.perf_counter()
        result = evaluate(notes, stage2=stage2)
        report(f"MTSamples, stage 1 + {label}", result,
               time.perf_counter() - start, len(notes))
        print(f"unparseable replies: {stage2.failures}/{len(notes)}", flush=True)


if __name__ == "__main__":
    main()
