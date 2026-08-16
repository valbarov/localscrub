# Evaluation results

Last measured 2026-08-16, after the NER spillover/label fixes and the
ADR 0002 containment amendment. Latency = wall clock / notes for the run
as executed, model load included. Against the synthetic corpus. Reproduce with:

```bash
localscrub synth -n 500 --seed 42 -o data/corpora/eval.jsonl
localscrub eval data/corpora/eval.jsonl          # stage 1 only
localscrub eval data/corpora/eval.jsonl --ner    # + pretrained clinical NER (CPU)
localscrub eval data/corpora/eval.jsonl --llm    # + stage-2 local LLM (Ollama)
```

**relaxed** = type match + character overlap (detection). **strict** = type
match + exact boundaries (calibration). **redacted** = fraction of gold spans
whose every character is removed from the output, counting fail-closed
escalations — the safety number.

## Stage 1 (rules only), 500 notes, seed 42

| entity | gold | relaxed P | relaxed R | relaxed F1 | strict P | strict R | strict F1 | redacted |
|---|---|---|---|---|---|---|---|---|
| NAME | 1437 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| GEO_SUBDIVISION | 277 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| DATE | 1202 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PHONE | 500 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| FAX | 79 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| EMAIL | 371 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| SSN | 144 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| MRN | 431 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| HEALTH_PLAN_ID | 142 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ACCOUNT_NUMBER | 69 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| URL | 221 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| IP_ADDRESS | 79 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CREDIT_CARD | 69 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **all (micro)** | 5021 | 1.00 | 0.66 | 0.79 | 0.93 | 0.61 | 0.74 | 0.66 |

Redaction recall **0.66**.

## Stage 1 + `[ner]` extra (obi/deid_roberta_i2b2, CPU), 500 notes, seed 42

43 ms/note on CPU (21.6 s / 500). Rows unchanged from stage 1 elided:

| entity | gold | relaxed P | relaxed R | relaxed F1 | strict P | strict R | strict F1 | redacted |
|---|---|---|---|---|---|---|---|---|
| NAME | 1437 | 0.97 | 1.00 | 0.98 | 0.81 | 0.84 | 0.83 | 0.84 |
| GEO_SUBDIVISION | 277 | 0.87 | 1.00 | 0.93 | 0.61 | 0.70 | 0.65 | 0.70 |
| **all (micro)** | 5021 | 0.98 | 1.00 | 0.99 | 0.88 | 0.89 | 0.89 | 0.94 |

Redaction recall **0.66 → 0.94**. The spillover-recheck fix shows up as
precision: NAME relaxed P 0.87 → 0.97, GEO 0.72 → 0.87 against the
previously published run, with recall unchanged at 1.00.

## All four configurations, 50 notes, seed 42

| config | relaxed micro F1 | strict micro F1 | redaction recall | NAME strict F1 | GEO fully redacted | latency/note |
|---|---|---|---|---|---|---|
| stage 1 | 0.79 | 0.74 | 0.65 | 0.00 | 0.00 | µs (CPU) |
| stage 1 + NER | **0.99** | 0.89 | 0.944 | 0.84 | **0.73** | 190 ms (CPU) |
| stage 1 + LLM | **0.99** | 0.86 | 0.940 | **0.89** | 0.00 | 7.0 s (GPU) |
| stage 1 + NER + LLM | **0.99** | **0.90** | **0.984** | 0.87 | **0.73** | 7.0 s (GPU) |

The NER+LLM redaction jump (0.944 → 0.984 vs. the previously published
run) is the ADR 0002 containment amendment at work: stage-2 spans that
fully contain a stage-1 fragment now supersede it, so NAME reaches 1.00
fully-redacted in both LLM configs. The price is boundary strictness —
the fuller spans ("Dr. Elena Vasquez" over gold "Elena Vasquez") cost the
LLM config NAME strict F1 0.93 → 0.89 while its NAME redaction rises to
1.00. Redacting more at the cost of typing/boundary credit is the trade
the amendment chose.

## MTSamples-injection benchmark (authentic prose)

Synthetic identifiers injected into public medical-transcription prose as
unlabeled narrative sentences (`localscrub mtsamples --fetch -n 100 --seed 42`,
transcripts 500–3500 chars, 4 injections/note plus a demographics header).
Recall here is exact; **precision is only a lower bound** — the public
transcripts contain their own date- and name-like strings (drug dosages,
"Dr. X" placeholders, real dates in prose) that are not in the injected gold,
so a detector that flags them is penalized for being right.

| config | notes | gold | relaxed R | relaxed P† | redaction recall | latency/note |
|---|---|---|---|---|---|---|
| stage 1 | 100 | 697 | 0.62 | 0.94 | 0.62 | µs (CPU) |
| stage 1 + NER | 100 | 697 | **1.00** | 0.89 | **0.999** | 172 ms (CPU) |
| stage 1 + LLM | 30 | 210 | 0.99 | 0.87 | 0.967 | 8.6 s (GPU) |
| stage 1 + NER + LLM | 30 | 210 | **1.00** | 0.87 | **1.000** | 25 s (GPU)‡ |

† lower bound, see above. The 2026-08-16 prompt change makes stage 2 emit
narrative-identifier spans (occupations, facility names, month-only dates)
that are correct but absent from the injected gold, so the LLM configs'
lower-bound precision is further understated than before.
‡ one of the 30 notes failed stage 2 after retry and was scored
stage-1-only; the retry timeouts dominate this run's average.

Per-type highlights (NER config, 100 notes): NAME relaxed recall 1.00
(strict 1.00 — narrative name boundaries are *cleaner* than template
signature blocks); GEO 1.00 relaxed / 0.96 fully redacted; every injected
identifier type ≥ 0.96 fully redacted. The LLM-only config repeats its
address-clipping weakness (GEO fully redacted 0.00 here); the combined
cascade fixes it (1.00) — on this benchmark the two models are
complementary rather than redundant. The LLM config's one "missed" SSN
(relaxed R 0.99) is a typing miss, not a leak: its characters are fully
redacted (SSN fully-redacted 1.00) inside a broader superseding span of a
different type.

## LoRA fine-tune: 1.7B specialist, before/after ([recipe](finetune.md))

> **Stale as of 2026-08-16:** the extraction prompt (which is also the SFT
> training prompt) gained explicit narrative-identifier categories after
> this adapter was trained, so the adapter now runs with a prompt it never
> saw. Regenerate the SFT corpus and retrain before comparing these rows
> against the tables above.

Qwen3-1.7B-Base, LoRA r=16, 2000 synthetic SFT examples (seed 7, disjoint
from eval seed 42), 17 minutes on one 16 GB GPU. Evaluated as stage 2 via
plain HF generation (no schema-constrained decoding — format discipline is
part of what is measured).

| config | corpus | unparseable | relaxed F1 | redaction recall |
|---|---|---|---|---|
| base 1.7B (before) | template 20 | 20/20 | 0.80 | 0.66 (= stage 1 alone) |
| base 1.7B (before) | MTSamples 15 | 15/15 | 0.75 | 0.61 (= stage 1 alone) |
| + LoRA (after) | template 50 | 0/50 | **1.00** | **1.000** |
| + LoRA (after) | MTSamples 30 | 0/30 | 0.87 | **0.924** |

Reading it:

- **Before**: the untuned base model produced unparseable output on every
  single note — it contributes nothing, and the cascade falls back to the
  stage-1 baseline. **After**: zero unparseable replies across 80 notes.
- The perfect template-corpus score is **optimistic by construction** — the
  train and eval corpora share the same 7 note skeletons (seeds differ,
  templates don't). The MTSamples row is the honest one: prose the model
  never saw, 0.61 → 0.92 redaction recall.
- The 1.7B specialist does not yet beat the 14B generalist (0.976) or the
  NER extra (0.999) on unseen prose — 2000 template-only examples teach the
  task format perfectly but narrative generalization only partly. The
  recipe's stated improvement path: mix MTSamples-injected notes (disjoint
  transcripts) and `--diversify` paraphrases into training.
- Latency (~17–20 s/note) is naive `transformers.generate`; serve the
  adapter through Ollama/vLLM for production-representative numbers.

## Reading these numbers honestly

- **The NER extra answers the "pretrained model vs fine-tune" question for
  the template corpus**: a 125M-parameter token classifier trained on the
  i2b2 de-id task gets the same relaxed F1 and *better* full-span address
  coverage than a 14B generative model, ~35× faster, on CPU. Since the
  2026-08-16 prompt and merge changes, adding the LLM on top of NER pays
  even on templates (redaction 0.944 → 0.984), and on the MTSamples
  benchmark the combination is what reaches 1.000 redaction recall — stage
  2 earns its latency exactly where the roadmap predicted: narrative
  identifiers in authentic prose.
- **NER and LLM fail differently on the same categories.** The LLM copies
  name spans completely (NAME fully-redacted 1.00; strict 0.89 only
  because superseding spans absorb titles) but clips addresses
  catastrophically (0% fully redacted); the token classifier covers whole
  addresses (73%) with name boundaries now close behind (NAME strict 0.84
  after the spillover-recheck and middle-initial fixes). A boundary-repair
  pass or fine-tune has a clear target in each.
- **Stage-1 NAME/GEO zeros are by design** (ADR 0001) — they are what the
  extras exist to close. The structured-identifier rows stay at 1.00 in
  every configuration: the additive merge (ADR 0002) never lets a model
  overrule a rule.
- **URL strict = 0.00** in all configs: `UrlRecognizer` swallows
  sentence-final punctuation. Known bug, tracked for a recognizer fix.
- Per-type regression floors are enforced in CI by
  `tests/test_evaluation.py`; a recognizer change that drops a structured
  type below 0.95 relaxed P/R fails the build.
