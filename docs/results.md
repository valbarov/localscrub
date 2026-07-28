# Evaluation results

Against the synthetic corpus. Reproduce with:

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

184 ms/note on CPU. Rows unchanged from stage 1 elided:

| entity | gold | relaxed P | relaxed R | relaxed F1 | strict P | strict R | strict F1 | redacted |
|---|---|---|---|---|---|---|---|---|
| NAME | 1437 | 0.87 | 1.00 | 0.93 | 0.69 | 0.79 | 0.73 | 0.84 |
| GEO_SUBDIVISION | 277 | 0.72 | 1.00 | 0.83 | 0.50 | 0.70 | 0.58 | 0.70 |
| **all (micro)** | 5021 | 0.94 | 1.00 | 0.97 | 0.83 | 0.88 | 0.85 | 0.94 |

Redaction recall **0.66 → 0.94**.

## All four configurations, 50 notes, seed 42

| config | relaxed micro F1 | strict micro F1 | redaction recall | NAME strict F1 | GEO fully redacted | latency/note |
|---|---|---|---|---|---|---|
| stage 1 | 0.79 | 0.74 | 0.65 | 0.00 | 0.00 | µs (CPU) |
| stage 1 + NER | **0.97** | 0.86 | **0.944** | 0.76 | **0.73** | 185 ms (CPU) |
| stage 1 + LLM | **0.97** | 0.86 | 0.933 | **0.93** | 0.03 | 11.4 s (GPU) |
| stage 1 + NER + LLM | 0.96 | 0.86 | **0.944** | 0.76 | **0.73** | ~8 s (GPU) |

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
| stage 1 + NER | 100 | 697 | **1.00** | 0.81 | **0.999** | 652 ms (CPU) |
| stage 1 + LLM | 30 | 210 | **1.00** | 0.89 | 0.976 | 6.0 s (GPU) |
| stage 1 + NER + LLM | 30 | 210 | **1.00** | 0.80 | **1.000** | 6.8 s (GPU) |

† lower bound, see above.

Per-type highlights (NER config, 100 notes): NAME relaxed recall 1.00
(strict 1.00 — narrative name boundaries are *cleaner* than template
signature blocks); GEO 1.00 relaxed / 0.96 fully redacted; every injected
identifier type ≥ 0.96 fully redacted. The LLM-only config repeats its
address-clipping weakness (GEO fully redacted 0.29); the combined cascade
fixes it (1.00) — on this benchmark the two models are complementary rather
than redundant, unlike on the template corpus.

## Reading these numbers honestly

- **The NER extra answers the "pretrained model vs fine-tune" question for
  the template corpus**: a 125M-parameter token classifier trained on the
  i2b2 de-id task gets the same relaxed F1 and *better* full-span address
  coverage than a 14B generative model, 60× faster, on CPU. Adding the LLM
  on top of NER buys nothing on templates (0.944 → 0.944) — but on the
  MTSamples benchmark the combination is what reaches 1.000 redaction
  recall, so stage 2 earns its latency exactly where the roadmap predicted:
  authentic narrative prose.
- **NER and LLM fail differently on the same categories.** The LLM copies
  name boundaries nearly perfectly (NAME strict 0.93) but clips addresses
  catastrophically (3% fully redacted); the token classifier covers whole
  addresses (73%) but drags titles into name spans (NAME strict 0.76). A
  boundary-repair pass or fine-tune has a clear target in each.
- **Stage-1 NAME/GEO zeros are by design** (ADR 0001) — they are what the
  extras exist to close. The structured-identifier rows stay at 1.00 in
  every configuration: the additive merge (ADR 0002) never lets a model
  overrule a rule.
- **URL strict = 0.00** in all configs: `UrlRecognizer` swallows
  sentence-final punctuation. Known bug, tracked for a recognizer fix.
- Per-type regression floors are enforced in CI by
  `tests/test_evaluation.py`; a recognizer change that drops a structured
  type below 0.95 relaxed P/R fails the build.
