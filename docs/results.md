# Evaluation results

Stage 1 (rules only) against the synthetic corpus. Reproduce with:

```bash
localscrub synth -n 500 --seed 42 -o data/corpora/eval.jsonl
localscrub eval data/corpora/eval.jsonl
```

**relaxed** = type match + character overlap (detection). **strict** = type
match + exact boundaries (calibration). **redacted** = fraction of gold spans
whose every character is removed from the output, counting fail-closed
escalations — the safety number.

## Stage 1, template corpus (500 notes, seed 42)

| entity | gold | relaxed P | relaxed R | relaxed F1 | strict P | strict R | strict F1 | redacted |
|---|---|---|---|---|---|---|---|---|
| NAME | 1437 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| GEO_SUBDIVISION | 554 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
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
| **all (micro)** | 5298 | 1.00 | 0.62 | 0.77 | 0.93 | 0.58 | 0.72 | 0.62 |

Overall: 500 notes, 5298 gold spans, redaction recall **0.62**.

## Reading these numbers honestly

- **NAME and GEO_SUBDIVISION are 0 by design, not by accident.** Stage 1 has
  no recognizer for them (ADR 0001); they are the reason stage 2 and the
  `[ner]` extra exist. They dominate the micro average — that 0.62 overall
  redaction recall *is* the headline gap stage 2 must close.
- **URL strict = 0.00** exposes a real boundary bug: `UrlRecognizer` swallows
  sentence-final punctuation (`…/x.` including the period), so every URL hit
  is overlap-correct but boundary-wrong. Tracked for a recognizer fix.
- The structured-identifier rows (1.00 across the board) say the corpus is
  easy for rules *where rules apply* — identifiers planted in labeled,
  well-formatted contexts. The `--diversify` pass and the MTSamples-injection
  benchmark exist to make the prose (not the identifiers) harder.
- Per-type regression floors are enforced in CI by
  `tests/test_evaluation.py`; a recognizer change that drops a structured
  type below 0.95 relaxed P/R fails the build.
