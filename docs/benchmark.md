# Benchmark: localscrub vs. rules-only vs. cloud

The comparison this project exists to win — and to report honestly where it
doesn't. Everything here runs on one machine, on public or synthetic data,
and is exactly reproducible.

## Systems compared

| system | what it is | where your data goes |
|---|---|---|
| **Presidio** (`presidio-analyzer` 2.2.364, default recognizers, spaCy `en_core_web_lg`, score ≥ 0.35) | the standard open-source rules-only baseline (ADR 0001: a benchmark baseline, never a runtime dependency) | stays local |
| **localscrub stage 1** | in-house regex/dictionary/checksum recognizers, pure Python | stays local |
| **localscrub stage 1 + NER** | + `obi/deid_roberta_i2b2`, a 355M-param (RoBERTa-large) token classifier (`[ner]` extra), CPU | stays local |
| **localscrub stage 1 + NER (onnx)** | the same model as an int8 ONNX export via onnxruntime (`[ner-onnx]` extra, `--ner-backend onnx`), CPU | stays local |
| **localscrub stage 1 + OpenMed 44M** | `--ner-model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`: DeBERTa-v3-small fine-tuned on Nemotron-PII (Apache-2.0), name/place labels only, CPU | stays local |
| **localscrub cascade (+ LLM)** | + stage-2 local LLM (Ollama, qwen3:14b) for narrative PHI and escalation adjudication | stays local (your GPU) |
| **cloud de-id APIs** | commercial PHI-detection endpoints | **leaves your machine** — the text to be scrubbed is sent to the provider |

All systems are scored by the same harness (`localscrub.evaluation.score`):
identical one-to-one span matching, identical redaction-recall definition.
**Redaction recall** — the fraction of gold spans whose every character is
removed from the output — is the headline safety number; a partial detection
that leaves half an address in the text does not count.

## Reproduce

```bash
localscrub synth -n 500 --seed 42 -o data/corpora/eval.jsonl
localscrub mtsamples --fetch -n 100 --seed 42 -o data/corpora/mtsamples-100.jsonl

localscrub eval data/corpora/eval.jsonl                # stage 1
localscrub eval data/corpora/eval.jsonl --ner          # + NER extra
localscrub eval data/corpora/eval.jsonl --ner --llm    # full cascade (GPU)

uv run --group bench python scripts/bench_presidio.py data/corpora/eval.jsonl
uv run --group bench python scripts/bench_presidio.py data/corpora/mtsamples-100.jsonl
```

Presidio type mapping, and what its typed metrics can and cannot be blamed
for, is documented at the top of `scripts/bench_presidio.py` (short version:
unmapped Presidio types still get redaction credit — scoring generosity
always leans toward the baseline, never toward us).

## Template corpus (500 synthetic notes, seed 42)

| system | relaxed F1 | strict F1 | redaction recall | latency/note |
|---|---|---|---|---|
| Presidio (rules-only) | 0.73 | 0.59 | 0.78 | 14 ms (CPU) |
| localscrub stage 1 | 0.79 | 0.74 | 0.66 | µs (CPU) |
| localscrub stage 1 + NER | **0.97** | **0.85** | **0.94** | 184 ms (CPU) |
| localscrub stage 1 + LLM | **0.97** | 0.86 | **0.94** | 8.0 s (GPU) |
| localscrub cascade (NER + LLM) | **0.97** | **0.85** | **0.94** | 17 s (GPU) |

## MTSamples injection benchmark (authentic prose, 100 notes, seed 42)

Precision is a lower bound on this benchmark for every system (the public
transcripts contain their own name- and date-like strings not in the injected
gold) — see [results.md](results.md).

| system | relaxed R | relaxed P† | redaction recall | latency/note |
|---|---|---|---|---|
| Presidio (rules-only) | 0.80 | 0.49 | 0.75 | 46 ms (CPU) |
| localscrub stage 1 | 0.62 | 0.94 | 0.62 | µs (CPU) |
| localscrub stage 1 + NER | **1.00** | 0.81 | **0.999** | 652 ms (CPU) |
| localscrub stage 1 + NER (onnx int8) | **1.00** | 0.89 | **0.999** | ~200 ms (CPU)‡ |
| localscrub stage 1 + OpenMed 44M | **1.00** | **0.93** | **1.000** | ~290 ms (CPU)§ |
| localscrub stage 1 + LLM | 0.99 | 0.82 | 0.967 | 7.2 s (GPU) |
| localscrub cascade (NER + LLM) | **1.00** | 0.76 | **0.999** | 7.4 s (GPU) |

† lower bound.
‡ same model, same per-type table as the torch row (relaxed F1 0.94, strict
F1 0.90 — the 0.89 precision reflects the current NER merge fixes, see
results.md); measured 2026-10-12 on a Core Ultra 9 (AVX-VNNI) laptop against
909 ms for the torch backend in the same session. End-to-end `localscrub
eval` on the 100 notes, model load included: 42 s vs 2 min 04 s; peak RSS
0.75 GB vs 1.6 GB. CPUs without VNNI will see less of the speed-up.
§ torch backend, same session and machine: relaxed F1 0.96, strict F1 0.92,
NAME precision 0.93 (vs 0.81 for `deid_roberta_i2b2`), GEO precision 0.78
(vs 0.88), peak RSS 0.73 GB. The 434M Large sibling scores the same at 6× the
cost. Int8 ONNX of this model drops redaction recall to 0.991 (per-tensor)
or 0.996 (per-channel) for ~90 ms — not worth it for a safety metric, so
`--ner-backend onnx` is only recommended with the default model. Run with
OpenMed's own 55-label de-id instead of inside the cascade, the same model
scores relaxed P 0.62: it also tags ages, genders, section headers and
more, and splits names and addresses into fragments.

## Reading it

- **Presidio beats bare stage 1 on redaction recall** (0.78 vs 0.66; 0.75 vs
  0.62). No surprise: spaCy gives it person/place names, which stage 1
  intentionally defers to the `[ner]` extra and stage 2 (ADR 0001). Rules-only
  localscrub is not the recommended configuration; it exists to be fast,
  calibrated, and honest about what it cannot see.
- **With the NER extra, localscrub wins decisively on both corpora**
  (0.94 vs 0.78 and 0.999 vs 0.75 redaction recall) — and the gap is
  concentrated exactly where a leak hurts:
  - **Clinical identifiers**: Presidio has no MRN or health-plan recognizer —
    it fully redacts 6% of MRNs and 15% of plan IDs on the template corpus
    (2% and 14% on MTSamples), and only by accident (its UK_NHS and date
    patterns happening to overlap). localscrub holds these at 1.00 in every
    configuration.
  - **Addresses**: Presidio never fully covered a single gold address on
    either corpus (GEO fully-redacted 0.00; strict F1 0.00) — spaCy tags
    "Dayton" but drops "412 Birch Lane". Partial address redaction is the
    canonical re-identification leak.
- **Presidio's strong suit is names** (NAME relaxed F1 0.94 template, strict
  0.87 — better boundaries than our NER extra's 0.73). Its structured-type
  rows (EMAIL, SSN, IP, credit card) are 1.00, same as ours.
- **The LLM stage earns its keep only on real prose.** On the template
  corpus every enhanced configuration converges at 0.94 redaction recall —
  NER-only, LLM-only, and the full cascade are indistinguishable there, at
  latencies spanning 184 ms to 17 s — synthetic templates hold no narrative
  PHI the NER misses. On MTSamples the LLM alone reaches 0.967 redaction recall
  (vs 0.62 bare) but still trails NER (0.999); NER+LLM matches NER's
  recall (696/697 gold spans) at lower precision. Stage 2's value is the
  narrative long tail and escalation adjudication, not bulk detection.
- **Stage-2 latency is bimodal and its tail is real.** Typical notes take
  ~5 s on an RTX 5080 Laptop; some notes push schema-constrained decoding
  to the `num_predict` cap (~50 s), and one note in 500 stalled the server
  for minutes before erroring. The eval harness retries a failed note once,
  then scores it without stage 2 and reports it (`llm_failures`; the
  template-500 run had exactly 1). The 17 s/note template average includes
  that tail — the median note is far cheaper.
- **Latency**: Presidio 14–46 ms/note; localscrub+NER 184–652 ms/note.
  Presidio is 4–14× faster than our recommended CPU configuration — if its
  accuracy profile is acceptable for your data, that is a real advantage;
  for clinical text the MRN/address rows above say it is not.
- **Cloud APIs**: not yet measured (requires sending the benchmark corpora to
  each provider and an account with each; planned before the article). The
  qualitative row stands regardless of the numbers: the input text crosses
  your trust boundary — see [threat-model.md](threat-model.md).

## Pending measurements

- [ ] Cloud API legs (accuracy, latency, $/note) — postponed; needs provider
      accounts and an explicit decision to send the (synthetic) corpora out.
- [ ] i2b2/n2c2 2014 corpus for literature-comparable numbers (blocked
      upstream, request drafted).
