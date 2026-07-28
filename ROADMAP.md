# Roadmap

## Architecture — the cascade

```
text ──> Stage 1: fast & wide (CPU, ~µs/doc)
         regex + dictionaries + small NER
         → catches obvious 90–95%; flags ambiguous spans/docs
     ──> Stage 2: narrow & smart (GPU, only ~10–20% of volume)
         local LLM (fine-tuned small open-weights model)
         → context-dependent PHI, unusual formats, clinical narrative
     ──> Output: redacted text + entity report (type, span, confidence, stage)
         + audit log (what was redacted and why)
```

Model backend is pluggable:

- `ollama` — default: runs entirely on your machine
- `vllm` — production throughput (continuous batching, quantized weights)
- `bedrock` — optional adapter for AWS-native teams (explicitly **not** local;
  see the trust-boundary notes, to be documented)

## Non-goals (v1)

- No claim of HIPAA Safe Harbor certification — localscrub reduces the problem to
  review-and-attest; residual-risk sign-off stays human.
- No cloud-hosted SaaS offering.
- No images/audio/DICOM — text only.
- English-only.
- No training or evaluation on real PHI, ever.

## Milestones

### M0 — skeleton ✅
- [x] Package layout, `pyproject.toml`, CI (ruff + pytest), Apache-2.0 license
- [x] Entity taxonomy: the 18 HIPAA Safe Harbor categories + generic PII set

### M1 — stage 1: rules & NER
- [ ] Presidio vs. plain-spaCy spike → pick the stage-1 engine
- [ ] Regex + dictionary recognizers for structured identifiers (SSN, MRN, phone, email, dates)
- [ ] Ambiguity scoring: what escalates to stage 2 (confidence thresholds + heuristics)
- [ ] CLI: `localscrub scrub file.txt` → redacted text + JSON entity report

### M2 — stage 2: local LLM + evaluation
- [ ] Synthetic corpus generator (fake clinical notes with planted entities, scored ground truth)
- [ ] MTSamples-injection benchmark: synthetic PHI planted at known positions in public
      medical-transcription samples → authentic prose, perfect ground truth, fully reproducible
- [ ] i2b2/n2c2 2014 de-identification corpus for literature-comparable numbers
      (currently unavailable upstream; evaluation-only when obtainable, never redistributed)
- [ ] Ollama backend: structured extraction on escalated spans
- [ ] Eval harness: per-entity precision/recall on every change; results table in repo
- [ ] LoRA fine-tune recipe on synthetic data, with before/after numbers

### M3 — v0.1
- [ ] Quickstart docs, honest accuracy table, threat-model / trust-boundary doc
- [ ] PyPI release
- [ ] Published benchmark: localscrub vs. rules-only vs. cloud APIs
      (accuracy, latency, cost, and where your data goes)

### Later
- [ ] vLLM throughput mode, stream-processing consumer (log pipelines)
- [ ] Additional fine-tuning recipes and faster processing paths
