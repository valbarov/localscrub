# localscrub

Local-first PHI/PII de-identification for clinical and free-form text.

Your data never leaves your machine: a fast rule/NER pass catches the obvious
identifiers, and a **local LLM** handles the context-dependent remainder — no
cloud APIs, no data crossing your trust boundary.

> **Status:** v0.1 — usable, honest about its limits, pre-1.0 API.
> New here? Start with the **[quickstart](docs/quickstart.md)**. Roadmap: [ROADMAP.md](ROADMAP.md).

## Why

Existing options force a bad trade:

- **Cloud de-id APIs** — accurate, but you send the sensitive data out to get it scrubbed.
- **Local rule/NER tools** — private, but miss context-dependent PHI in free text
  ("the patient's sister works at the bakery on Elm Street").

localscrub runs a two-stage cascade entirely on your hardware: cheap detection for
the 90%+, a small fine-tuned local model for the ambiguous rest.

## Usage

```bash
localscrub scrub notes.txt                      # print redacted text
localscrub scrub notes.txt --json               # print the JSON entity report
localscrub scrub notes.txt -o clean.txt --report report.json
localscrub scrub --report-only < notes.txt      # detect without rewriting
localscrub scrub notes.txt --fail-on-detect     # CI gate: exit 3 if anything found
localscrub scrub notes.txt --llm                # + stage 2: local LLM via Ollama
localscrub scrub notes.txt --ner                # + pretrained clinical NER (CPU)
localscrub scrub notes.txt --ner --ner-backend onnx   # same model, int8 ONNX: ~3x faster

localscrub synth -n 500 --seed 42 -o data/corpora/dev.jsonl   # synthetic eval corpus
localscrub synth -n 500 --seed 42 --diversify -o dev.jsonl    # + local-LLM paraphrase (Ollama)
localscrub mtsamples --fetch -n 100 --seed 42 -o mts.jsonl    # injection benchmark: real prose
localscrub eval data/corpora/dev.jsonl                        # per-entity precision/recall
```

Current accuracy numbers (including what stage 1 honestly cannot catch yet)
live in [docs/results.md](docs/results.md). On our benchmark of authentic
medical-transcription prose, localscrub with the `[ner]` extra fully redacts
**99.9%** of planted identifiers on CPU, vs. **75%** for Presidio, the
standard rules-only baseline — measured by the same harness, reproducible
locally: [docs/benchmark.md](docs/benchmark.md).

Deploying on real clinical text? Read the trust-boundary and residual-risk
discussion first: [docs/threat-model.md](docs/threat-model.md).

Or from Python:

```python
from localscrub import scrub

result = scrub("Reach the patient at test@example.com.")
result.redacted      # 'Reach the patient at [EMAIL_1].'
result.entities      # (Entity(type=EMAIL, start=21, end=37, confidence=0.95, ...),)
result.escalations   # spans stage 1 couldn't settle — stage 2's work queue
```

Stage 1 (rules) currently detects: emails, phone/fax numbers, SSNs, credit
cards (Luhn-validated), IP addresses, URLs, dates and ages over 89, and
labeled MRN / health-plan / account identifiers. Uncertain or ambiguous spans
are redacted fail-closed *and* flagged for stage-2 adjudication. The `[ner]`
extra (`pip install 'localscrub[ner]'`, then `--ner`) adds a pretrained
clinical NER model for person, facility, and employer names and places,
on CPU; `[ner-onnx]` with `--ner-backend onnx` runs the same model as an
int8 ONNX export, ~3× faster. Stage 2 (`--llm`, or
`scrub(text, stage2=LlmExtractor())`) runs a local Ollama model for
narrative identifiers and escalation adjudication; see
[ADR 0002](docs/decisions/0002-stage2-llm-interface.md) for its contract.

## Entity coverage

All 18 HIPAA Safe Harbor identifier categories (45 CFR 164.514(b)(2)), plus
generic PII that matters in practice: credit cards, passports, national IDs,
and leaked credentials/secrets. See `src/localscrub/entities.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
