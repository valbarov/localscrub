# Quickstart

localscrub removes PHI/PII from text without the text ever leaving your
machine. Five minutes from install to a redacted note, in three tiers — each
tier adds accuracy and cost, and you can stop at any of them.

## Tier 0 — rules only (pure Python, microseconds)

```bash
pip install localscrub
```

```bash
$ echo "Pt reachable at (555) 014-2367, MRN 4821937, seen 03/14/2024." | localscrub scrub
Pt reachable at [PHONE_1], MRN [MRN_1], seen [DATE_1].
```

This tier catches structured identifiers — phones, faxes, emails, SSNs,
MRNs, health-plan and account numbers, dates, ages over 89, URLs, IPs,
credit cards — with calibrated confidence, and **fail-closed handling of
what it can't settle**: ambiguous spans are redacted *and* flagged for
stage-2 review rather than silently passed through.

What it does **not** catch: person names and addresses in running prose.
That is a design decision (see [ADR 0001](decisions/0001-stage1-engine.md)),
not an oversight — those need a model, and models are optional extras here.
The honest per-entity numbers for every tier live in [results.md](results.md).

## Tier 1 — + clinical NER (CPU, ~200 ms/note) — recommended

```bash
pip install 'localscrub[ner]'
```

```bash
localscrub scrub note.txt --ner
```

Adds `obi/deid_roberta_i2b2` — a 125M-parameter token classifier trained on
the i2b2 de-identification task — for names and places, on CPU. On our
benchmark of authentic medical-transcription prose this tier fully redacts
**99.9%** of planted identifiers ([benchmark.md](benchmark.md)).

## Tier 2 — + local LLM (GPU, seconds/note)

Requires [Ollama](https://ollama.com) running locally with a model pulled
(default `qwen3:14b`):

```bash
localscrub scrub note.txt --ner --llm
```

Stage 2 reads the note once more, extracts context-dependent identifiers the
rules and NER can miss ("the patient's sister works at the bakery on Elm
Street"), and adjudicates every span stage 1 escalated. Detections merge
additively — a model can never overrule or un-redact a rule hit
([ADR 0002](decisions/0002-stage2-llm-interface.md)).

## Outputs that aren't just redacted text

```bash
localscrub scrub note.txt --json               # entity report: type, span, confidence, stage
localscrub scrub note.txt --report-only        # detect without rewriting
localscrub scrub note.txt --audit audit.jsonl  # append-only audit log (offsets, no text)
localscrub scrub note.txt --fail-on-detect     # CI gate: exit 3 if anything found
```

From Python:

```python
from localscrub import scrub

result = scrub("Reach the patient at test@example.com.")
result.redacted      # 'Reach the patient at [EMAIL_1].'
result.entities      # (Entity(type=EMAIL, start=21, end=37, confidence=0.95, ...),)
result.escalations   # spans stage 1 couldn't settle — stage 2's work queue
```

## Check it against your own threat model

Every accuracy claim above is reproducible on your machine:

```bash
localscrub synth -n 500 --seed 42 -o eval.jsonl   # synthetic notes, exact gold spans
localscrub eval eval.jsonl --ner                  # per-entity precision/recall table
```

Then read [threat-model.md](threat-model.md) for what localscrub defends
against, what it explicitly does not, and why "review and attest" is still
your job.
