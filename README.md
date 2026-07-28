# localscrub

Local-first PHI/PII de-identification for clinical and free-form text.

Your data never leaves your machine: a fast rule/NER pass catches the obvious
identifiers, and a **local LLM** handles the context-dependent remainder — no
cloud APIs, no data crossing your trust boundary.

> **Status:** early development, pre-release. See [ROADMAP.md](ROADMAP.md).

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
```

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
are redacted fail-closed *and* flagged for stage-2 adjudication. Person and
place names require the NER extra / stage 2 — not yet released.

## Entity coverage

All 18 HIPAA Safe Harbor identifier categories (45 CFR 164.514(b)(2)), plus
generic PII that matters in practice: credit cards, passports, national IDs,
and leaked credentials/secrets. See `src/localscrub/entities.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
