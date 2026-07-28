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

## Planned interface

```bash
localscrub scrub notes.txt            # redacted text + JSON entity report
localscrub scrub --report-only logs/  # detect without rewriting
```

## Entity coverage

All 18 HIPAA Safe Harbor identifier categories (45 CFR 164.514(b)(2)), plus
generic PII that matters in practice: credit cards, passports, national IDs,
and leaked credentials/secrets. See `src/localscrub/entities.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
