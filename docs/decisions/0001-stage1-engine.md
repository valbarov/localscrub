# ADR 0001: Stage-1 engine — in-house recognizers, not Presidio

- **Status:** accepted
- **Date:** 2026-07-28

## Context

Stage 1 of the cascade needs fast, CPU-only detection of structured identifiers
(SSNs, phone numbers, emails, dates, labeled MRNs, …) with confidence scores that
drive the escalation decision to stage 2. The candidates were:

1. **Microsoft Presidio** (`presidio-analyzer`) as the stage-1 engine, extended
   with custom recognizers;
2. **plain spaCy** NER plus our own regex layer;
3. **in-house recognizers** (regex + dictionaries + checksums), with NER as an
   optional extra later.

## Decision

Build the stage-1 recognizer layer in-house. Do not depend on Presidio at
runtime. Add spaCy-based NAME/GEO detection later as an optional `[ner]` extra
so the base install stays pure-Python (Typer is the only runtime dependency).

## Rationale

- **Benchmark integrity.** The project's headline evaluation is
  "localscrub vs. rules-only vs. cloud APIs", with Presidio as the natural
  rules-only baseline. If stage 1 *is* Presidio, that comparison collapses into
  "Presidio vs. Presidio + LLM" and localscrub reads as a Presidio plugin.
  Owning stage 1 keeps the three-way comparison clean; Presidio appears as a
  benchmark baseline (a `bench` dependency group), never as a runtime dependency.
- **Dependency weight.** `presidio-analyzer` hard-depends on spaCy, so even
  "Presidio for the regex parts" drags a large model ecosystem into every
  install. The M1 identifier set is entirely regex + checksum + context words.
- **Escalation semantics.** The cascade lives or dies on calibrated confidence
  bands (see `recognizers/base.py`). Mapping Presidio's per-recognizer scores
  and context enhancers onto our bands would have to be done anyway; owning the
  scoring means the calibration contract is explicit and testable.
- **Licensing is not a discriminator.** Presidio and spaCy are both MIT,
  compatible with Apache-2.0 either way.

## Consequences

- We maintain our own recognizers and their false-positive/negative profiles;
  the eval harness (M2) must measure us against Presidio to prove this was
  worth it.
- NAME and GEO_SUBDIVISION detection is deferred to the `[ner]` extra / stage 2;
  until then, stage 1 does not catch person or place names — documented in the
  README's honesty table when it ships.
- The Luhn/SSA/calendar validation logic is ours to test exhaustively (see
  the recognizer test tables).
