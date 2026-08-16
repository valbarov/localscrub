# ADR 0002: Stage-2 LLM interface — verbatim snippets, additive merge

- **Status:** accepted (amended 2026-08-16: containment supersedes)
- **Date:** 2026-07-28

## Context

Stage 2 receives what stage 1 could not settle: escalated spans, plus the
categories rules cannot see at all (NAME, GEO_SUBDIVISION, narrative
identifiers). A local LLM must turn a document into typed spans, and its
output must merge with stage-1 results without weakening the fail-closed
guarantees. The candidate interface designs were:

1. **offset extraction** — the model returns `(type, start, end)` directly;
2. **verbatim-snippet extraction** — the model returns `(type, exact substring)`
   and we locate every occurrence by string search;
3. **rewrite** — the model returns the redacted document itself, and spans are
   recovered by diffing.

## Decision

Use verbatim-snippet extraction (`stage2.py`), constrained to a JSON schema
via Ollama's `format` parameter. Merge additively in `analyzer.apply_stage2`:
stage-1 detections win overlaps; escalations the model confirms are resolved;
escalations the model is silent on remain escalated and redacted. Stage 2
never removes or shrinks anything stage 1 produced.

**Amendment (2026-08-16) — containment supersedes.** A stage-2 span that
strictly contains every stage-1 entity it overlaps replaces those entities
("retired from the fire department" over NER's bare "department"). The
original rule assumed stage-1 boundaries are the better-calibrated ones,
which is true for structured identifiers but inverted when stage 1 caught
only a fragment of a narrative identifier: keeping the fragment redacts
"department" while leaking "retired from the fire", exactly the partial
leak the pipeline exists to prevent. Strict containment keeps the original
rule's spirit — an identical or partially overlapping stage-1 span still
wins — and the merge still only ever redacts more, never less. The
superseded fragments' placeholder types are lost in favor of the stage-2
type; that is the same typing-quality bound the Consequences section
already accepts.

## Rationale

- **LLMs copy; they don't count.** Character offsets from a language model
  are unreliable, and one bad offset corrupts a redaction. Exact substrings
  are natural for a transformer to produce and cheap to verify — a snippet
  either occurs in the document or the item is dropped.
- **Coreference for free.** Locating every occurrence of a returned snippet
  redacts repeated mentions from a single answer, matching how the numbered
  placeholders already preserve coreference.
- **Silence must not unredact.** A local model's failure to mention a flagged
  span is weak evidence of cleanliness; the roadmap's non-goal ("residual-risk
  sign-off stays human") requires that unconfirmed escalations survive to the
  report. Confirmation, by contrast, upgrades the queue: the span is covered
  by an entity and leaves the human's pile.
- **Rewrite diffing was rejected** because a paraphrasing model can silently
  alter non-PHI clinical content, and span recovery from a diff is exactly the
  offset-fragility problem again.

## Consequences

- When stage 1's best-guess type disagrees with a stage-2 confirmation, the
  stage-1 entity (and its placeholder type) wins; typing quality on those
  spans is bounded by stage-1 calibration until a smarter reconciliation is
  worth its complexity.
- One model call per document, whole-document context; long documents will
  need chunking (deferred until an input longer than a model context shows up
  in practice).
- The backend seam is `analyzer.Stage2` (a Protocol) and `llm.OllamaClient`;
  vLLM/Bedrock adapters implement the same `generate` shape later without
  touching the merge.
- Cascade accuracy is measured by `localscrub eval --llm` against the same
  gold corpora as stage 1; numbers live in docs/results.md.
