# DBMI / n2c2 data access request (draft)

Where to submit: the DBMI Data Portal (portal.dbmi.hms.harvard.edu) →
"n2c2 NLP Research Data Sets" → register an account, request the
**2014 De-identification and Heart Disease Risk Factors Challenge**
corpus (Track 1: de-identification), and sign the Data Use Agreement.
The portal form asks for requester details and a research-purpose
statement; drafts below.

---

## Requester details

- Name: Vadim Albarov
- Email: vadima@wotch.health
- Affiliation: Wotch Health (role/title as appropriate)
- Project: localscrub — open-source, local-first PHI/PII de-identification
  (Apache-2.0), https://github.com/<org>/localscrub  <!-- fill in -->

## Research purpose statement (short form, ~150 words)

> I am the maintainer of localscrub, an open-source (Apache-2.0),
> local-first de-identification system for clinical free text. It uses a
> two-stage cascade: rule-based recognizers for structured identifiers,
> and a locally hosted language model for context-dependent PHI, so that
> protected text never leaves the user's infrastructure.
>
> I request the 2014 De-identification and Heart Disease Risk Factors
> corpus strictly for **evaluation**: measuring per-entity precision and
> recall of the system against the corpus's gold PHI annotations, so its
> accuracy can be reported in terms comparable to the published
> de-identification literature. The corpus will not be used to train or
> fine-tune any model; all training in the project uses synthetic data
> only. Only aggregate metrics (precision/recall/F1 per PHI category)
> will be published. The data will be stored locally, access-restricted,
> never redistributed, and handled in full accordance with the DUA.

## Supporting detail (if the form has a longer description field)

- **Why this corpus**: it is the community-standard benchmark for
  clinical de-identification; without it, our published numbers cannot be
  compared to prior systems. Our current evaluation uses synthetic
  corpora and an injection benchmark on public medical-transcription
  text, which we document as a stand-in.
- **Evaluation-only, enforced in code and policy**: the project's public
  roadmap commits to "evaluation-only when obtainable, never
  redistributed"; the repository's version-control configuration excludes
  the data directories so the corpus cannot enter the codebase; no
  training pipeline reads from those paths.
- **No re-identification**: the data will be used solely to score
  automated redaction against the provided gold labels; no attempt will
  be made to re-identify individuals, link records, or contact anyone.
- **Storage and disposal**: stored on encrypted local machines available
  only to the signatory; deleted upon DUA termination or on request.
- **Publication**: only corpus-level aggregate metrics and error
  categories will appear in the project's documentation or any article;
  no note text, no excerpts, no derived per-record artifacts.

## After approval — repo handling checklist

1. Download official archives to `data/i2b2/` (already gitignored with a
   "must never enter the repo" comment).
2. Write a thin local parser for the XML gold format (keep the
   zero-dependency policy; BigBIO's loader is a reference).
3. Add `localscrub eval` support for the corpus path; report numbers in
   docs/results.md alongside the synthetic and MTSamples tables.
4. Never: commit, redistribute, train, fine-tune, or paste note text into
   prompts sent to any non-local service (stage 2 is local-only anyway).
