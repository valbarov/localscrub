# M2 build-and-test notes — what each step did and how it helped

Working notes from building the M2 evaluation stack (2026-07-28), kept as
raw material for the release article. All numbers are reproducible: seeds
are in each command, hardware was a single consumer GPU running Ollama
with `qwen3:14b` (9.3 GB, 100% GPU per `ollama ps`).

## The arc in one paragraph

We could not measure anything until we had ground truth, and we refused to
use real PHI to get it. So: generate synthetic notes where every planted
identifier's exact offsets are known (step 1), make the prose less
template-shaped with a local LLM that provably cannot damage the labels
(step 2), build the scorer (step 3), and only then build the thing being
scored — stage 2 (step 4). Each step existed to make the next one honest.

## Step 1 — synthetic corpus generator (`localscrub synth`)

**What:** Template-rendered clinical notes (7 variants, 6 note types) with
fabricated identifiers planted at recorded offsets; JSONL with exact gold
spans. Deterministic from a seed. All values from reserved ranges (555-01XX
phones, RFC 2606 domains, RFC 5737 IPs, Luhn-valid test-prefix cards);
names are random pairings from invented lists.

**How it helped:**
- Perfect ground truth without a single human annotation hour and without
  any real PHI — the corpus *is* its own label set.
- Determinism means an eval number is a property of the code, not of a
  dataset file; corpora are gitignored and regenerated at will.
- Deliberate hard negatives (ages under 90, bare years, blood pressures
  like 126/78) planted as non-entities let the eval punish false positives,
  not just misses.
- Article hook: "the test set is a function, not a file."

## Step 2 — LLM prose diversification (`synth --diversify`)

**What:** Mask every gold span behind a sentinel token (`[[E3]]`), have the
local model paraphrase the connective prose, re-substitute values, recompute
offsets. A rewrite is rejected (note keeps template prose) if any sentinel
is dropped/duplicated **or** if re-running stage 1 on the rebuilt text finds
identifier-shaped strings outside gold spans — i.e., the model invented PHI.

**Result:** smoke test 3/3 rewrites accepted, spans exact after rebuild,
prose genuinely restructured.

**How it helped:**
- Removes the "detector memorizes the template" objection before anyone
  raises it, while keeping labels exact *by construction* — the model never
  sees an identifier, so it cannot corrupt one.
- The fail-closed validation loop (using our own stage 1 as the tripwire
  for invented identifiers) cost ~10 lines and closes the biggest
  ground-truth-corruption risk.
- Article hook: "how to let an LLM touch your test set without trusting it."

## Step 3 — eval harness (`localscrub eval`)

**What:** Per-entity precision/recall/F1 in two matching modes — relaxed
(type + character overlap) and strict (type + exact boundaries) — plus
**redaction recall**: the fraction of gold spans whose every character is
removed from the output, counting fail-closed escalations, regardless of
assigned type. One-to-one matching so double detections can't inflate
recall. Regression floors (≥0.95 relaxed P/R for structured types) run in
CI on every change.

**First results, stage 1 only (500 notes, seed 42):**
- Structured identifiers (DATE, PHONE, FAX, EMAIL, SSN, MRN, plan/account
  IDs, IP, credit card): 1.00 across the board.
- NAME 0.00, GEO_SUBDIVISION 0.00 (no stage-1 recognizer — by design).
- Overall redaction recall 0.62 — the number stage 2 must move.
- **The strict column caught a real bug on its first run:** URL relaxed
  1.00 but strict 0.00 — `UrlRecognizer` swallows sentence-final periods.
  Overlap-only scoring would never have surfaced it.

**How it helped:**
- Three metrics, three different failure classes: relaxed = "did you find
  it", strict = "are your boundaries calibrated", redaction recall = "is
  the output actually safe". Keeping them separate is what made the URL bug
  and the later GEO problem visible as *different* problems.
- The honest zeros (NAME/GEO) turned "stage 2 is on the roadmap" into a
  quantified gap: 38% of gold spans unprotected without it.
- Article hook: "your eval's first job is to embarrass you precisely."

## Step 4 — stage-2 Ollama backend (`scrub --llm`, ADR 0002)

**What:** `LlmExtractor` asks the local model for **verbatim snippets** +
type under a JSON schema (Ollama `format`), never character offsets — LLMs
copy substrings reliably but cannot count. We locate every occurrence by
string search (coreference for free). Merge is additive and fail-closed:
stage-1 detections win overlaps; escalations the model confirms are
resolved; escalations it is silent on stay escalated and redacted — a
14B model's "no" never unredacts anything.

**Cascade vs stage 1 (50 notes, seed 42, qwen3:14b, temperature 0):**

| metric | stage 1 | cascade |
|---|---|---|
| redaction recall (overall) | 0.62 | **0.89** |
| relaxed micro F1 | 0.76 | **0.95** |
| NAME relaxed recall | 0.00 | **0.98** |
| NAME strict recall | 0.00 | **0.92** |
| GEO relaxed recall | 0.00 | 0.53 |
| GEO spans fully redacted | 0.00 | **0.12** |
| relaxed micro precision | 1.00 | 0.95 |

**How it helped / what we learned:**
- The two-stage thesis holds on first contact: rules stay perfect where
  rules apply (the merge never lets the LLM overrule them), and the LLM
  closes most of the context-dependent gap.
- NAME is nearly solved by an off-the-shelf 14B model with a good contract;
  no fine-tuning needed for 0.98 relaxed recall.
- **GEO is the open problem and it's a boundary problem, not a detection
  problem**: the model finds "Cedar Vale" but clips "4050 Mossbank Blvd"
  and ZIPs, so only 12% of geo spans are *fully* redacted. Partial
  redaction of an address is still a leak — redaction recall is the metric
  that exposes this; relaxed F1 (0.57) makes it look twice as good as it
  is. First target for the LoRA fine-tune.
- Precision cost of stage 2 is real but small (1.00 → 0.95 micro): typing
  confusions (plan ID vs account number) and occasional non-PHI flags.
- Snippet-not-offset was the single highest-leverage design choice: every
  extraction is verifiable against the source text before it becomes an
  entity, so a hallucinated span dies at a `str.find` instead of corrupting
  a redaction.
- Article hook: "ask the model for what it can do (copy), verify what it
  can't (locate), and never let its silence undo a redaction."

## Caveats to state in any write-up

- All numbers are on the *template* synthetic corpus — easy prose, labeled
  identifier contexts. The MTSamples-injection benchmark (authentic
  medical-transcription prose) is the next, harder test; expect every
  number above to drop.
- 50-note cascade sample (LLM calls are ~seconds each); rerun at 500 before
  publishing numbers.
- Reproducibility of stage 2 is best-effort (fixed seed, temperature 0, but
  tied to model build + hardware).
- No real PHI was used anywhere, at any step, including in the prompts sent
  to the local model — worth stating explicitly in the article.
