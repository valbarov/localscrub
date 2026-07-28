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

## Step 5 — the `[ner]` extra: pretrained model vs. fine-tune, answered empirically

**The question** (from the maintainer): would a model pretrained for de-id
beat fine-tuning our LLM? **Answer: for template-corpus categories, yes,
decisively — and it reshapes what the fine-tune is for.**

**What:** `obi/deid_roberta_i2b2` (125M-parameter token classifier trained
on the i2b2 2014 de-id corpus) wrapped as a stage-1 recognizer behind an
optional `[ner]` extra (`pip install 'localscrub[ner]'`, flags `--ner`).
Only PATIENT/STAFF→NAME and LOC→GEO labels are mapped; dates, phones,
emails stay with the regexes, whose boundaries are already exact.

**Headline numbers (50 notes, seed 42, all on the same corpus):**

| config | relaxed F1 | redaction recall | NAME strict F1 | GEO fully redacted | latency/note |
|---|---|---|---|---|---|
| stage 1 | 0.79 | 0.65 | 0.00 | 0.00 | µs, CPU |
| + NER | 0.97 | 0.944 | 0.76 | 0.73 | 185 ms, CPU |
| + LLM | 0.97 | 0.933 | 0.93 | 0.03 | 11.4 s, GPU |
| + NER + LLM | 0.96 | 0.944 | 0.76 | 0.73 | ~8 s, GPU |

**The debugging story (best material in this step — three real bugs and a
gold-standard flaw, all surfaced by the eval harness in one afternoon):**

1. *First run: EMAIL recall fell 1.00 → 0.53 when NER was added.* Adding a
   recognizer must never hurt another's typed recall — the eval caught an
   integration bug immediately. Root cause #1: our span-merge glued model
   spans across a paragraph break (`name@example.org.\n\nNext Name` became
   one NAME), because the gap regex treated `.\n\n` as joinable punctuation.
2. *Second run: still 0.53.* Root cause #2 was upstream: the HF pipeline's
   own aggregation produced a single PATIENT group across that same
   paragraph break. Fix: split every model group at newlines before
   anything else — no identifier spans a line break.
3. *Third run: EMAIL 0.78, closer.* Root cause #3: the model labels
   `darius.ashcombe@example.com` itself as a PATIENT name — because emails
   literally contain patient names. Fix: a name/place span containing `@`
   is dropped at the source (a person name never contains `@`).
4. *GEO recall pinned at 0.52 regardless.* Not a model bug — a **gold
   standard flaw**: the corpus planted street and city as two gold spans
   where reality has one postal address, so the model's single correct
   address span could only ever match half the gold. Fixed in the corpus
   (single `{address}` span). Lesson: eval numbers can indict the gold, not
   the system; 0.52 with precision 0.72 was the tell.

After all four fixes: EMAIL back to 1.00, GEO relaxed recall 0.52 → 1.00,
overall relaxed micro recall 1.00.

**What it means for the roadmap:**
- NER and LLM fail *differently* on the same categories: the LLM copies
  name boundaries nearly perfectly (strict 0.93) but clips addresses
  (3% fully covered); the token classifier covers whole addresses (73%)
  but drags titles/credentials into name spans (strict 0.76).
  Complementary failure modes, one metric pair to prove it.
- With NER in stage 1, the LLM adds *nothing* on this corpus (0.944 →
  0.944) at 43× the latency. Stage 2's justification now rests entirely on
  narrative PHI and adjudication — which templates don't test and
  MTSamples-injection will.
- The LoRA fine-tune's target sharpened: not "GEO in general" but the two
  residual boundary behaviors above.
- Article hooks: "a 125M model beat a 14B model at the thing it was
  trained for — 60× faster, on CPU"; "when your eval indicts your gold
  standard"; "adding a detector must never make another detector worse."

**Costs:** the `[ner]` extra pulls torch + transformers (~2.5 GB installed);
the base install stays pure-Python (ADR 0001 holds). Model download ~500 MB
on first use. All CPU: 184 ms/note, no GPU required.

## Caveats to state in any write-up

- All numbers are on the *template* synthetic corpus — easy prose, labeled
  identifier contexts. The MTSamples-injection benchmark (authentic
  medical-transcription prose) is the next, harder test; expect every
  number above to drop.
- 50-note cascade sample for LLM configs (calls are ~seconds each); rerun at
  500 before publishing numbers. Stage-1 and NER configs are already at 500.
- The corpus was rebuilt in step 5 (single-span addresses), so step-4 tables
  measured on the older two-span gold are not directly comparable to
  step-5 tables; the four-way table in step 5 is the apples-to-apples one.
- Reproducibility of stage 2 is best-effort (fixed seed, temperature 0, but
  tied to model build + hardware).
- No real PHI was used anywhere, at any step, including in the prompts sent
  to the local model — worth stating explicitly in the article.
