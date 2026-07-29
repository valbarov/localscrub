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

## Step 6 — MTSamples-injection benchmark: the test that lets stage 2 earn its keep

**What:** Synthetic identifiers injected into ~5000 public
medical-transcription samples (mtsamples.com via a pinned-checksum mirror;
downloaded, never redistributed). Injections are *unlabeled narrative
sentences* ("The patient resides at 9302 Mossbank Rd, Quarry Bend, NM
61856.") woven between real sentences at deterministic positions, plus a
small demographics header. Same JSONL gold format, same `localscrub eval`.
`localscrub mtsamples --fetch -n 100 --seed 42`. Benchmark parameters
stated, not silent: transcripts 500–3500 chars, 4 injections/note.

**The methodological point worth an article section:** on an injection
benchmark, *recall is exact but precision is only a lower bound* — the
public prose contains its own date-like and name-like strings ("Dr. X"
placeholders, real dates, drug dosages), so a detector flagging them is
penalized for being right. Redaction recall is the number to trust here.

**Results:**

| config | notes | relaxed R | redaction recall | latency/note |
|---|---|---|---|---|
| stage 1 | 100 | 0.62 | 0.62 | µs, CPU |
| + NER | 100 | 1.00 | 0.999 | 652 ms, CPU |
| + LLM | 30 | 1.00 | 0.976 | 6.0 s, GPU |
| + NER + LLM | 30 | 1.00 | **1.000** | 6.8 s, GPU |

**What it means:**
- **The cascade thesis survives contact with real prose.** On templates,
  NER made the LLM redundant; on authentic narrative, the combination is
  what reaches 1.000 — the LLM still clips addresses (GEO fully redacted
  0.29 alone) and NER still covers them, so the additive merge composes
  complementary detectors exactly as designed.
- NAME boundaries are *cleaner* in narrative (NER strict recall 1.00 here
  vs 0.79 on templates) — signature blocks ("Dictated by X, MD"), not
  narrative mentions, are the hard boundary case.
- Real prose stress-tested the engineering, not just the models. Three
  fixes came out of this step: NER chunking for >512-token documents
  (overlapping 1500-char windows), `num_ctx` 8192 for stage 2 (Ollama's
  4096 default would silently truncate long notes), and a `num_predict`
  cap + salvage parser after one uncapped schema-constrained generation
  looped past a 10-minute timeout. Truncated extraction lists are now
  parsed to the last complete item instead of crashing — each item is
  independently verified against the source text, so a lost tail is the
  same failure class as a model miss.
- Costs: NER 652 ms/note on these longer documents (vs 185 ms on
  templates); LLM ~6 s/note once capped.
- Article hooks: "recall is exact, precision is a lower bound — the
  injection-benchmark trade"; "the LLM was redundant until the prose got
  real"; "your benchmark should break your plumbing before it breaks your
  model."

## Step 7 — LoRA fine-tune: the specialist experiment closes M2

**What:** the full recipe (docs/finetune.md): `localscrub sft` emits 2000
prompt→completion pairs — the *exact* inference-time stage-2 prompt (built
by running stage 1 per note, escalation hints included) paired with gold
JSON. Qwen3-1.7B-Base + LoRA r=16, bf16, 17 minutes on the RTX 5080
(16 GB). No real PHI anywhere, including prompts. Train seed 7, eval seed
42, disjoint.

**Before/after (the roadmap's asked-for numbers):**

| | unparseable replies | MTSamples redaction recall |
|---|---|---|
| base 1.7B | 35/35 (every note) | 0.61 (= stage 1 alone) |
| + LoRA | **0/80** | **0.92** |

Template corpus after: 1.000 redaction recall — but train/eval share the
same 7 note skeletons, so that row is optimistic by construction and says
so in results.md.

**What the experiment actually taught:**
- **Format discipline is the cheapest thing to buy with fine-tuning.** The
  base model never once produced parseable output; the tuned one never
  failed. That transition alone converts a useless stage 2 into a working
  one — before any accuracy gains.
- **Task learning ≠ generalization.** 2000 examples from 7 templates gave a
  perfect in-distribution score and a partial out-of-distribution one
  (0.92 vs the 14B generalist's 0.976 and NER's 0.999 on MTSamples). The
  gap is the training-data diversity gap, and the recipe documents the fix:
  mix in MTSamples-injected notes and `--diversify` paraphrases.
- **The synthetic corpus is the asset.** The whole loop — data, training,
  eval — runs from `localscrub synth`/`sft` seeds; the specialist can be
  retrained from scratch in under half an hour on consumer hardware.
- Engineering footnotes for the article: the RTX 5080 (Blackwell) needs
  cu13x torch builds; the "before" eval needed a token cap (an untuned
  model rambles to whatever ceiling it gets — 4096 tokens ≈ minutes/note)
  and per-note progress output to be debuggable at all.
- Article hooks: "the fine-tune's first product is not accuracy, it's
  parseability"; "perfect on templates, 92% on real prose — the difference
  is your training data's imagination, not your model"; "a de-id specialist
  you can retrain in 17 minutes."

## Master numbers table (redaction recall — the safety metric)

Every configuration, both corpora, one place. Template = synthetic notes
(seed 42); MTSamples = injection benchmark on authentic prose. Sample sizes
in parentheses.

| config | template | MTSamples | latency/note | hardware |
|---|---|---|---|---|
| stage 1 (rules) | 0.65 (50) | 0.62 (100) | µs | CPU |
| stage 1 + NER | 0.944 (50) | 0.999 (100) | 185–652 ms | CPU |
| stage 1 + LLM (qwen3:14b) | 0.933 (50) | 0.976 (30) | 6–11 s | GPU |
| stage 1 + NER + LLM | 0.944 (50) | **1.000** (30) | 7–9 s | GPU |
| stage 1 + base 1.7B (LoRA "before") | 0.66 (20) | 0.61 (15) | 28–34 s† | GPU |
| stage 1 + tuned 1.7B (LoRA "after") | 1.000* (50) | 0.924 (30) | 17–20 s† | GPU |

\* optimistic: train/eval share note skeletons. † naive `transformers.generate`,
not production serving. Relaxed/strict P-R-F1 per entity type: docs/results.md.

Other numbers an article might want: 2000 SFT examples, 17-minute training
run (Qwen3-1.7B-Base, LoRA r=16, RTX 5080 Laptop 16 GB); base model
unparseable 35/35 notes → tuned 0/80; corpus = 7 templates, 6 note types,
13 entity types, ~10 gold spans/note; MTSamples = 4999 public transcripts,
3196 in the 500–3500-char band, 697 gold spans per 100 notes.

## Decisions and dead ends (the paths-not-taken log)

- **Presidio as stage 1 — rejected before M2** (ADR 0001): would collapse
  the headline benchmark into "Presidio vs Presidio+LLM"; kept as the
  planned rules-only *baseline* instead, never a dependency.
- **Model offsets — rejected** (ADR 0002): LLMs can't count characters;
  verbatim snippets + string search won. The single highest-leverage design
  decision of the milestone.
- **Rewrite-and-diff de-id — rejected** (ADR 0002): a paraphrasing model
  can silently alter clinical content.
- **spaCy for the `[ner]` extra — superseded**: a de-id-specific token
  classifier (`obi/deid_roberta_i2b2`) was chosen over general-purpose
  spaCy NER because it is trained on the exact task; the plan predates the
  result that made it obvious.
- **Kaggle re-upload of i2b2 ("relation extraction") — rejected on
  licensing**: i2b2/n2c2 corpora are under a Harvard DBMI Data Use
  Agreement prohibiting redistribution; a third-party Kaggle copy conveys
  no rights (and that particular corpus is the 2010 relations task — no
  PHI labels anyway). Legitimate route: DBMI Data Portal DUA for the 2014
  de-id corpus. A privacy tool cannot be sloppy about data provenance.
- **Operational potholes worth a sidebar**: uv wouldn't swap CPU→CUDA torch
  without `--reinstall` and an explicit `+cu130` pin (Blackwell needs
  cu13x); a gitignore *trailing comment* silently unignored the 17 MB
  MTSamples CSV and it reached the git index once (caught at staging);
  uncapped schema-constrained decoding looped past a 10-minute timeout
  (fix: `num_predict` cap + salvage parser); the first "before" eval looked
  frozen because an untuned model rambles to any token ceiling — per-note
  progress output and a token cap made it debuggable.

## Caveats to state in any write-up

- Steps 1–5 numbers are on the *template* synthetic corpus — easy prose,
  labeled identifier contexts. Step 6 (MTSamples) is the authentic-prose
  test; the injected-identifier numbers held up, but real i2b2-style
  evaluation (naturally occurring PHI, not injected) remains out of reach
  until the corpus becomes obtainable.
- LLM configs sampled at 30–50 notes (calls are ~seconds each); rerun at
  the full sizes before publishing numbers. Stage-1 and NER configs are at
  100–500 notes.
- The corpus was rebuilt in step 5 (single-span addresses), so step-4 tables
  measured on the older two-span gold are not directly comparable to
  step-5 tables; the four-way table in step 5 is the apples-to-apples one.
- Reproducibility of stage 2 is best-effort (fixed seed, temperature 0, but
  tied to model build + hardware).
- No real PHI was used anywhere, at any step, including in the prompts sent
  to the local model — worth stating explicitly in the article.
