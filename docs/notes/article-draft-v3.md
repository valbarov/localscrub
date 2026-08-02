# Your data never leaves the machine - and you can check my math

*Building localscrub, a local-first PHI de-identification cascade, and
benchmarking it honestly against the standard baseline - on one consumer
laptop, with zero real patient data.*

---

De-identifying clinical text today forces a bad trade. Cloud de-id APIs are
accurate, but you send the sensitive data out in order to scrub it - the
text crosses your trust boundary before a single character is redacted.
Local rule-based tools keep the data home, but miss exactly the PHI that
matters most: the context-dependent kind. A regex will catch an SSN every
time; it will never catch "the patient's sister works at the bakery on Elm
Street."

[localscrub](https://github.com/valbarov/localscrub) is my attempt to
refuse the trade. It runs a two-stage cascade entirely on your hardware: a
fast rules-and-NER pass for the well-formatted identifiers - phones, emails,
dates, account numbers - and a local LLM, served by Ollama with no network
egress, for the ambiguous remainder. (How much each stage carries is an
empirical question; the benchmark below answers it rather than assuming.)
It's on PyPI as v0.1 (`pip install localscrub`), and every number in this
article reproduces from seeds on a single RTX 5080 laptop.

This is the story of building it - and more importantly, of *measuring* it,
because a privacy tool with unverifiable accuracy claims is just a liability
with a nice README. Along the way: a test set that is a function rather than
a file, a 125-million-parameter model that beat a 14-billion-parameter one,
an eval harness that indicted its own gold standard, and one cursed note
that killed a three-hour benchmark at 99% complete.

## The ground truth problem: a test set that is a function, not a file

You cannot measure a de-identifier without ground truth, and I refused to
use real PHI to get it. The gold-standard clinical de-id corpus (i2b2/n2c2
2014) sits behind a data use agreement, and scraping a third-party re-upload
would make a privacy project sloppy about data provenance on day one.

So the corpus is generated. `localscrub synth` renders synthetic clinical
notes from templates - seven variants across six note types - with
fabricated identifiers planted at recorded character offsets. Every phone
number is from the reserved 555-01XX block, every domain from RFC 2606,
every IP from RFC 5737, every credit card Luhn-valid on a test prefix. The
output is JSONL with exact gold spans, deterministic from a seed:

```bash
localscrub synth -n 500 --seed 42 -o eval.jsonl
```

That determinism buys something subtle: an eval number becomes a property
of the *code*, not of a dataset file. Corpora are gitignored and
regenerated at will. The test set is a function, not a file.

Template-generated text invites an obvious objection: a detector could
memorize the templates. The answer is `--diversify`, which lets a local LLM
paraphrase the connective prose *without ever seeing an identifier*: every
gold span is masked behind a sentinel token (`[[E3]]`), the model rewrites
around the sentinels, values are re-substituted, offsets recomputed. A
rewrite is rejected if any sentinel is dropped or duplicated - or if
re-running stage 1 on the rebuilt text finds identifier-shaped strings
outside the gold spans, i.e. the model *invented* PHI. The validation loop
cost about ten lines and closes the biggest ground-truth-corruption risk.
That is how you let an LLM touch your test set without trusting it.

## An eval whose first job is to embarrass you precisely

The harness (`localscrub eval`) reports three numbers per entity type, and
keeping them separate turned out to matter more than any single one:

- **relaxed** (type + character overlap): *did you find it?*
- **strict** (type + exact boundaries): *are your boundaries calibrated?*
- **redaction recall** - the fraction of gold spans whose every character
  is removed from the output, by any means, under any label: *is the output
  actually safe?*

Redaction recall is the safety metric, and it is deliberately unforgiving:
a detection that leaves half an address in the text does not count. Partial
redaction of an address is still a leak.

The three-metric split earned its keep on the very first run. Stage 1's URL
recognizer scored relaxed 1.00 and strict **0.00** - it was swallowing
sentence-final periods on every single URL. Overlap-only scoring would never
have surfaced it. The same first run put honest zeros on the board: NAME
0.00, GEO 0.00, because stage 1 has no name recognizer by design. Overall
redaction recall: 0.62. That 0.62 turned "stage 2 is on the roadmap" into a
quantified gap - 38% of gold spans unprotected without it.

## The cascade, and the one design decision that carried the project

Stage 2 asks a local model (qwen3:14b via Ollama) to extract
context-dependent PHI. The contract is the highest-leverage decision in the
codebase: the model returns **verbatim snippets plus a type - never
character offsets**. LLMs cannot count characters, but they copy substrings
reliably. Every returned snippet is located in the source text by string
search, which redacts repeated mentions for free, and a hallucinated span
dies at a `str.find` instead of corrupting a redaction. Ask the model for
what it can do (copy), verify what it can't (locate).

The merge with stage 1 is additive and fail-closed. Stage-1 detections win
overlaps - rules are better calibrated where rules apply. Ambiguous spans
stage 1 flagged are put to the model for adjudication, but only in one
direction: an escalation the model confirms is resolved; one it stays
silent on remains escalated and gets redacted anyway. A 14B model's "no"
never unredacts anything.

First contact, 50 notes: redaction recall 0.62 → 0.89, NAME relaxed recall
0.00 → 0.98. And one open wound: the model found "Cedar Vale" but clipped
"4050 Mossbank Blvd", fully covering only 12% of address spans. Relaxed F1
made GEO look twice as healthy as it was; redaction recall told the truth.

## A 125M model beat the 14B model - 40× faster, on CPU

Before reaching for fine-tuning, I tried the boring thing: an off-the-shelf
de-id-specific token classifier (`obi/deid_roberta_i2b2`, 125M parameters,
trained on the i2b2 2014 corpus) wrapped as an optional stage-1 recognizer
(`pip install 'localscrub[ner]'`). Only its name and location labels are
mapped; dates, phones, and emails stay with the regexes, whose boundaries
are already exact.

On the template corpus it was decisive: redaction recall 0.94 at 184 ms
per note on CPU - matching the 14B model at the categories it was trained
for, roughly forty times faster, no GPU.

Let me concede the framing objection before anyone raises it: a specialist
trained on exactly this task beating a prompted generalist is expected, not
shocking. The finding is the *size of the trade* - equal recall at
one-fortieth the latency, no GPU - and it matters because the default
recipe today is "throw an LLM at it," and for structured text the default
is measurably wrong. Nor was the 14B handicapped: it ran qwen3:14b at
temperature 0 with schema-constrained decoding and the same
escalation-hint prompt that inference uses
([`extraction_prompt`](https://github.com/valbarov/localscrub/blob/main/src/localscrub/stage2.py)
in the repo).

The two models fail *differently*,
and that mattered later: the LLM copies name boundaries nearly perfectly
but clips addresses; the token classifier covers whole addresses but drags
titles and credentials into name spans. Complementary failure modes,
measurable as such.

Integrating it produced the best debugging afternoon of the project - three
real bugs and a gold-standard flaw, all surfaced by the eval:

1. Adding NER dropped EMAIL recall from 1.00 to 0.53. A merge bug glued
   model spans across paragraph breaks (`name@example.org.\n\nNext Name`
   became one NAME span).
2. Still 0.53 after the fix - the *upstream* HuggingFace pipeline
   aggregation produced the same cross-paragraph span. No identifier spans
   a line break; split on newlines first.
3. EMAIL 0.78 - the model labels `darius.ashcombe@example.com` as a
   PATIENT name, because emails literally contain patient names. A person
   name never contains `@`; drop such spans at the source.
4. GEO recall pinned at 0.52 no matter what. Not a model bug: the **gold
   standard was wrong** - the corpus planted street and city as two spans
   where reality has one postal address, so the model's single correct span
   could only ever match half the gold. The tell was 0.52 recall alongside
   0.72 precision. Sometimes the eval indicts the gold, not the system.

Rule of integration, now baked into regression floors: adding a detector
must never make another detector worse.

## The prose gets real: an injection benchmark

Templates were too easy, and by this point provably so. The next test
injects synthetic identifiers into ~5,000 authentic public
medical-transcription samples (mtsamples.com - downloaded with a pinned
checksum, never redistributed). Injections are unlabeled narrative
sentences woven between real sentences at deterministic positions:

```bash
localscrub mtsamples --fetch -n 100 --seed 42 -o mts.jsonl
```

One methodological point worth stating plainly: on an injection benchmark,
**recall is exact but precision is only a lower bound.** The real
transcripts contain their own name-like and date-like strings - "Dr. X"
placeholders, real dates - so a detector flagging them is penalized for
being right. Redaction recall over the injected gold is the number to
trust.

## The results - including the negative one

Every configuration, both corpora, all at full size, scored by the
identical harness: 500 template notes carrying 5,021 gold spans, and 100
MTSamples notes carrying 697 injected gold spans - both from seed 42.
Redaction recall - the safety metric - plus precision on the
authentic-prose corpus, where over-flagging shows:

| config | template | MTSamples | MTS precision† | latency/note | hardware |
|---|---|---|---|---|---|
| Presidio (rules-only baseline) | 0.78 | 0.75 | 0.49 | 14–46 ms | CPU |
| stage 1 (rules) | 0.66 | 0.62 | 0.94 | µs | CPU |
| stage 1 + NER | 0.94 | **0.999** | 0.81 | 184–652 ms | CPU |
| stage 1 + LLM | 0.94 | 0.967 | 0.82 | 7–8 s | GPU |
| stage 1 + NER + LLM | 0.94 | **0.999** | 0.76 | 7–17 s | GPU |

† relaxed precision on the injection benchmark - a lower bound for every
system, per the previous section.

Because three nines invite scrutiny, here are the raw counts behind the
headline number. 0.999 is **696 of 697** injected spans fully redacted
(Wilson 95% CI 0.992–0.9997); one more miss would read 0.997, so treat the
third digit as "one miss in this sample," not a stability claim. And the
one miss deserves naming: in `2034 Harrowgate Rd, Lantern Hill, VT 93695`,
the NER covered the street line ("2034 Harrowgate Rd") and the
state-plus-ZIP ("VT 93695") but dropped the city - "Lantern Hill" leaked
from between two redactions. The
address-clipping failure mode, surviving at the very tail. (The template
0.94, for comparison, is 4,711 of 5,021 - 310 misses; at that sample size
the second digit is doing honest work.)

Two findings, one of them a negative result I think the field under-reports.

**On synthetic templates, the LLM buys nothing.** Stage 1 + NER,
stage 1 + LLM, and the full cascade all converge at 0.94 redaction recall -
and at 0.97 relaxed F1 - at latencies spanning 184 milliseconds to 17
seconds. The residual 6% is corpus-bound, not detector-bound. If your text
is structured and identifier-dense, a good token classifier is all the
model you need, and it runs on CPU.

**On authentic prose, the LLM earns its keep.** Bare rules manage 0.62;
adding the LLM lifts that to 0.967; NER+LLM reaches 0.999. The two
detectors compose exactly as the merge was designed to: the LLM still
clips addresses, NER still covers them. But recall is not free - the
precision column tells the other half. The cascade over-flags on narrative
text (0.81 → 0.76 versus NER alone, both lower bounds), and over-redaction
has a real cost in clinical text: every falsely scrubbed token is signal a
downstream reader loses. That trade is why localscrub's model is
review-and-attest rather than fire-and-forget - and note the baseline pays
the same toll, with Presidio at 0.49 precision on this corpus. Recall is
what the LLM buys; know which side of the trade your application needs.

### Versus Presidio, fairly

Microsoft's Presidio (rules + spaCy, the standard open-source baseline)
**beats bare stage 1** on redaction recall - 0.78 vs 0.66 - because spaCy
gives it person and place names, which stage 1 intentionally defers. It is
also 13–14× faster than our recommended CPU configuration, and its NAME
boundaries are *better* than our NER extra's (strict F1 0.87 vs 0.73).
Every scoring ambiguity was resolved in the baseline's favor - its unmapped
types still earn redaction credit.

With the NER extra, localscrub wins where a leak hurts most. Presidio never
fully covered a single gold address on either corpus - spaCy tags "Dayton"
but drops "412 Birch Lane" - and it has no MRN or health-plan recognizer,
fully redacting 2–6% of MRNs and ≤15% of plan IDs. localscrub holds those
at 1.00 in every configuration.

## The 17-minute specialist

A fair question at this point: if a pretrained 125M classifier already hits
0.999, why train anything? Because a token classifier cannot take stage 2's
seat. It tags a fixed label set - ask it about an identifier type it wasn't
trained on and it has no opinion - and it cannot adjudicate the ambiguous
spans stage 1 escalates. Stage 2's contract, verbatim snippets plus types
as JSON, is a conversation, and only an instruction-following model can
hold up its end. The 14B holds it up at seven seconds a note. The question
worth 17 minutes of GPU time is whether a small model can be *taught* to.

`localscrub sft` emits 2,000
training pairs - the exact inference-time stage-2 prompt, escalation hints
included, paired with gold JSON - and a LoRA recipe (r=16, bf16) tunes
Qwen3-1.7B-Base in 17 minutes on the laptop.

The before/after is stark, but not where I expected. The base 1.7B model
produced unparseable output on **35 of 35** notes; the tuned one failed on
0 of 80. The fine-tune's first product is not accuracy - it's
*parseability*. Accuracy followed: MTSamples redaction recall 0.61 → 0.92.
(Template recall hit a perfect 1.000, which is optimistic by construction -
train and eval share note skeletons; the docs say so.)

The 0.92-vs-0.999 gap against the big-model cascade is a training-data
diversity gap, not a capacity verdict - 2,000 examples from 7 templates
generalize only partway to real prose, and the recipe documents the fix
(mix in MTSamples-injected and diversified notes). The deeper point: the
synthetic corpus is the asset. Data, training, and eval all regenerate from
seeds; the specialist retrains from scratch in under half an hour on
consumer hardware, with no real PHI anywhere in the loop - including the
prompts.

## Operational lessons (the part I wish someone had written for me)

**One bad note must not cost you the run.** Stage-2 latency is bimodal:
~5 s per note typically, ~50 s when schema-constrained decoding runs away
to the 4,096-token cap - and in each 500-note pass, exactly one template
note (deterministic at temperature 0) stalled the Ollama server for
minutes before returning HTTP 500. The first time, that single note killed
a 2-hour-50-minute benchmark at note ~495 with nothing written, because the
eval had no per-note error handling. The fix - retry the note once, then
score it without stage 2 and report an `llm_failures` count - turned the
second occurrence into a 4.8-second retry-and-continue. If you benchmark
local LLMs, build this in before your first long run, not after.

**Cap your decoders.** Uncapped schema-constrained generation once looped
past a ten-minute timeout. A `num_predict` cap plus a salvage parser (parse
the longest well-formed prefix of a truncated extraction list; every item
is independently verified against the source anyway) converts runaway
decoding from a crash into a bounded cost.

**Assorted potholes:** a gitignore *trailing comment* silently unignored a
17 MB dataset and it reached the git index once; Blackwell GPUs need cu13x
torch builds and uv needed `--reinstall` with an explicit `+cu130` pin to
swap them; an untuned base model with no token cap looks exactly like a
frozen process.

## Caveats, stated rather than buried

- The template-corpus numbers are on easy, identifier-dense synthetic
  prose; MTSamples-injection is the authentic-prose test, and the numbers
  held there. But injected identifiers are not naturally occurring PHI:
  real i2b2-style evaluation remains the gap.
- On data leakage: MTSamples is a well-known public corpus, so the
  *transcripts* may well sit in a model's pretraining data
  (`obi/deid_roberta_i2b2` was trained on i2b2 2014, not MTSamples; for
  the LLM's pretraining the honest answer is unknown). The scored
  identifiers, however, are not MTSamples content: every gold span is
  synthetic, seeded, and injected - no pretraining corpus contains the
  answers. What leakage *could* do is make the surrounding prose feel
  familiar, flattering the numbers relative to truly unseen clinical
  text - one more reason the i2b2 evaluation (behind a data use
  agreement, unlikely to be memorized) is the next benchmark.
- Injection-benchmark precision is a lower bound for every system measured.
- Stage-2 reproducibility is best-effort: fixed seed, temperature 0, but
  tied to model build and hardware.
- No real PHI was used anywhere, at any step - including in every prompt
  sent to the local model.
- localscrub does not claim HIPAA Safe Harbor certification. It reduces the
  problem to review-and-attest; residual-risk sign-off stays human. The
  [threat model](https://github.com/valbarov/localscrub/blob/main/docs/threat-model.md)
  spells out the trust boundary, including what a hijacked stage-2 model
  can and cannot do (it can over-redact or falsely confirm; it can never
  unredact).

## What's next

Two measurements are deliberately absent, and they are the next article.
**Cloud de-id APIs**: the accuracy/latency/cost legs require sending the
benchmark corpora to each provider - synthetic or not, that crossing of the
trust boundary deserves its own explicit decision and write-up, because it
is the exact trade this project exists to interrogate. And **i2b2/n2c2
2014**: the literature-comparable corpus of naturally occurring PHI, access
request filed and pending. When both land, the comparison table gets its
last rows.

Until then: the code is on
[GitHub](https://github.com/valbarov/localscrub), the package is on
[PyPI](https://pypi.org/project/localscrub/), and every number above
regenerates from a seed. Check my math.
