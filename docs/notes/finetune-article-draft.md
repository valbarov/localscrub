# I fine-tuned a 1.7B model on data that doesn't exist - a 17-minute recipe

*The full methodology behind localscrub's stage-2 specialist: synthetic
training data with exact labels, training on the serving distribution, and
why the fine-tune's first product is not accuracy - it's parseability.*

---

In [the last article](https://dev.to/vadim_albarov/a-125m-model-beat-a-14b-llm-at-de-identifying-medical-text-40x-faster-on-cpu-201a)
I benchmarked [localscrub](https://github.com/valbarov/localscrub), a
local-first PHI de-identification cascade, and buried a teaser near the end:
a 1.7-billion-parameter model, LoRA-tuned in 17 minutes on a consumer
laptop, went from producing *zero* parseable replies to 0.92 redaction
recall on authentic clinical prose. This article is the recipe - every
command, every hyperparameter, and the three methodology decisions that I
think matter more than any of the numbers.

The de-identification task is the case study, but the recipe generalizes to
any structured-extraction job you want a small local model to do: if you
can generate your training data and you train on the exact prompt your
inference code sends, a specialist you can retrain from scratch in under
half an hour is within reach on one 16 GB GPU.

One constraint shaped everything: **no real patient data anywhere in the
loop.** Not in the training set, not in the eval set, not in a single
prompt sent to any model at any step. By the end you'll see that this
constraint wasn't a handicap - the synthetic pipeline it forced turned out
to be the most valuable asset in the project.

## Why fine-tune at all, when a pretrained classifier already wins?

Fair question, because the last article's headline was an off-the-shelf
125M token classifier hitting 0.999 redaction recall. If that's available,
why train anything?

Because a token classifier cannot take stage 2's seat. localscrub's second
stage has a conversational contract: it receives the note plus *escalation
hints* - ambiguous spans that the rules engine flagged but couldn't resolve
- and returns verbatim snippets with types, as JSON, adjudicating the
escalations along the way. A token classifier tags a fixed label set; ask
it about an identifier type it wasn't trained on and it has no opinion, and
it can adjudicate "is this flagged span actually PHI?" only for the types
it already knows. Only an instruction-following model can hold up the whole
contract.

A 14B generalist (qwen3:14b via Ollama) holds it up at 5-8 seconds per
note. The question worth 17 minutes of GPU time: can a model an eighth that
size be *taught* to?

## The training data costs nothing and leaks nothing

The usual fine-tuning bottleneck is labeled data. Clinical de-id makes it
worse: the gold-standard corpus (i2b2/n2c2 2014) sits behind a data use
agreement, and real notes are radioactive - one mishandled training example
and your privacy tool has a privacy incident.

localscrub sidesteps both because its synthetic corpus generator already
existed for evaluation. `localscrub synth` renders clinical notes from
templates with fabricated identifiers planted at recorded character
offsets - every phone from the reserved 555-01XX block, every domain from
RFC 2606, every credit card Luhn-valid on a test prefix - deterministic
from a seed. The eval harness scores against those exact gold spans.

The insight that unlocks fine-tuning: **a corpus with perfect gold spans is
also a perfect SFT dataset.** One command emits training pairs instead of
eval records:

```bash
localscrub sft -n 2000 --seed 7 -o data/corpora/sft-2000.jsonl
```

Two thousand examples, generated in seconds, with two properties paid
datasets can't match:

- **Zero annotation noise.** The labels aren't human annotations of
  generated text - the generator *planted* the identifiers, so the gold is
  exact by construction. No inter-annotator disagreement, no boundary
  fuzziness, no label budget.
- **Zero exposure.** There is no real PHI to leak because there is no real
  PHI. The training set is as regenerable and disposable as the eval set -
  corpora are gitignored, adapters are gitignored, and everything rebuilds
  from a seed.

Seed discipline matters here: training uses seed 7, evaluation uses seed
42. Disjoint seeds mean no example overlap - but I'll flag the honest
limitation now rather than in the fine print: the template corpus has only
seven note skeletons, so train and eval share *structure* even though they
share no *content*. Every template-corpus "after" number below is
optimistic by construction. The MTSamples benchmark - authentic medical
transcription prose that never entered the fine-tune in any form - is the
honest generalization test, and it's the one I'll ask you to judge the
recipe by.

## The one rule: train on the serving distribution

Here is the decision I'd defend hardest, and the one I see skipped most
often in fine-tuning write-ups: **each training prompt is built by the
actual inference code.**

`localscrub sft` doesn't format notes into some training-time template that
resembles what inference does. For every example, it *runs stage 1 on the
note* - the real rules engine, producing real escalation hints - and then
calls `localscrub.stage2.extraction_prompt`, the same function the serving
path calls, to build the prompt. The completion is the extractor's exact
reply format, and it parses with the same salvage-tolerant parser
inference uses.

The tuned model never sees a prompt shape at inference time that it didn't
see thousands of times in training, escalation hints included. Most
"fine-tuned model underperforms in production" stories I've read trace back
to exactly this seam: the training data was formatted by a script that
*approximated* the serving prompt, and the approximation drifted.
Eliminating the seam costs nothing - reuse the inference code - and it's
free insurance.

Two corollaries of the same principle:

- **Raw prompt→completion, no chat template.** The serving path sends plain
  text, so training does too. This isn't just simplicity - format
  discipline is part of what the before/after numbers measure, and wrapping
  everything in a chat template would train a different behavior than the
  one being served and scored.
- **Completion-only loss.** The model is graded on its reply, not on its
  ability to predict the note back. Standard practice, but it composes with
  the above: the loss covers exactly the tokens the serving path will
  consume.

## The recipe

Hardware: one RTX 5080 Laptop GPU, 16 GB. Model: Qwen3-1.7B-Base
(Apache-2.0). Total wall-clock for training: **17 minutes.**

```bash
# 1. training data: 2000 examples, seed disjoint from eval seeds
uv run localscrub sft -n 2000 --seed 7 -o data/corpora/sft-2000.jsonl

# 2. train (bf16 LoRA, ~17-30 min on 16 GB)
uv run --group train python scripts/finetune_lora.py \
    --data data/corpora/sft-2000.jsonl \
    --out models/stage2-qwen3-1.7b-lora

# 3. the authentic-prose benchmark (public MTSamples CSV, synthetic injections)
uv run localscrub mtsamples --fetch -n 100 --seed 42 \
    -o data/corpora/mtsamples-100.jsonl

# 4. before/after as stage 2, standard eval harness
uv run --group train python scripts/eval_lora.py \
    --template-notes 20 --mts-notes 15 \
    --mtsamples data/corpora/mtsamples-100.jsonl          # before (base) - fewer
                                                          # notes: ~30 s each
uv run --group train python scripts/eval_lora.py \
    --adapter models/stage2-qwen3-1.7b-lora \
    --mtsamples data/corpora/mtsamples-100.jsonl          # after (50 + 30 notes)
```

Hyperparameters, all overridable via flags: LoRA r=16, α=32, dropout 0.05,
all-linear targets; learning rate 1e-4 with cosine schedule and 3% warmup;
2 epochs; effective batch 16 (batch 2 × gradient accumulation 8); max
length 2048; bf16; completion-only loss; seed 7.

Nothing exotic - r=16 all-linear LoRA at lr 1e-4 is close to community
defaults for this model size, and that's deliberate. The recipe's leverage
is in the data (exact labels, serving distribution), not in hyperparameter
heroics. If your first instinct on a weak result is to sweep learning
rates, look at your data pipeline first.

Deployment note: to serve the adapter through Ollama, convert it with
llama.cpp's `convert_lora_to_gguf.py` and reference it from a `Modelfile`
`ADAPTER` line. The evaluation script talks to the model via HF
`transformers` directly, which skips that step for the before/after.

## Reading the before/after: parseability first, accuracy second

The numbers, scored by the same harness as every other configuration in
the project. **Redaction recall** is the safety metric - the fraction of
gold spans whose every character is removed from the output; a half-redacted
address counts as a miss, because a half-redacted address is still a leak.

| config | corpus | unparseable replies | redaction recall |
|---|---|---|---|
| base 1.7B (before) | template, 20 notes | 20/20 | 0.66 (= stage 1 alone) |
| base 1.7B (before) | MTSamples, 15 notes | 15/15 | 0.61 (= stage 1 alone) |
| + LoRA (after) | template, 50 notes | 0/50 | 1.000* |
| + LoRA (after) | MTSamples, 30 notes | 0/30 | **0.924** |

\* optimistic by construction - train and eval share note skeletons, as
conceded above. The MTSamples row is the one to trust.

Look at the "before" rows first, because they're the finding I didn't
expect. The base model didn't merely underperform - it produced unparseable
output on **every single note**, 35 of 35. Rambling, format drift, never
once a reply the parser could consume. Since localscrub's merge is
fail-closed, a useless stage 2 degrades the cascade to exactly the stage-1
baseline - the 0.66 and 0.61 aren't the small model helping a little,
they're the cascade running as if stage 2 weren't there.

After 17 minutes of LoRA: zero unparseable replies across 80 notes.

**The fine-tune's first product is not accuracy - it's parseability.**
That transition alone converts a dead stage 2 into a working one, before
any detection improvements register. Accuracy followed: 0.61 → 0.92
redaction recall on prose the fine-tune never saw. If you're evaluating
whether a small model can replace a big one in a structured pipeline, check
format compliance before you check task skill - in my case it was the
entire difference between "useless" and "viable," and it's the cheapest
thing fine-tuning buys.

## What the remaining gap is made of - and how I know

On MTSamples, the 17-minute specialist scores 0.924. The 14B generalist
scores 0.976 in the same seat on the same notes (0.967 over the full
100-note benchmark); the pretrained NER extra reaches 0.999. So
the specialist does not yet win on unseen prose - and it's worth being
precise about *why*, because the two candidate explanations point at
opposite fixes.

If the gap were **capacity** - 1.7B just isn't enough model - the fix is a
bigger model, and the 17-minute story dies. If the gap is **training-data
diversity** - 2,000 examples from 7 note skeletons teach the task format
perfectly but narrative prose only partly - the fix is better data, and the
whole pipeline stays consumer-hardware-sized.

The evidence leans diversity. The tuned model scores a *perfect* 1.000
in-distribution while dropping to 0.92 out-of-distribution - the signature
of a distribution gap, not of a model too small for the task it was
trained on. It isn't proof: a 1.7B model can ace seven memorized skeletons
and still lack the parameters for open narrative prose. But diversity is
the cheaper hypothesis, and - unlike capacity - this pipeline can test it
without buying anything.

Which makes the fix mechanical rather than speculative, and the pipeline
already contains both halves:

- **Mix in MTSamples-injected notes** (`localscrub mtsamples`) from
  transcripts disjoint with the eval set - authentic narrative rhythm
  around synthetic gold spans.
- **Mix in `--diversify` paraphrases.** This is the corpus generator's
  trick for letting a local LLM rewrite template prose *without ever seeing
  an identifier*: every gold span is masked behind a sentinel token
  (`[[E3]]`), the model paraphrases around the sentinels, values are
  re-substituted and offsets recomputed. A rewrite is rejected if any
  sentinel is dropped or duplicated, or if re-running stage 1 on the
  rebuilt text finds identifier-shaped strings outside the gold spans -
  i.e., the model invented PHI. Labels stay exact by construction; only the
  prose diversifies.

Same generator, same seeds, same 17-minute training loop - just a corpus
with more imagination. That's the deeper point of the whole exercise:
**the synthetic corpus is the asset.** Data, training, and eval all
regenerate from seeds; the model is almost a by-product. When the training
data is a function rather than a file, "collect more data" becomes a code
change.

## Operational potholes (the section I wish every recipe had)

**An untuned model with no token cap looks exactly like a frozen process.**
The first "before" eval appeared to hang. It hadn't - the base model was
rambling toward a 4,096-token ceiling on every note, minutes per note, with
nothing printed. Two fixes, both boring, both essential: a hard token cap
on generation, and per-note progress output. Instrument your "before" run
as carefully as your "after" - it's the row of the table nobody expects to
debug, and the model most likely to misbehave is sitting in it.

**Eval latency is not serving latency.** The ~17-20 s/note these runs
clock is naive `transformers.generate` - no batching, no serving
engine. Serve the adapter through Ollama or vLLM before quoting latency
anywhere it matters.

**Blackwell GPUs need cu13x torch builds**, and uv wouldn't swap the
default CPU torch for the CUDA build without `--reinstall` and an explicit
pin: `uv pip install --reinstall "torch==2.13.0+cu130" --index-url
https://download.pytorch.org/whl/cu130`. The training dependency group
(peft, trl, datasets, accelerate) is deliberately not part of any install
extra - trainers opt in, users never download it.

**Adapters land in a gitignored `models/` directory.** Weights never enter
the repo - and after the last article's gitignore-trailing-comment incident
silently un-ignored a 17 MB dataset, I now test what gitignore actually
ignores rather than trusting what it appears to say.

## Caveats, stated rather than buried

- The template-corpus 1.000 is optimistic by construction (shared note
  skeletons); judge the recipe by the MTSamples 0.924.
- MTSamples injections are synthetic identifiers in authentic prose, not
  naturally occurring PHI; i2b2-style evaluation is pending a data use
  agreement.
- MTSamples is a well-known public corpus, so the *transcripts* may well
  sit in the base model's pretraining data - "unseen" means unseen by the
  fine-tune. The injected identifiers are freshly generated either way.
- 0.924 does not beat the 14B generalist (0.976 on the same notes) or the
  pretrained NER extra (0.999). The claim is not "small model wins" - it's that format
  discipline plus most of the accuracy is purchasable for 17 minutes of
  consumer GPU time and zero real records, with a documented path for the
  rest.
- No real PHI was used anywhere in this work - training, eval, or any
  prompt sent to any model.

## Check my math

The entire loop - corpus, training pairs, fine-tune, before/after eval -
regenerates from the seeds in the commands above on one 16 GB GPU: seconds
for the data, 17 minutes for the training, and roughly 40 more minutes of
naive generation for the two evals - call it an hour end to end. The code is on
[GitHub](https://github.com/valbarov/localscrub), the package is on
[PyPI](https://pypi.org/project/localscrub/), and the recipe lives in
[docs/finetune.md](https://github.com/valbarov/localscrub/blob/main/docs/finetune.md).

If you try the recipe on your own extraction task - de-id or otherwise -
I'd genuinely like to hear where it breaks. The next article sends the
benchmark to the cloud de-id APIs, and that trust-boundary crossing gets a
write-up of its own.
