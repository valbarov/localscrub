# Threat model and trust boundary

What localscrub defends against, what it explicitly does not, and where your
data actually goes. The audience is whoever has to sign off on using this on
real clinical text.

## The claim, precisely

**"Your data never leaves your machine"** means: no byte of the input text,
the redacted output, or any derived report crosses a network boundary you
did not set up yourself. It does **not** mean the output is guaranteed free
of PHI — de-identification here is *review-and-attest*: localscrub reduces
the residual-risk surface and hands a human an auditable work queue;
sign-off stays human (a project non-goal in [ROADMAP.md](../ROADMAP.md)).

## Data flows

```
input text ──> stage 1: in-process Python (regex/dictionaries/checksums)
           ──> [ner] extra: in-process PyTorch inference, CPU, local weights
           ──> stage 2 (--llm): HTTP to localhost:11434 (your Ollama server)
           ──> outputs: redacted text, JSON report, JSONL audit log — local files
```

Complete inventory of network egress, none of which carries input text:

| when | what goes out | contains your data? |
|---|---|---|
| `pip install localscrub` | package downloads from PyPI | no |
| first `--ner` run | `obi/deid_roberta_i2b2` weights from Hugging Face | no |
| `ollama pull` (manual) | model weights from the Ollama registry | no |
| `localscrub mtsamples --fetch` | a public benchmark CSV (SHA-256 pinned) | no |

After models are cached, everything runs air-gapped. If `--ollama-host`
points somewhere other than localhost, *you* have moved the trust boundary —
the flag exists for LAN GPU boxes; pointing it at a hosted endpoint
re-creates exactly the cloud-API data flow this project exists to avoid.
The roadmap's optional Bedrock adapter, if it ships, will carry the same
warning in bold: it is explicitly **not** local.

## Fail-closed properties (enforced by code, tested in CI)

- **Uncertainty redacts.** A span in the escalation band is redacted under
  its best-guess type *and* flagged for stage-2/human review
  (`redact_escalated` defaults on). Ambiguous overlaps escalate rather than
  silently picking a winner.
- **Models can add, never subtract** ([ADR 0002](decisions/0002-stage2-llm-interface.md)).
  Stage-2 output merges additively: it cannot remove, shrink, or overrule a
  stage-1 detection, and its *silence* about an escalated span never
  unredacts it.
- **Snippets are verified.** Stage 2 returns verbatim substrings, not
  offsets; anything that doesn't literally occur in the document is dropped.
- **Regression floors.** CI fails if any structured identifier type drops
  below 0.95 precision/recall (`tests/test_evaluation.py`).

## Threats considered

**Missed PHI (the primary risk).** Quantified, not hand-waved: redaction
recall per configuration and corpus is published in
[results.md](results.md) / [benchmark.md](benchmark.md), including what each
tier honestly cannot see (bare stage 1 catches no names or addresses — by
design; use the `[ner]` extra). Candidates below the confidence floor are
dropped *silently* — lower `floor_threshold` if your risk posture demands a
noisier review queue.

**Partial-redaction leaks.** A clipped address ("[GEO_1] Lane, Dayton") can
re-identify. We measure and publish *fully-redacted* rates per type, not
just overlap recall — it is the headline number precisely because partial
credit is worthless here.

**Prompt injection via the note text.** Stage 2 feeds untrusted text to an
LLM, so assume an adversarial document can steer the model completely. The
merge contract caps the blast radius: a fully hijacked model **cannot
unredact anything** — it can only (a) fail to add detections, leaving you
with stage-1 behavior, (b) add spurious spans (over-redaction, a nuisance),
or (c) falsely "confirm" an escalated span, which keeps it redacted but
launders it out of the human-review queue. (c) is the real one: if your
inputs may be adversarial, review escalations *and* stage-2 confirmations.

**Model supply chain.** Both models run locally with no tool access and no
network path, so a malicious model cannot exfiltrate — the exposure is
detection quality, which the eval harness measures against *your* corpus.
Weights come from Hugging Face / the Ollama registry under those
ecosystems' integrity mechanisms; pin revisions if your policy requires it.

**Output-side leaks.** Know what each artifact contains: the redacted text
is the product; the **JSON report contains the original matched text** (it
must, for human review) — treat it with the same controls as the source
document; the **audit log contains offsets, types, and confidences only, no
text** — safe to ship to a SIEM. Placeholders are numbered per type and
carry no content.

**Regression as a threat.** A recognizer "improvement" that silently drops
recall is a privacy incident with a delay. Hence per-type CI floors and the
`--fail-on-detect` exit code for gating pipelines on fixtures that must
stay clean.

## Explicitly out of scope (v1)

- **Certification.** No claim of HIPAA Safe Harbor compliance; localscrub
  produces evidence for *your* expert determination or attestation.
- **Inference and linkage attacks** on what remains: years, rare diagnoses,
  facility patterns. Safe Harbor's residual-risk clause is a human judgment.
- **Obfuscated PHI** (leetspeak, deliberate misspellings) beyond what the
  models generalize to; adversarial *evasion* is not benchmarked yet.
- **Non-text media** (images, audio, DICOM), non-English text.
- **Host compromise.** localscrub inherits the machine's security: memory,
  swap, crash dumps, and shell history are the platform's problem. Run it
  in the same enclave you'd trust with the unredacted text.
- **Training on real PHI** — never done, in any pipeline, including the
  fine-tune recipe (synthetic corpus only).

## Operational checklist

1. Pull models once, then run air-gapped if policy allows.
2. Use `[ner]` at minimum for free-text clinical notes (see the
   [benchmark](benchmark.md) for why rules-only is not enough).
3. Keep JSON reports under source-document controls; audit logs are the
   shareable artifact.
4. Review the escalation queue; on possibly-adversarial input, review
   stage-2 confirmations too.
5. Re-run `localscrub eval` on a corpus that looks like *your* data before
   trusting any number in ours.
