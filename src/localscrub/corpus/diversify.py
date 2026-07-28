"""LLM prose diversification for the synthetic corpus (optional GPU pass).

Template-rendered notes share their phrasing, which lets a detector overfit
to the templates instead of the identifiers. This pass paraphrases the
connective prose with a local Ollama model while keeping ground truth exact
by construction:

1. every gold span is masked behind a sentinel token (``[[E3]]``) before the
   model sees the note — the model never touches an identifier;
2. a rewrite is accepted only if every sentinel survives exactly once and no
   unexpected sentinel appears;
3. the rebuilt text is re-analyzed with the stage-1 recognizers — if any
   detection or escalation falls outside a gold span, the model invented
   identifier-shaped text and the rewrite is rejected;
4. any rejection keeps the original note (fail-closed).

Residual risk: invented NAME/GEO prose cannot be machine-checked (stage 1
has no recognizer for them); the prompt forbids it and rejected sentinels
catch most drift. Reproducibility is best-effort: a fixed seed is passed to
Ollama, which repeats output for the same model build and hardware.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from localscrub.analyzer import Analyzer
from localscrub.corpus.generator import GoldSpan, SyntheticNote

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_HOST = "http://localhost:11434"

_SENTINEL_RE = re.compile(r"\[\[E(\d+)\]\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PROMPT = """\
You are rewriting a fully synthetic clinical note used to test a
de-identification system. Paraphrase the note: vary the sentence structure,
wording, and section phrasing while preserving the clinical meaning.

Hard rules:
- Tokens like [[E0]] stand for protected values. Keep every token EXACTLY as
  written, each appearing exactly once. Never add, drop, renumber, or
  reformat a token.
- Do not invent names, initials, dates, ages over 89, phone numbers,
  addresses, emails, or identifiers of any kind. Outside the tokens, the
  prose must contain no identifying details.
- Output only the rewritten note text: no commentary, no markdown fences.

Note:
{note}
"""


class OllamaError(RuntimeError):
    """The Ollama server could not be reached or returned garbage."""


def _ollama_generate(prompt: str, *, model: str, host: str, seed: int, timeout: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"seed": seed, "temperature": 0.7},
    }
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise OllamaError(f"Ollama request to {host} failed: {exc}") from exc
    reply = data.get("response")
    if not isinstance(reply, str):
        raise OllamaError(f"unexpected Ollama response shape: {list(data)}")
    return reply


def _mask(note: SyntheticNote) -> str:
    """Replace every gold span with its sentinel, right to left."""
    text = note.text
    for i in range(len(note.spans) - 1, -1, -1):
        span = note.spans[i]
        text = f"{text[: span.start]}[[E{i}]]{text[span.end :]}"
    return text


def _clean(reply: str) -> str:
    reply = _THINK_RE.sub("", reply).strip()
    if reply.startswith("```"):
        reply = reply.strip("`").strip()
    return reply


def _rebuild(reply: str, spans: Sequence[GoldSpan]) -> tuple[str, tuple[GoldSpan, ...]] | None:
    """Substitute values back; None if the sentinel set was damaged."""
    found = [int(m.group(1)) for m in _SENTINEL_RE.finditer(reply)]
    if sorted(found) != list(range(len(spans))):
        return None
    parts: list[str] = []
    new_spans: list[GoldSpan] = []
    length = 0
    cursor = 0
    for match in _SENTINEL_RE.finditer(reply):
        literal = reply[cursor : match.start()]
        parts.append(literal)
        length += len(literal)
        span = spans[int(match.group(1))]
        new_spans.append(GoldSpan(span.type, length, length + len(span.text), span.text))
        parts.append(span.text)
        length += len(span.text)
        cursor = match.end()
    parts.append(reply[cursor:])
    new_spans.sort(key=lambda s: s.start)
    return "".join(parts), tuple(new_spans)


def _no_stray_identifiers(text: str, spans: Sequence[GoldSpan], analyzer: Analyzer) -> bool:
    """Every stage-1 detection/escalation must overlap a gold span."""

    def covered(start: int, end: int) -> bool:
        return any(start < s.end and end > s.start for s in spans)

    result = analyzer.analyze(text)
    return all(covered(e.start, e.end) for e in result.entities) and all(
        covered(e.start, e.end) for e in result.escalations
    )


def diversify_corpus(
    notes: Sequence[SyntheticNote],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    seed: int = 0,
    timeout: float = 120.0,
    generate: Callable[[str, int], str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[SyntheticNote], int]:
    """Paraphrase each note's prose; return (notes, fallback count).

    A note falls back to its template prose whenever the rewrite fails
    validation. ``generate(prompt, note_seed)`` may be injected for testing.
    """
    if generate is None:

        def generate(prompt: str, note_seed: int) -> str:
            return _ollama_generate(
                prompt, model=model, host=host, seed=note_seed, timeout=timeout
            )

    analyzer = Analyzer()
    out: list[SyntheticNote] = []
    fallbacks = 0
    for i, note in enumerate(notes):
        prompt = _PROMPT.format(note=_mask(note))
        reply = _clean(generate(prompt, seed + i))
        rebuilt = _rebuild(reply, note.spans)
        if rebuilt is None or not _no_stray_identifiers(rebuilt[0], rebuilt[1], analyzer):
            out.append(note)
            fallbacks += 1
        else:
            text, spans = rebuilt
            out.append(
                SyntheticNote(id=note.id, note_type=note.note_type, text=text, spans=spans)
            )
        if on_progress is not None:
            on_progress(i + 1, len(notes))
    return out, fallbacks
