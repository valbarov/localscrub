"""Stage 2: local-LLM extraction of context-dependent PHI (ADR 0002).

The model is asked for **verbatim snippets** plus a type, never for character
offsets — LLMs cannot count characters reliably, but they copy substrings
well. Every occurrence of a returned snippet is located in the source text
by string search, which also redacts repeated mentions (coreference) from a
single answer. Output is constrained to a JSON schema via Ollama's ``format``
parameter, so a malformed reply is an error, not a parse gamble.

The stage-2 contract is additive and fail-closed:

- stage 2 **adds** entities stage 1 missed; on overlap, the stage-1
  detection wins (rules are better calibrated where rules apply);
- an escalation the model **confirms** (any extraction overlapping it) is
  resolved; one the model stays silent on **remains escalated** — a local
  model's "no" is not strong enough evidence to unredact anything.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from localscrub.analyzer import Escalation
from localscrub.entities import Entity, EntityType, Stage
from localscrub.llm import OllamaClient, OllamaError

#: everything text-representable; FULL_FACE_PHOTO cannot occur in text
STAGE2_TYPES: tuple[EntityType, ...] = tuple(
    t for t in EntityType if t is not EntityType.FULL_FACE_PHOTO
)

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [t.value for t in STAGE2_TYPES]},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
        }
    },
    "required": ["entities"],
}

_PROMPT = """\
You are stage 2 of a PHI/PII de-identification pipeline for clinical and
free-form text. Stage 1 (rules) already caught well-formatted identifiers;
your job is everything context-dependent: person names, addresses and other
geographic subdivisions smaller than a state, unusual identifier formats,
and identifying details buried in narrative.

List every identifier in the document below. For each, return its type and
the EXACT verbatim substring as it appears in the document — copy it
character for character, do not normalize, expand, or trim it. If the same
identifier appears several times, one entry is enough. Return an empty list
if there is nothing.

Types: {types}
{flagged}
Document:
{document}
"""

_FLAGGED = """
Stage 1 flagged these uncertain spans. For each one that is an identifier,
include it in your output with the correct type; if it is not an
identifier, simply leave it out:
{spans}
"""


def extraction_prompt(text: str, escalations: Sequence[Escalation] = ()) -> str:
    """The stage-2 prompt for ``text`` — also the SFT training prompt.

    Fine-tuning trains on exactly what inference sends (including stage-1
    escalation hints), so the tuned model never sees a distribution shift.
    """
    flagged = ""
    if escalations:
        spans = "\n".join(
            f'- "{esc.text}" (chars {esc.start}-{esc.end}, reason: {esc.reason})'
            for esc in escalations
        )
        flagged = _FLAGGED.format(spans=spans)
    types = ", ".join(t.value for t in STAGE2_TYPES)
    return _PROMPT.format(types=types, flagged=flagged, document=text)


def _parse_reply(reply: str) -> list[object]:
    """The reply's ``entities`` list; salvage complete items if truncated.

    ``num_predict`` bounds runaway decoding, so a very long extraction list
    can arrive cut mid-object. Every complete item is still usable — each is
    independently verified against the source text — so parse the longest
    well-formed prefix rather than discarding the note. A reply with no
    parseable prefix is still an error.
    """
    try:
        data = json.loads(reply)
    except ValueError:
        closers = [m.end() for m in re.finditer(r"\}", reply)]
        for end in reversed(closers[-64:]):
            try:
                data = json.loads(reply[:end] + "]}")
            except ValueError:
                continue
            break
        else:
            raise OllamaError(
                f"stage-2 reply is not schema-shaped: {reply[:200]!r}"
            ) from None
    try:
        items = data["entities"]
    except (KeyError, TypeError) as exc:
        raise OllamaError(f"stage-2 reply has no entity list: {reply[:200]!r}") from exc
    if not isinstance(items, list):
        raise OllamaError(f"stage-2 reply has no entity list: {reply[:200]!r}")
    return items


class LlmExtractor:
    """Extracts context-dependent PHI with a local LLM backend."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        confidence: float = 0.85,
        seed: int = 0,
        num_ctx: int = 8192,
    ) -> None:
        self.client = client if client is not None else OllamaClient()
        self.confidence = confidence
        self.seed = seed
        # Ollama defaults to a 4096-token window; long documents (e.g. the
        # MTSamples benchmark) would be silently truncated at that size.
        self.num_ctx = num_ctx
        # Bounds runaway schema-constrained decoding (it can loop until the
        # request timeout). Truncated replies are salvaged item by item.
        self.num_predict = 4096

    def extract(
        self, text: str, escalations: Sequence[Escalation] = ()
    ) -> tuple[Entity, ...]:
        """All LLM-found identifier spans in ``text``, sorted, non-overlapping."""
        reply = self.client.generate(
            extraction_prompt(text, escalations),
            schema=_SCHEMA,
            seed=self.seed,
            temperature=0.0,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        items = _parse_reply(reply)

        candidates: list[Entity] = []
        for item in items:
            entity_type = self._valid_type(item)
            snippet = item.get("text", "") if isinstance(item, dict) else ""
            if entity_type is None or not snippet or snippet not in text:
                continue
            start = 0
            while (start := text.find(snippet, start)) != -1:
                candidates.append(
                    Entity(
                        type=entity_type,
                        start=start,
                        end=start + len(snippet),
                        text=snippet,
                        confidence=self.confidence,
                        stage=Stage.LLM,
                    )
                )
                start += len(snippet)

        # Longer snippet wins overlaps ("Alma Quimby" over "Quimby").
        candidates.sort(key=lambda e: (e.start - e.end, e.start))
        accepted: list[Entity] = []
        for candidate in candidates:
            if not any(candidate.start < a.end and a.start < candidate.end for a in accepted):
                accepted.append(candidate)
        accepted.sort(key=lambda e: e.start)
        return tuple(accepted)

    @staticmethod
    def _valid_type(item: object) -> EntityType | None:
        if not isinstance(item, dict):
            return None
        try:
            entity_type = EntityType(item.get("type"))
        except ValueError:
            return None
        return entity_type if entity_type in STAGE2_TYPES else None
