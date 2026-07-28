"""Pretrained clinical NER recognizer — the optional ``[ner]`` extra.

Backed by a token-classification model fine-tuned for de-identification
(default: ``obi/deid_roberta_i2b2``, trained on the i2b2 2014 corpus), so
stage 1 gains the categories rules cannot see. CPU inference, milliseconds
per note. Install with ``pip install localscrub[ner]``.

Only labels the regex layer cannot cover are mapped — person names and
sub-state geography. Dates, phones, emails, and IDs stay with the regex
recognizers, whose boundaries and calibration are already exact; emitting
them here would only add boundary noise.

Adjacent same-type spans separated by short punctuation are merged
("4050 Mossbank Blvd" + "Cedar Vale" + "OH" + "44120" becomes one span):
a partially redacted address is still a leak, so span *completeness* is
the point of this recognizer.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Protocol

from localscrub.entities import Entity, EntityType, Stage

DEFAULT_NER_MODEL = "obi/deid_roberta_i2b2"

#: model label -> our taxonomy; unlisted labels are ignored on purpose
_LABEL_MAP: dict[str, EntityType] = {
    "PATIENT": EntityType.NAME,
    "STAFF": EntityType.NAME,
    "LOC": EntityType.GEO_SUBDIVISION,
}

#: what may separate two spans that are really one identifier: spaces, commas,
#: hyphens — never newlines or periods, which end the identifier's context
_JOINABLE = re.compile(r"^[ ,\-–]{1,4}$")

#: sentence punctuation the model drags into a span is never the identifier's
_TRAILING_JUNK = re.compile(r"[\s.,;:]+$")


class TokenClassifier(Protocol):
    """The slice of a transformers token-classification pipeline we use."""

    def __call__(self, text: str) -> list[dict[str, Any]]: ...


def _load_pipeline(model: str) -> TokenClassifier:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the [ner] extra is not installed; run: pip install 'localscrub[ner]'"
        ) from exc
    return pipeline("token-classification", model=model, aggregation_strategy="first")


class NerRecognizer:
    """NAME/GEO detection with a pretrained de-identification model.

    ``confidence_cap`` bounds the model score before it enters the
    calibration bands: scores below the accept threshold escalate
    fail-closed like any other recognizer's.
    """

    name: ClassVar[str] = "ner"

    def __init__(
        self,
        model: str = DEFAULT_NER_MODEL,
        *,
        pipeline: TokenClassifier | None = None,
        confidence_cap: float = 0.95,
    ) -> None:
        self._pipeline = pipeline if pipeline is not None else _load_pipeline(model)
        self.confidence_cap = confidence_cap

    def analyze(self, text: str) -> list[Entity]:
        raw: list[tuple[EntityType, int, int, float]] = []
        for item in self._pipeline(text):
            entity_type = _LABEL_MAP.get(item.get("entity_group", ""))
            if entity_type is None:
                continue
            start, end = int(item["start"]), int(item["end"])
            score = float(item["score"])
            # The pipeline can aggregate one label run across a paragraph
            # break ("name@example.org.\n\nNext Name"); no identifier spans a
            # line break, so split model groups at newlines first.
            for line in re.finditer(r"[^\n]+", text[start:end]):
                segment = _TRAILING_JUNK.sub("", line.group())
                trimmed = segment.lstrip()
                # A person or place never contains "@": that's the model
                # labeling the human name inside an email address.
                if not trimmed or "@" in trimmed:
                    continue
                line_start = start + line.start() + (len(segment) - len(trimmed))
                raw.append((entity_type, line_start, line_start + len(trimmed), score))

        raw.sort(key=lambda r: r[1])
        merged: list[tuple[EntityType, int, int, float]] = []
        for etype, start, end, score in raw:
            if merged:
                prev_type, prev_start, prev_end, prev_score = merged[-1]
                gap = text[prev_end:start]
                if etype is prev_type and prev_end <= start and _JOINABLE.match(gap):
                    merged[-1] = (prev_type, prev_start, end, min(prev_score, score))
                    continue
            merged.append((etype, start, end, score))

        return [
            Entity(
                type=etype,
                start=start,
                end=end,
                text=text[start:end],
                confidence=min(score, self.confidence_cap),
                stage=Stage.RULES,
            )
            for etype, start, end, score in merged
        ]
