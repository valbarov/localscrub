"""Recognizer framework for stage 1 (rules).

Confidence calibration contract
-------------------------------
Recognizers assign confidence on a shared scale so the analyzer's thresholds
mean the same thing across entity types:

- ``0.95-1.0``  structurally near-unambiguous or checksum/label-validated
  (an email address; a Luhn-valid card number; ``MRN: 4859302``)
- ``0.80-0.94`` strong format, validated (dashed SSN passing area rules;
  ISO date; parseable IP address; scheme-full URL)
- ``0.40-0.79`` plausible format that needs context (bare NANP phone number;
  ambiguous numeric date) — the escalation band
- ``< 0.40``    noise floor (a bare 9-digit run with no context); dropped by
  the analyzer unless a context word boosts it into the band above

Context words found near a match add ``context_boost`` to its confidence,
capped at 1.0. A recognizer's ``validate`` hook can adjust confidence further
or reject the match outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from localscrub.entities import Entity, EntityType, Stage


@runtime_checkable
class Recognizer(Protocol):
    """Anything that can find entities in text."""

    name: str

    def analyze(self, text: str) -> list[Entity]: ...


@dataclass(frozen=True, slots=True)
class Pattern:
    """One regex a recognizer matches, with its base confidence.

    ``group`` selects which regex group becomes the entity span — useful for
    labeled identifiers where the label (``MRN:``) is evidence but only the
    value should be redacted.
    """

    regex: re.Pattern[str]
    confidence: float
    group: int | str = 0


class RegexRecognizer:
    """Config-driven recognizer: patterns + optional validation + context boost.

    Subclasses set the class attributes and may override ``validate`` (checksum
    or range checks; return a confidence delta, or ``None`` to reject the
    match) and ``entity_type_for`` (per-match type choice, e.g. FAX vs PHONE).
    """

    name: ClassVar[str]
    entity_type: ClassVar[EntityType]
    patterns: ClassVar[tuple[Pattern, ...]]
    context_words: ClassVar[frozenset[str]] = frozenset()
    context_boost: ClassVar[float] = 0.2
    window_before: ClassVar[int] = 40
    window_after: ClassVar[int] = 10

    def analyze(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                delta = self.validate(match)
                if delta is None:
                    continue
                start, end = match.span(pattern.group)
                window = self._window(text, start, end)
                confidence = pattern.confidence + delta
                if self.context_words and any(w in window for w in self.context_words):
                    confidence += self.context_boost
                confidence = min(max(confidence, 0.0), 1.0)
                entities.append(
                    Entity(
                        type=self.entity_type_for(match, window),
                        start=start,
                        end=end,
                        text=text[start:end],
                        confidence=confidence,
                        stage=Stage.RULES,
                    )
                )
        return entities

    def validate(self, match: re.Match[str]) -> float | None:
        """Return a confidence delta for the match, or ``None`` to reject it."""
        return 0.0

    def entity_type_for(self, match: re.Match[str], window: str) -> EntityType:
        """Choose the entity type for a match; ``window`` is lowercased context."""
        return self.entity_type

    def _window(self, text: str, start: int, end: int) -> str:
        lo = max(0, start - self.window_before)
        hi = min(len(text), end + self.window_after)
        return text[lo:hi].lower()
