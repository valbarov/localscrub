"""Stage-1 orchestration: run recognizers, resolve overlaps, decide escalation.

This module is the seam between the cascade's stages. Everything stage 2
needs — the spans it must adjudicate and why — arrives as ``Escalation``
records on the ``ScrubResult``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from localscrub.config import ScrubConfig
from localscrub.entities import Entity
from localscrub.recognizers import Recognizer, default_recognizers
from localscrub.redact import apply_redactions


@dataclass(frozen=True, slots=True)
class Escalation:
    """A span stage 1 could not settle, handed to stage 2 (or a human).

    ``reason`` is ``"low_confidence"`` (single candidate in the escalation
    band) or ``"ambiguous_type"`` (two detections of different types competed
    for the span; ``candidates`` holds them best-first).
    """

    start: int
    end: int
    text: str
    reason: str
    candidates: tuple[Entity, ...]


@dataclass(frozen=True, slots=True)
class ScrubResult:
    """Everything the pipeline produced for one document."""

    text: str
    redacted: str
    entities: tuple[Entity, ...]
    escalations: tuple[Escalation, ...]
    placeholders: dict[int, str] = field(default_factory=dict)

    @property
    def escalated(self) -> bool:
        """Document-level flag: does anything here need stage 2?"""
        return bool(self.escalations)


class Analyzer:
    """Runs recognizers over text and applies thresholding + overlap rules."""

    def __init__(
        self,
        recognizers: Sequence[Recognizer] | None = None,
        config: ScrubConfig | None = None,
    ) -> None:
        self.recognizers = tuple(recognizers) if recognizers is not None else default_recognizers()
        self.config = config or ScrubConfig()

    def analyze(self, text: str) -> ScrubResult:
        config = self.config
        candidates = [
            entity
            for recognizer in self.recognizers
            for entity in recognizer.analyze(text)
            if entity.confidence >= config.floor_threshold
        ]
        # Highest confidence first; longer span breaks ties (prefer the more
        # complete match); position for determinism.
        candidates.sort(key=lambda e: (-e.confidence, e.start - e.end, e.start))

        accepted: list[Entity] = []
        escalations: list[Escalation] = []
        for candidate in candidates:
            winner = next(
                (a for a in accepted if a.start < candidate.end and candidate.start < a.end),
                None,
            )
            if winner is None:
                accepted.append(candidate)
            elif (
                candidate.type != winner.type
                and winner.confidence - candidate.confidence <= config.ambiguity_margin
            ):
                escalations.append(
                    Escalation(
                        start=candidate.start,
                        end=candidate.end,
                        text=candidate.text,
                        reason="ambiguous_type",
                        candidates=(winner, candidate),
                    )
                )

        final = [e for e in accepted if e.confidence >= config.accept_threshold]
        low_band = [e for e in accepted if e.confidence < config.accept_threshold]
        escalations.extend(
            Escalation(start=e.start, end=e.end, text=e.text, reason="low_confidence",
                       candidates=(e,))
            for e in low_band
        )

        to_redact = final + low_band if config.redact_escalated else final
        to_redact.sort(key=lambda e: e.start)
        redacted, placeholders = apply_redactions(
            text, to_redact, numbered=config.numbered_placeholders
        )
        escalations.sort(key=lambda e: e.start)
        return ScrubResult(
            text=text,
            redacted=redacted,
            entities=tuple(to_redact),
            escalations=tuple(escalations),
            placeholders=placeholders,
        )


def scrub(text: str, config: ScrubConfig | None = None) -> ScrubResult:
    """Scrub ``text`` with the default recognizer set."""
    return Analyzer(config=config).analyze(text)
