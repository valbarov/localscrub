"""Stage-1 orchestration: run recognizers, resolve overlaps, decide escalation.

This module is the seam between the cascade's stages. Everything stage 2
needs — the spans it must adjudicate and why — arrives as ``Escalation``
records on the ``ScrubResult``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

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


class Stage2(Protocol):
    """Anything that can extract context-dependent PHI from a document."""

    def extract(self, text: str, escalations: Sequence[Escalation]) -> Sequence[Entity]: ...


def apply_stage2(
    result: ScrubResult, stage2: Stage2, config: ScrubConfig | None = None
) -> ScrubResult:
    """Merge a stage-2 pass into a stage-1 result (additive, fail-closed).

    Stage-2 entities that overlap a stage-1 detection are dropped — rules
    are better calibrated where rules apply. Escalations the model confirmed
    (any extraction overlaps them) are resolved; the rest stay escalated:
    stage-2 silence never unredacts anything (ADR 0002).
    """
    cfg = config if config is not None else ScrubConfig()
    extracted = tuple(stage2.extract(result.text, result.escalations))
    if not extracted:
        return result

    added = [
        e
        for e in extracted
        if not any(e.start < a.end and a.start < e.end for a in result.entities)
    ]
    entities = sorted([*result.entities, *added], key=lambda e: e.start)
    remaining = tuple(
        esc
        for esc in result.escalations
        if not any(esc.start < e.end and e.start < esc.end for e in extracted)
    )
    redacted, placeholders = apply_redactions(
        result.text, entities, numbered=cfg.numbered_placeholders
    )
    return ScrubResult(
        text=result.text,
        redacted=redacted,
        entities=tuple(entities),
        escalations=remaining,
        placeholders=placeholders,
    )


def scrub(
    text: str,
    config: ScrubConfig | None = None,
    stage2: Stage2 | None = None,
    recognizers: Sequence[Recognizer] | None = None,
) -> ScrubResult:
    """Scrub ``text`` (default recognizer set unless given), optionally cascading."""
    result = Analyzer(recognizers=recognizers, config=config).analyze(text)
    if stage2 is not None:
        result = apply_stage2(result, stage2, config)
    return result
