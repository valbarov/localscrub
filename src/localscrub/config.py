"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScrubConfig:
    """Thresholds and output options for the scrub pipeline.

    Confidence bands (see ``recognizers.base`` for the calibration contract):

    - ``>= accept_threshold``: redacted, stage-1 final
    - ``[floor_threshold, accept_threshold)``: escalated to stage 2; with
      ``redact_escalated`` (the fail-closed default) the span is still
      redacted under its best-guess type
    - ``< floor_threshold``: dropped silently

    ``ambiguity_margin`` controls overlap adjudication: when two detections of
    *different* types compete for the same span and their confidences are
    within this margin, the loser is escalated rather than discarded.
    """

    accept_threshold: float = 0.8
    floor_threshold: float = 0.4
    ambiguity_margin: float = 0.1
    redact_escalated: bool = True
    numbered_placeholders: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.floor_threshold <= self.accept_threshold <= 1.0:
            raise ValueError(
                f"need 0 <= floor ({self.floor_threshold}) <= "
                f"accept ({self.accept_threshold}) <= 1"
            )
