"""US Social Security number recognizer."""

from __future__ import annotations

import re
from typing import ClassVar

from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer


class SsnRecognizer(RegexRecognizer):
    """SSNs with SSA structural rules (no 000/666/9xx area, 00 group, 0000 serial).

    A separated ``123-45-6789`` is strong on its own (0.85). A bare 9-digit run
    is noise-floor (0.30) — reportable only when a context word like ``SSN:``
    boosts it into the escalation band, where stage 2 (or a human) adjudicates.
    """

    name: ClassVar[str] = "ssn"
    entity_type: ClassVar[EntityType] = EntityType.SSN
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(
            re.compile(
                r"(?<![\d-])(?P<area>\d{3})[- ](?P<grp>\d{2})[- ](?P<serial>\d{4})(?![\d-])"
            ),
            confidence=0.85,
        ),
        Pattern(
            re.compile(r"(?<![\d-])(?P<area>\d{3})(?P<grp>\d{2})(?P<serial>\d{4})(?![\d-])"),
            confidence=0.30,
        ),
    )
    context_words: ClassVar[frozenset[str]] = frozenset({"ssn", "social security"})

    def validate(self, match: re.Match[str]) -> float | None:
        area = match.group("area")
        if area == "000" or area == "666" or area.startswith("9"):
            return None
        if match.group("grp") == "00" or match.group("serial") == "0000":
            return None
        return 0.0
