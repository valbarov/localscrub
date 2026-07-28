"""Date and age recognizers.

HIPAA Safe Harbor (45 CFR 164.514(b)(2)(i)(C)) covers all date elements more
specific than a year, plus ages over 89. Bare years are therefore excluded by
design. Blood-pressure-style fractions (``120/80``) never match: the numeric
pattern requires three components.
"""

from __future__ import annotations

import re
from datetime import date
from typing import ClassVar

from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)


class DateRecognizer(RegexRecognizer):
    """ISO and month-name dates at 0.90; ambiguous numeric dates at 0.70.

    ``01/02/03``-style dates are calendar-checked under both MDY and DMY
    readings; they stay in the escalation band because the reading itself is
    ambiguous.
    """

    name: ClassVar[str] = "date"
    entity_type: ClassVar[EntityType] = EntityType.DATE
    patterns: ClassVar[tuple[Pattern, ...]] = (
        # 2024-03-15
        Pattern(
            re.compile(r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)"),
            confidence=0.90,
        ),
        # March 15, 2024 / Mar. 15 2024
        Pattern(
            re.compile(
                rf"\b{_MONTH}\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
                re.IGNORECASE,
            ),
            confidence=0.90,
        ),
        # 15 March 2024 / 15 Mar. 2024
        Pattern(
            re.compile(rf"\b(?P<day>\d{{1,2}})\s+{_MONTH}\.?,?\s+\d{{4}}\b", re.IGNORECASE),
            confidence=0.90,
        ),
        # 3/15/2024, 15-3-24 — ambiguous ordering
        Pattern(
            re.compile(r"(?<![\d/-])(?P<a>\d{1,2})[/-](?P<b>\d{1,2})[/-](?P<yy>\d{2,4})(?![\d/-])"),
            confidence=0.70,
        ),
    )
    context_words: ClassVar[frozenset[str]] = frozenset(
        {"dob", "born", "birth", "admitted", "discharged", "seen on", "visit"}
    )

    def validate(self, match: re.Match[str]) -> float | None:
        groups = match.groupdict()
        if groups.get("y"):
            try:
                date(int(groups["y"]), int(groups["m"]), int(groups["d"]))
            except ValueError:
                return None
            return 0.0
        if groups.get("yy"):
            a, b = int(groups["a"]), int(groups["b"])
            mdy = 1 <= a <= 12 and 1 <= b <= 31
            dmy = 1 <= b <= 12 and 1 <= a <= 31
            return 0.0 if mdy or dmy else None
        if groups.get("day"):
            return 0.0 if 1 <= int(groups["day"]) <= 31 else None
        return 0.0


class AgeOver89Recognizer(RegexRecognizer):
    """Ages over 89, a Safe Harbor date element. Ages 89 and under are not PHI."""

    name: ClassVar[str] = "age_over_89"
    entity_type: ClassVar[EntityType] = EntityType.DATE
    patterns: ClassVar[tuple[Pattern, ...]] = (
        # 92 years old / 92-year-old
        Pattern(
            re.compile(r"(?<!\d)(?P<age>\d{2,3})[-\s]*years?[-\s]old\b", re.IGNORECASE),
            confidence=0.90,
        ),
        # 92 yo / 92 y/o
        Pattern(
            re.compile(r"(?<!\d)(?P<age>\d{2,3})\s*y/?o\b", re.IGNORECASE),
            confidence=0.90,
        ),
        # age: 92 / aged 92
        Pattern(
            re.compile(r"\baged?\s*:?\s*(?P<age>\d{2,3})\b", re.IGNORECASE),
            confidence=0.90,
        ),
    )

    def validate(self, match: re.Match[str]) -> float | None:
        return 0.0 if int(match.group("age")) > 89 else None
