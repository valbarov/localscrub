"""Email, phone, and fax recognizers."""

from __future__ import annotations

import re
from typing import ClassVar

from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer


class EmailRecognizer(RegexRecognizer):
    name: ClassVar[str] = "email"
    entity_type: ClassVar[EntityType] = EntityType.EMAIL
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
            confidence=0.95,
        ),
    )


class PhoneFaxRecognizer(RegexRecognizer):
    """NANP phone numbers; emitted as FAX when 'fax' appears in the context window.

    A bare 7-digit local number is plausible-but-weak (0.5): it lands in the
    escalation band unless a context word lifts it.
    """

    name: ClassVar[str] = "phone"
    entity_type: ClassVar[EntityType] = EntityType.PHONE
    patterns: ClassVar[tuple[Pattern, ...]] = (
        # (555) 123-4567, +1 (555) 123-4567
        Pattern(
            re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(\d{3}\)\s?\d{3}[-. ]\d{4}(?!\d)"),
            confidence=0.70,
        ),
        # 555-123-4567, 555.123.4567, +1 555 123 4567
        Pattern(
            re.compile(r"(?<![\d-])(?:\+?1[-. ])?\d{3}[-. ]\d{3}[-. ]\d{4}(?![\d-])"),
            confidence=0.70,
        ),
        # 7-digit local: 123-4567
        Pattern(
            re.compile(r"(?<![\d-])\d{3}[-.]\d{4}(?![\d-])"),
            confidence=0.50,
        ),
    )
    context_words: ClassVar[frozenset[str]] = frozenset(
        {"phone", "cell", "mobile", "tel", "call", "fax", "reach"}
    )

    def entity_type_for(self, match: re.Match[str], window: str) -> EntityType:
        if "fax" in window:
            return EntityType.FAX
        return EntityType.PHONE
