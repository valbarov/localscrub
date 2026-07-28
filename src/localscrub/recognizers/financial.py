"""Payment card and account number recognizers."""

from __future__ import annotations

import re
from typing import ClassVar

from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer

_SEPARATORS = re.compile(r"[ -]")


def luhn_ok(digits: str) -> bool:
    """Luhn checksum (ISO/IEC 7812) over a string of ASCII digits."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _known_iin(digits: str) -> bool:
    """Major industry IIN prefixes (Visa, Mastercard, Amex, Discover)."""
    if digits.startswith("4") or digits[:4] == "6011":
        return True
    if digits[:2] in {"34", "37", "65"} or digits[:2] in {"51", "52", "53", "54", "55"}:
        return True
    return len(digits) >= 4 and 2221 <= int(digits[:4]) <= 2720


class CreditCardRecognizer(RegexRecognizer):
    """13-19 digit runs (optionally space/dash separated); Luhn must pass.

    A failed checksum rejects the match outright — a random digit run is not
    a card number. A recognized issuer prefix adds a small bonus.
    """

    name: ClassVar[str] = "credit_card"
    entity_type: ClassVar[EntityType] = EntityType.CREDIT_CARD
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"), confidence=0.90),
    )
    context_words: ClassVar[frozenset[str]] = frozenset(
        {"card", "visa", "mastercard", "amex", "cc"}
    )

    def validate(self, match: re.Match[str]) -> float | None:
        digits = _SEPARATORS.sub("", match.group())
        if not 13 <= len(digits) <= 19 or not luhn_ok(digits):
            return None
        return 0.05 if _known_iin(digits) else 0.0


class AccountNumberRecognizer(RegexRecognizer):
    """Labeled account numbers — the label is the evidence.

    The value must contain at least one digit so prose after the label
    ("account balance") is never mistaken for an identifier.
    """

    name: ClassVar[str] = "account_number"
    entity_type: ClassVar[EntityType] = EntityType.ACCOUNT_NUMBER
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(
            re.compile(
                r"\b(?:account|acct)\.?\s*(?:number|no\.?|#)?\s*[:#]?\s*"
                r"(?P<id>(?=[A-Za-z-]*\d)[A-Za-z0-9-]{5,17})",
                re.IGNORECASE,
            ),
            confidence=0.85,
            group="id",
        ),
    )
