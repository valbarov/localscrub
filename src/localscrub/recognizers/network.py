"""URL and IP address recognizers."""

from __future__ import annotations

import ipaddress
import re
from typing import ClassVar

from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer

_VERSION_PREFIX = re.compile(r"(?:\bversion\.?\s*|\bv\.?\s*)$", re.IGNORECASE)


class UrlRecognizer(RegexRecognizer):
    """Scheme-full URLs at high confidence; bare ``www.`` hosts a notch lower.

    Bare domains without a scheme or ``www.`` are deliberately out of scope
    for stage 1 — too noisy for a rules pass.
    """

    name: ClassVar[str] = "url"
    entity_type: ClassVar[EntityType] = EntityType.URL
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(re.compile(r"https?://[^\s<>\"')\]]+"), confidence=0.90),
        Pattern(re.compile(r"\bwww\.[^\s<>\"')\]]+"), confidence=0.75),
    )


class IpAddressRecognizer(RegexRecognizer):
    """IPv4/IPv6 candidates validated with the ``ipaddress`` module.

    Dotted quads preceded by ``v``/``version`` (release strings like
    ``v1.2.3.4``) are demoted into the escalation band rather than rejected.
    """

    name: ClassVar[str] = "ip_address"
    entity_type: ClassVar[EntityType] = EntityType.IP_ADDRESS
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), confidence=0.90),
        Pattern(
            re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f:]*[0-9A-Fa-f]\b"),
            confidence=0.90,
        ),
    )
    context_words: ClassVar[frozenset[str]] = frozenset({"ip", "addr", "host"})
    context_boost: ClassVar[float] = 0.1

    def validate(self, match: re.Match[str]) -> float | None:
        try:
            ipaddress.ip_address(match.group())
        except ValueError:
            return None
        prefix = match.string[max(0, match.start() - 12) : match.start()]
        if _VERSION_PREFIX.search(prefix):
            return -0.2
        return 0.0
