"""Entity taxonomy for localscrub.

Covers the 18 HIPAA Safe Harbor identifier categories
(45 CFR 164.514(b)(2)(i)(A)-(R)) plus generic PII types that fall outside
HIPAA but matter in practice (payment cards, passports, leaked credentials).

FULL_FACE_PHOTO cannot occur in text input; it is kept so entity reports can
state Safe Harbor coverage explicitly ("category Q: not applicable to text").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    """Which cascade stage produced a detection."""

    RULES = "rules"  # stage 1: regex / dictionary / NER
    LLM = "llm"  # stage 2: local model


class EntityType(StrEnum):
    # HIPAA Safe Harbor, 45 CFR 164.514(b)(2)(i), letters (A)-(R)
    NAME = "NAME"  # (A)
    GEO_SUBDIVISION = "GEO_SUBDIVISION"  # (B) address, city, ZIP, geocode
    DATE = "DATE"  # (C) all date elements except year; ages > 89
    PHONE = "PHONE"  # (D)
    FAX = "FAX"  # (E)
    EMAIL = "EMAIL"  # (F)
    SSN = "SSN"  # (G)
    MRN = "MRN"  # (H) medical record number
    HEALTH_PLAN_ID = "HEALTH_PLAN_ID"  # (I) health plan beneficiary number
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"  # (J)
    LICENSE_NUMBER = "LICENSE_NUMBER"  # (K) certificate / license number
    VEHICLE_ID = "VEHICLE_ID"  # (L) VIN, license plate
    DEVICE_ID = "DEVICE_ID"  # (M) device identifier / serial
    URL = "URL"  # (N)
    IP_ADDRESS = "IP_ADDRESS"  # (O)
    BIOMETRIC_ID = "BIOMETRIC_ID"  # (P) finger/voice prints
    FULL_FACE_PHOTO = "FULL_FACE_PHOTO"  # (Q) n/a for text; see module docstring
    OTHER_UNIQUE_ID = "OTHER_UNIQUE_ID"  # (R) any other unique identifier

    # Generic PII beyond HIPAA
    CREDIT_CARD = "CREDIT_CARD"
    PASSPORT = "PASSPORT"
    NATIONAL_ID = "NATIONAL_ID"  # non-US SSN equivalents
    SECRET = "SECRET"  # API keys, tokens, passwords


#: The 18 Safe Harbor categories, in (A)-(R) order.
SAFE_HARBOR: tuple[EntityType, ...] = (
    EntityType.NAME,
    EntityType.GEO_SUBDIVISION,
    EntityType.DATE,
    EntityType.PHONE,
    EntityType.FAX,
    EntityType.EMAIL,
    EntityType.SSN,
    EntityType.MRN,
    EntityType.HEALTH_PLAN_ID,
    EntityType.ACCOUNT_NUMBER,
    EntityType.LICENSE_NUMBER,
    EntityType.VEHICLE_ID,
    EntityType.DEVICE_ID,
    EntityType.URL,
    EntityType.IP_ADDRESS,
    EntityType.BIOMETRIC_ID,
    EntityType.FULL_FACE_PHOTO,
    EntityType.OTHER_UNIQUE_ID,
)


@dataclass(frozen=True, slots=True)
class Entity:
    """A single detected identifier in a source text.

    Offsets are Unicode codepoint positions into the original text,
    ``start`` inclusive, ``end`` exclusive.
    """

    type: EntityType
    start: int
    end: int
    text: str
    confidence: float
    stage: Stage

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid span [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"text length {len(self.text)} does not match span [{self.start}, {self.end})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
