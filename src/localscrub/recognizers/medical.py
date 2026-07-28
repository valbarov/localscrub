"""Medical record and health plan identifier recognizers.

These identifiers have no intrinsic format — a bare MRN is indistinguishable
from any other number — so stage 1 only detects the *labeled* forms, where the
label itself is the evidence. Unlabeled instances are context-dependent PHI
and belong to stage 2.
"""

from __future__ import annotations

import re
from typing import ClassVar

from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer


class MrnRecognizer(RegexRecognizer):
    name: ClassVar[str] = "mrn"
    entity_type: ClassVar[EntityType] = EntityType.MRN
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(
            re.compile(
                r"\b(?:MRN|medical record (?:number|no\.?)|record\s*#)\s*[:#]?\s*"
                r"(?P<id>(?=[A-Za-z-]*\d)[A-Za-z0-9-]{5,12})",
                re.IGNORECASE,
            ),
            confidence=0.90,
            group="id",
        ),
    )


class HealthPlanIdRecognizer(RegexRecognizer):
    name: ClassVar[str] = "health_plan_id"
    entity_type: ClassVar[EntityType] = EntityType.HEALTH_PLAN_ID
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(
            re.compile(
                r"\b(?:member|policy|plan|beneficiary|subscriber)\s*(?:id|number|no\.?|#)"
                r"\s*[:#]?\s*(?P<id>(?=[A-Za-z-]*\d)[A-Za-z0-9-]{5,17})",
                re.IGNORECASE,
            ),
            confidence=0.85,
            group="id",
        ),
    )
