"""localscrub — local-first PHI/PII de-identification.

Two-stage cascade: fast rule/NER detection for the obvious identifiers,
a local LLM for the context-dependent remainder. No cloud APIs.

>>> from localscrub import scrub
>>> result = scrub("Reach the patient at test@example.com.")
>>> result.redacted
'Reach the patient at [EMAIL_1].'
"""

from localscrub.analyzer import Analyzer, Escalation, ScrubResult, scrub
from localscrub.config import ScrubConfig
from localscrub.entities import SAFE_HARBOR, Entity, EntityType, Stage

__version__ = "0.0.1.dev0"

__all__ = [
    "SAFE_HARBOR",
    "Analyzer",
    "Entity",
    "EntityType",
    "Escalation",
    "ScrubConfig",
    "ScrubResult",
    "Stage",
    "__version__",
    "scrub",
]
