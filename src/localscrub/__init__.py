"""localscrub — local-first PHI/PII de-identification.

Two-stage cascade: fast rule/NER detection for the obvious identifiers,
a local LLM for the context-dependent remainder. No cloud APIs.

>>> from localscrub import scrub
>>> result = scrub("Reach the patient at test@example.com.")
>>> result.redacted
'Reach the patient at [EMAIL_1].'

Cascade into stage 2 with a local model:

>>> from localscrub import LlmExtractor, scrub          # doctest: +SKIP
>>> scrub(text, stage2=LlmExtractor()).redacted         # doctest: +SKIP
"""

from localscrub.analyzer import Analyzer, Escalation, ScrubResult, apply_stage2, scrub
from localscrub.config import ScrubConfig
from localscrub.entities import SAFE_HARBOR, Entity, EntityType, Stage
from localscrub.llm import OllamaClient, OllamaError
from localscrub.stage2 import LlmExtractor

__version__ = "0.1.0"

__all__ = [
    "SAFE_HARBOR",
    "Analyzer",
    "Entity",
    "EntityType",
    "Escalation",
    "LlmExtractor",
    "OllamaClient",
    "OllamaError",
    "ScrubConfig",
    "ScrubResult",
    "Stage",
    "__version__",
    "apply_stage2",
    "scrub",
]
