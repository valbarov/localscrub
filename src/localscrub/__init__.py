"""localscrub — local-first PHI/PII de-identification.

Two-stage cascade: fast rule/NER detection for the obvious identifiers,
a local LLM for the context-dependent remainder. No cloud APIs.
"""

from localscrub.entities import SAFE_HARBOR, Entity, EntityType, Stage

__version__ = "0.0.1.dev0"

__all__ = ["SAFE_HARBOR", "Entity", "EntityType", "Stage", "__version__"]
