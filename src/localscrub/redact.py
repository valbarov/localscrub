"""Placeholder rendering and offset-safe replacement."""

from __future__ import annotations

from collections.abc import Sequence

from localscrub.entities import Entity, EntityType


def apply_redactions(
    text: str, entities: Sequence[Entity], numbered: bool = True
) -> tuple[str, dict[int, str]]:
    """Replace entity spans with placeholders like ``[SSN_1]``.

    With ``numbered=True``, identical original text of the same type always
    gets the same index, preserving coreference for human reviewers and for
    the stage-2 model. Returns the redacted text and a mapping from each
    entity's start offset to its placeholder.

    Entities must be non-overlapping (the analyzer guarantees this).
    """
    ordered = sorted(entities, key=lambda e: e.start)
    counters: dict[EntityType, int] = {}
    assigned: dict[tuple[EntityType, str], str] = {}
    placeholders: dict[int, str] = {}
    for entity in ordered:
        if numbered:
            key = (entity.type, entity.text)
            if key not in assigned:
                counters[entity.type] = counters.get(entity.type, 0) + 1
                assigned[key] = f"[{entity.type.value}_{counters[entity.type]}]"
            placeholders[entity.start] = assigned[key]
        else:
            placeholders[entity.start] = f"[{entity.type.value}]"

    # Replace right-to-left so earlier offsets stay valid.
    redacted = text
    for entity in reversed(ordered):
        redacted = redacted[: entity.start] + placeholders[entity.start] + redacted[entity.end :]
    return redacted, placeholders
