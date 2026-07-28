"""Synthetic corpus generation: render templates, record gold spans.

The generator is fully deterministic: the same ``(count, seed, note_types)``
always yields byte-identical notes, so evaluation numbers are reproducible
from the seed alone and corpora never need to be committed (they belong under
``data/corpora/``, which is gitignored).

Ground truth is a ``GoldSpan`` — a peer of ``localscrub.entities.Entity``
rather than a subtype, because gold labels carry no confidence and no stage.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from localscrub.corpus.templates import ENTITY_SLOTS, FILLER_BANKS, TEMPLATES, NoteTemplate
from localscrub.corpus.values import Fabricator
from localscrub.entities import EntityType

_SLOT = re.compile(r"\{([a-z_0-9]+)\}")


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """A planted identifier and its exact location in the note text."""

    type: EntityType
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid span [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"text length {len(self.text)} does not match span [{self.start}, {self.end})"
            )


@dataclass(frozen=True, slots=True)
class SyntheticNote:
    """One generated note: text plus its complete, exact ground truth."""

    id: str
    note_type: str
    text: str
    spans: tuple[GoldSpan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "note_type": self.note_type,
            "text": self.text,
            "entities": [
                {"type": s.type.value, "start": s.start, "end": s.end, "text": s.text}
                for s in self.spans
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SyntheticNote:
        entities = data["entities"]
        assert isinstance(entities, list)
        return cls(
            id=str(data["id"]),
            note_type=str(data["note_type"]),
            text=str(data["text"]),
            spans=tuple(
                GoldSpan(
                    type=EntityType(e["type"]),
                    start=int(e["start"]),
                    end=int(e["end"]),
                    text=str(e["text"]),
                )
                for e in entities
            ),
        )


def _slot_key(slot: str) -> str:
    """Resolve a slot name to its registry key.

    An exact registry name wins (``age89`` is itself a key); otherwise the
    trailing digit that distinguishes independent repeats is stripped
    (``provider2`` -> ``provider``).
    """
    if slot in ENTITY_SLOTS or slot in FILLER_BANKS:
        return slot
    return slot.rstrip("0123456789")


def _fabricate(key: str, fab: Fabricator, values: dict[str, str]) -> str:
    if key in ("patient", "provider", "relative"):
        return fab.name()
    if key == "address":
        return fab.address()
    if key in ("dob", "visit_date", "admit_date", "discharge_date", "followup_date"):
        return fab.date()
    if key == "age89":
        return fab.age_over_89()
    if key in ("phone", "fax"):
        return fab.phone()
    if key == "email":
        return fab.email(values.get("patient"))
    if key == "ssn":
        return fab.ssn()
    if key == "mrn":
        return fab.mrn()
    if key == "plan_id":
        return fab.health_plan_id()
    if key == "account":
        return fab.account_number()
    if key == "url":
        return fab.url()
    if key == "ip":
        return fab.ip_address()
    if key == "credit_card":
        return fab.credit_card()
    raise KeyError(f"no fabricator for entity slot {key!r}")


def _render(template: NoteTemplate, fab: Fabricator) -> tuple[str, tuple[GoldSpan, ...]]:
    values: dict[str, str] = {}
    parts: list[str] = []
    spans: list[GoldSpan] = []
    length = 0
    cursor = 0
    for match in _SLOT.finditer(template.body):
        literal = template.body[cursor : match.start()]
        parts.append(literal)
        length += len(literal)
        slot = match.group(1)
        key = _slot_key(slot)
        if slot not in values:
            if key in ENTITY_SLOTS:
                values[slot] = _fabricate(key, fab, values)
            elif key in FILLER_BANKS:
                values[slot] = fab.rng.choice(FILLER_BANKS[key])
            else:
                raise KeyError(f"unknown slot {slot!r} in template {template.note_type!r}")
        value = values[slot]
        entity_type = ENTITY_SLOTS.get(key)
        if entity_type is not None:
            spans.append(GoldSpan(entity_type, length, length + len(value), value))
        parts.append(value)
        length += len(value)
        cursor = match.end()
    parts.append(template.body[cursor:])
    return "".join(parts), tuple(spans)


def generate_corpus(
    count: int,
    seed: int = 0,
    note_types: Sequence[str] | None = None,
) -> list[SyntheticNote]:
    """Generate ``count`` synthetic notes, deterministically from ``seed``."""
    templates = TEMPLATES
    if note_types is not None:
        wanted = set(note_types)
        known = {t.note_type for t in TEMPLATES}
        if unknown := wanted - known:
            raise ValueError(
                f"unknown note type(s) {sorted(unknown)}; known: {sorted(known)}"
            )
        templates = tuple(t for t in TEMPLATES if t.note_type in wanted)
    rng = random.Random(seed)
    fab = Fabricator(rng)
    notes = []
    for i in range(count):
        template = rng.choice(templates)
        text, spans = _render(template, fab)
        notes.append(
            SyntheticNote(id=f"synth-{i:06d}", note_type=template.note_type, text=text, spans=spans)
        )
    return notes


def write_jsonl(notes: Iterable[SyntheticNote], path: Path) -> None:
    """Write one JSON object per note; the eval harness's input format."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for note in notes:
            fh.write(json.dumps(note.to_dict(), ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: Path) -> list[SyntheticNote]:
    """Read a corpus written by :func:`write_jsonl`."""
    notes = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                notes.append(SyntheticNote.from_dict(json.loads(line)))
    return notes
