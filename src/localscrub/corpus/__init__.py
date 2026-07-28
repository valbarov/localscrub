"""Synthetic clinical-note corpus with exact ground truth (ROADMAP M2).

Notes are rendered from templates with planted, fully synthetic identifiers;
every planted span's type and character offsets are recorded, so per-entity
precision/recall can be computed without any human annotation — and without
any real PHI, ever.

    from localscrub.corpus import generate_corpus

    notes = generate_corpus(100, seed=42)
    notes[0].text                  # the note
    notes[0].spans                 # gold (type, start, end, text) tuples
"""

from __future__ import annotations

from localscrub.corpus.generator import (
    GoldSpan,
    SyntheticNote,
    generate_corpus,
    read_jsonl,
    write_jsonl,
)
from localscrub.corpus.templates import NOTE_TYPES, TEMPLATES, NoteTemplate

__all__ = [
    "NOTE_TYPES",
    "TEMPLATES",
    "GoldSpan",
    "NoteTemplate",
    "SyntheticNote",
    "generate_corpus",
    "read_jsonl",
    "write_jsonl",
]
