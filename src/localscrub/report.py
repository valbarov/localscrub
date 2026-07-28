"""JSON entity report and JSONL audit log.

The *report* is the local review artifact: it contains the original
identifier text so a human can verify each decision. Treat it with the same
care as the source document.

The *audit log* is the durable trail: one JSONL line per decision, offsets
and types only — never the identifier text — so it can be retained after the
source documents are gone.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localscrub.analyzer import ScrubResult

SCHEMA_VERSION = 1


def build_report(result: ScrubResult, source: str | None = None) -> dict[str, Any]:
    """Build the JSON-serializable entity report for one document."""
    from localscrub import __version__

    escalated_spans = {(esc.start, esc.end) for esc in result.escalations}
    counts: dict[str, int] = {}
    for entity in result.entities:
        counts[entity.type.value] = counts.get(entity.type.value, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "localscrub", "version": __version__},
        "source": source,
        "escalated": result.escalated,
        "entities": [
            {
                "type": entity.type.value,
                "start": entity.start,
                "end": entity.end,
                "text": entity.text,
                "confidence": round(entity.confidence, 3),
                "stage": entity.stage.value,
                "placeholder": result.placeholders.get(entity.start),
                "escalated": (entity.start, entity.end) in escalated_spans,
            }
            for entity in result.entities
        ],
        "escalations": [
            {
                "start": esc.start,
                "end": esc.end,
                "text": esc.text,
                "reason": esc.reason,
                "candidates": [
                    {"type": c.type.value, "confidence": round(c.confidence, 3)}
                    for c in esc.candidates
                ],
            }
            for esc in result.escalations
        ],
        "stats": {"chars": len(result.text), "counts": counts},
    }


def write_audit(result: ScrubResult, source: str | None, path: Path) -> None:
    """Append one JSONL line per redaction/escalation decision to ``path``."""
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[dict[str, Any]] = [
        {
            "ts": ts,
            "source": source,
            "action": "redact",
            "type": entity.type.value,
            "start": entity.start,
            "end": entity.end,
            "confidence": round(entity.confidence, 3),
            "stage": entity.stage.value,
        }
        for entity in result.entities
    ]
    lines.extend(
        {
            "ts": ts,
            "source": source,
            "action": "escalate",
            "reason": esc.reason,
            "start": esc.start,
            "end": esc.end,
        }
        for esc in result.escalations
    )
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
