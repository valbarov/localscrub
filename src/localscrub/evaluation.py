"""Per-entity evaluation of the cascade against a gold-labeled corpus.

Two matching modes are reported side by side:

- **relaxed** — type match plus any character overlap; the headline
  detection number (a detector that found the identifier but clipped a
  trailing character still found it);
- **strict** — type match plus exact boundaries; the calibration number.

Matching is one-to-one per note: each prediction consumes at most one gold
span and vice versa, so a double detection cannot inflate recall.

Separately, **redaction recall** is the safety number: the fraction of gold
spans whose every character is removed from the output, whether by an
accepted entity or a fail-closed escalation, regardless of the type either
was assigned. A system can have poor typing and still redact safely.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from localscrub.analyzer import Analyzer, Stage2, apply_stage2
from localscrub.config import ScrubConfig
from localscrub.corpus.generator import GoldSpan, SyntheticNote
from localscrub.entities import EntityType
from localscrub.recognizers import Recognizer


@dataclass(frozen=True, slots=True)
class TypeMetrics:
    """Counts and derived scores for one entity type (or the micro total)."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    notes: int
    gold_by_type: dict[EntityType, int]
    strict: dict[EntityType, TypeMetrics]
    relaxed: dict[EntityType, TypeMetrics]
    redacted_by_type: dict[EntityType, int]

    @property
    def total_gold(self) -> int:
        return sum(self.gold_by_type.values())

    @property
    def redaction_recall(self) -> float:
        total = self.total_gold
        return sum(self.redacted_by_type.values()) / total if total else 0.0

    def _micro(self, per_type: dict[EntityType, TypeMetrics]) -> TypeMetrics:
        return TypeMetrics(
            tp=sum(m.tp for m in per_type.values()),
            fp=sum(m.fp for m in per_type.values()),
            fn=sum(m.fn for m in per_type.values()),
        )

    @property
    def strict_overall(self) -> TypeMetrics:
        return self._micro(self.strict)

    @property
    def relaxed_overall(self) -> TypeMetrics:
        return self._micro(self.relaxed)

    def to_dict(self) -> dict[str, object]:
        types = sorted(
            set(self.gold_by_type) | set(self.strict) | set(self.relaxed),
            key=list(EntityType).index,
        )
        return {
            "notes": self.notes,
            "gold": self.total_gold,
            "redaction_recall": round(self.redaction_recall, 4),
            "relaxed_overall": self.relaxed_overall.to_dict(),
            "strict_overall": self.strict_overall.to_dict(),
            "types": {
                t.value: {
                    "gold": self.gold_by_type.get(t, 0),
                    "redacted": self.redacted_by_type.get(t, 0),
                    "relaxed": self.relaxed.get(t, TypeMetrics(0, 0, 0)).to_dict(),
                    "strict": self.strict.get(t, TypeMetrics(0, 0, 0)).to_dict(),
                }
                for t in types
            },
        }

    def format_table(self) -> str:
        """Markdown results table, one row per type plus the micro total."""
        header = (
            "| entity | gold | relaxed P | relaxed R | relaxed F1 "
            "| strict P | strict R | strict F1 | redacted |\n"
            "|---|---|---|---|---|---|---|---|---|"
        )
        types = sorted(
            set(self.gold_by_type) | set(self.strict) | set(self.relaxed),
            key=list(EntityType).index,
        )
        rows = []
        for t in types:
            gold = self.gold_by_type.get(t, 0)
            rel = self.relaxed.get(t, TypeMetrics(0, 0, 0))
            st = self.strict.get(t, TypeMetrics(0, 0, 0))
            redacted = self.redacted_by_type.get(t, 0)
            red = f"{redacted / gold:.2f}" if gold else "—"
            rows.append(
                f"| {t.value} | {gold} | {rel.precision:.2f} | {rel.recall:.2f} "
                f"| {rel.f1:.2f} | {st.precision:.2f} | {st.recall:.2f} "
                f"| {st.f1:.2f} | {red} |"
            )
        rel, st = self.relaxed_overall, self.strict_overall
        rows.append(
            f"| **all (micro)** | {self.total_gold} | {rel.precision:.2f} "
            f"| {rel.recall:.2f} | {rel.f1:.2f} | {st.precision:.2f} "
            f"| {st.recall:.2f} | {st.f1:.2f} | {self.redaction_recall:.2f} |"
        )
        return "\n".join([header, *rows])


@dataclass(frozen=True, slots=True)
class NotePrediction:
    """One system's output for one note, in scoring terms.

    ``spans`` are the typed detections; ``removed`` are the character
    intervals absent from the system's redacted output (for the cascade
    that is entities plus fail-closed escalations; for an external
    baseline it is usually just its detections). Keeping the two separate
    lets redaction recall credit an untyped removal that typed matching
    would not.
    """

    spans: tuple[tuple[EntityType, int, int], ...]
    removed: tuple[tuple[int, int], ...]


def _overlaps(a_start: int, a_end: int, b: GoldSpan) -> bool:
    return a_start < b.end and a_end > b.start


def _covered(span: GoldSpan, intervals: list[tuple[int, int]]) -> bool:
    """True if every character of ``span`` lies inside the interval union."""
    pos = span.start
    for start, end in sorted(intervals):
        if start > pos:
            break
        pos = max(pos, end)
        if pos >= span.end:
            return True
    return pos >= span.end


def score(
    notes: Sequence[SyntheticNote],
    predictions: Sequence[NotePrediction],
) -> EvalReport:
    """Score per-note predictions against gold spans.

    System-agnostic: ``evaluate`` feeds it the cascade's output, and
    benchmark baselines (e.g. Presidio in ``scripts/bench_presidio.py``)
    feed it theirs, so every system is measured by identical matching
    and redaction-recall rules.
    """
    if len(notes) != len(predictions):
        raise ValueError(f"{len(notes)} notes but {len(predictions)} predictions")
    gold_by_type: dict[EntityType, int] = defaultdict(int)
    redacted_by_type: dict[EntityType, int] = defaultdict(int)
    counts: dict[bool, dict[EntityType, list[int]]] = {
        True: defaultdict(lambda: [0, 0, 0]),  # strict -> [tp, fp, fn]
        False: defaultdict(lambda: [0, 0, 0]),
    }

    for note, pred in zip(notes, predictions, strict=True):
        for span in note.spans:
            gold_by_type[span.type] += 1

        for strict in (True, False):
            matched: set[int] = set()
            for etype, start, end in pred.spans:
                hit = None
                for i, span in enumerate(note.spans):
                    if i in matched or span.type is not etype:
                        continue
                    if strict:
                        if (start, end) == (span.start, span.end):
                            hit = i
                            break
                    elif _overlaps(start, end, span):
                        hit = i
                        break
                if hit is None:
                    counts[strict][etype][1] += 1
                else:
                    matched.add(hit)
                    counts[strict][etype][0] += 1
            for i, span in enumerate(note.spans):
                if i not in matched:
                    counts[strict][span.type][2] += 1

        for span in note.spans:
            if _covered(span, list(pred.removed)):
                redacted_by_type[span.type] += 1

    return EvalReport(
        notes=len(notes),
        gold_by_type=dict(gold_by_type),
        strict={t: TypeMetrics(*c) for t, c in counts[True].items()},
        relaxed={t: TypeMetrics(*c) for t, c in counts[False].items()},
        redacted_by_type=dict(redacted_by_type),
    )


def evaluate(
    notes: Sequence[SyntheticNote],
    config: ScrubConfig | None = None,
    stage2: Stage2 | None = None,
    recognizers: Sequence[Recognizer] | None = None,
) -> EvalReport:
    """Run the analyzer (optionally cascading into stage 2) and score it."""
    cfg = config if config is not None else ScrubConfig()
    analyzer = Analyzer(recognizers=recognizers, config=cfg)
    predictions: list[NotePrediction] = []

    for note in notes:
        result = analyzer.analyze(note.text)
        if stage2 is not None:
            result = apply_stage2(result, stage2, cfg)
        removed = [(e.start, e.end) for e in result.entities]
        if cfg.redact_escalated:
            removed += [(e.start, e.end) for e in result.escalations]
        predictions.append(
            NotePrediction(
                spans=tuple((e.type, e.start, e.end) for e in result.entities),
                removed=tuple(removed),
            )
        )

    return score(notes, predictions)
