"""Presidio rules-only baseline for the published benchmark — see docs/benchmark.md.

Runs Microsoft Presidio (`presidio-analyzer`, default recognizers, spaCy
`en_core_web_lg`) over a gold-labeled JSONL corpus and scores it with the
same harness as localscrub (`localscrub.evaluation.score`), so the two
systems are measured by identical matching and redaction-recall rules:

    uv run --group bench python scripts/bench_presidio.py data/corpora/eval.jsonl
    uv run --group bench python scripts/bench_presidio.py data/corpora/mtsamples-100.jsonl

Presidio is a benchmark baseline only, never a runtime dependency
(ADR 0001). Scoring notes:

- Presidio has one PHONE_NUMBER type; localscrub's gold distinguishes
  PHONE from FAX. Fax detections therefore count as PHONE false positives
  plus FAX misses in the typed metrics — but redaction recall, the safety
  number, credits them fully. Same story for any typed mismatch.
- Predictions with no localscrub equivalent (e.g. NRP) are excluded from
  the typed metrics but still count toward redaction recall — generous to
  the baseline, never to us.
- MRN and HEALTH_PLAN_ID have no Presidio recognizer at all; those rows
  measure exactly that.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from localscrub.corpus import read_jsonl
from localscrub.entities import EntityType
from localscrub.evaluation import NotePrediction, score

#: Presidio entity type -> localscrub EntityType. Unlisted types (NRP,
#: CRYPTO, IBAN_CODE, ...) have no gold equivalent; see module docstring.
PRESIDIO_TO_LOCALSCRUB: dict[str, EntityType] = {
    "PERSON": EntityType.NAME,
    "LOCATION": EntityType.GEO_SUBDIVISION,
    "DATE_TIME": EntityType.DATE,
    "PHONE_NUMBER": EntityType.PHONE,
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "US_SSN": EntityType.SSN,
    "US_BANK_NUMBER": EntityType.ACCOUNT_NUMBER,
    "US_DRIVER_LICENSE": EntityType.LICENSE_NUMBER,
    "MEDICAL_LICENSE": EntityType.LICENSE_NUMBER,
    "URL": EntityType.URL,
    "IP_ADDRESS": EntityType.IP_ADDRESS,
    "CREDIT_CARD": EntityType.CREDIT_CARD,
    "US_PASSPORT": EntityType.PASSPORT,
    "US_ITIN": EntityType.NATIONAL_ID,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("corpus", type=Path, help="gold-labeled JSONL corpus")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="minimum Presidio confidence to keep a detection (default 0.35, "
        "the value Presidio's own demo uses)",
    )
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args()

    notes = read_jsonl(args.corpus)
    if not notes:
        raise SystemExit(f"corpus {args.corpus} is empty")

    warnings.filterwarnings("ignore")
    from presidio_analyzer import AnalyzerEngine

    t0 = time.perf_counter()
    engine = AnalyzerEngine(default_score_threshold=args.score_threshold)
    engine.analyze(text="warm-up", language="en")
    load_s = time.perf_counter() - t0

    predictions: list[NotePrediction] = []
    unmapped: dict[str, int] = {}
    t0 = time.perf_counter()
    for note in notes:
        results = engine.analyze(text=note.text, language="en")
        spans: list[tuple[EntityType, int, int]] = []
        removed: list[tuple[int, int]] = []
        for r in results:
            removed.append((r.start, r.end))
            mapped = PRESIDIO_TO_LOCALSCRUB.get(r.entity_type)
            if mapped is None:
                unmapped[r.entity_type] = unmapped.get(r.entity_type, 0) + 1
            else:
                spans.append((mapped, r.start, r.end))
        predictions.append(NotePrediction(spans=tuple(spans), removed=tuple(removed)))
    latency_ms = (time.perf_counter() - t0) * 1000 / len(notes)

    report = score(notes, predictions)
    if args.json:
        data = report.to_dict()
        data["latency_ms_per_note"] = round(latency_ms, 1)
        data["unmapped_predictions"] = unmapped
        print(json.dumps(data, indent=2))
    else:
        print(report.format_table())
        print(
            f"\n{report.notes} notes, {report.total_gold} gold spans; "
            f"redaction recall {report.redaction_recall:.3f}; "
            f"{latency_ms:.0f} ms/note (engine load {load_s:.1f} s, excluded)"
        )
        if unmapped:
            dropped = ", ".join(f"{k}×{v}" for k, v in sorted(unmapped.items()))
            print(f"unmapped predictions (redaction credit only): {dropped}")


if __name__ == "__main__":
    main()
