import pytest

from localscrub.corpus import GoldSpan, SyntheticNote, generate_corpus
from localscrub.entities import EntityType
from localscrub.evaluation import TypeMetrics, evaluate

REPORT = evaluate(generate_corpus(100, seed=13))

STRONG = (
    EntityType.EMAIL,
    EntityType.SSN,
    EntityType.MRN,
    EntityType.HEALTH_PLAN_ID,
    EntityType.ACCOUNT_NUMBER,
    EntityType.IP_ADDRESS,
    EntityType.CREDIT_CARD,
    EntityType.PHONE,
    EntityType.FAX,
    EntityType.DATE,
)


def note(text, spans):
    return SyntheticNote(id="t", note_type="test", text=text, spans=tuple(spans))


def test_type_metrics_derivations():
    m = TypeMetrics(tp=8, fp=2, fn=2)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)
    assert TypeMetrics(0, 0, 0).f1 == 0.0


def test_exact_hit_counts_in_both_modes():
    text = "Contact test@example.com today."
    report = evaluate([note(text, [GoldSpan(EntityType.EMAIL, 8, 24, "test@example.com")])])
    assert report.strict[EntityType.EMAIL].tp == 1
    assert report.relaxed[EntityType.EMAIL].tp == 1
    assert report.redaction_recall == 1.0


def test_wrong_type_is_fp_plus_fn_but_still_redacted():
    # Gold mislabels the span; detection metrics punish it, redaction doesn't care.
    text = "Contact test@example.com today."
    report = evaluate([note(text, [GoldSpan(EntityType.PHONE, 8, 24, "test@example.com")])])
    assert report.relaxed[EntityType.EMAIL].fp == 1
    assert report.relaxed[EntityType.PHONE].fn == 1
    assert report.redaction_recall == 1.0


def test_boundary_slip_is_relaxed_hit_strict_miss():
    # The URL recognizer swallows the trailing period; gold does not include it.
    text = "See https://www.example.com/x. Thanks."
    report = evaluate(
        [note(text, [GoldSpan(EntityType.URL, 4, 29, "https://www.example.com/x")])]
    )
    assert report.relaxed[EntityType.URL].tp == 1
    assert report.strict[EntityType.URL].tp == 0
    assert report.strict[EntityType.URL].fp == 1


def test_undetected_gold_is_fn_and_unredacted():
    text = "Seen by Marisol Norwood on rounds."
    report = evaluate([note(text, [GoldSpan(EntityType.NAME, 8, 23, "Marisol Norwood")])])
    assert report.relaxed[EntityType.NAME].fn == 1
    assert report.redaction_recall == 0.0


def test_double_detection_cannot_inflate_recall():
    # Two same-type gold spans, one detectable value each; matching is one-to-one.
    text = "Contact test@example.com or test@example.com."
    spans = [
        GoldSpan(EntityType.EMAIL, 8, 24, "test@example.com"),
        GoldSpan(EntityType.EMAIL, 28, 44, "test@example.com"),
    ]
    report = evaluate([note(text, spans)])
    assert report.relaxed[EntityType.EMAIL].tp == 2
    assert report.relaxed[EntityType.EMAIL].fp == 0


# --- regression floors on the synthetic corpus (deterministic, seed 13) ---


@pytest.mark.parametrize("etype", STRONG)
def test_structured_types_relaxed_floor(etype):
    metrics = REPORT.relaxed[etype]
    assert REPORT.gold_by_type[etype] > 0
    assert metrics.recall >= 0.95
    assert metrics.precision >= 0.95


@pytest.mark.parametrize("etype", STRONG)
def test_structured_types_fully_redacted(etype):
    assert REPORT.redacted_by_type[etype] == REPORT.gold_by_type[etype]


def test_names_and_geo_are_honest_misses():
    # Stage 1 has no NAME/GEO recognizer ([ner] extra / stage 2); when that
    # lands, raise these into floors instead of zero-assertions.
    assert REPORT.relaxed.get(EntityType.NAME, TypeMetrics(0, 0, 0)).tp == 0
    assert REPORT.relaxed.get(EntityType.GEO_SUBDIVISION, TypeMetrics(0, 0, 0)).tp == 0


def test_report_round_trips_to_dict():
    data = REPORT.to_dict()
    assert data["notes"] == 100
    assert data["gold"] == REPORT.total_gold
    assert data["types"]["EMAIL"]["relaxed"]["recall"] >= 0.95


def test_format_table_has_micro_row():
    table = REPORT.format_table()
    assert "**all (micro)**" in table
    assert "| EMAIL |" in table
