import pytest

from localscrub.analyzer import scrub
from localscrub.entities import EntityType, Stage
from localscrub.recognizers import NerRecognizer, default_recognizers

NOTE = "Pt Alma Quimby lives at 4050 Mossbank Blvd, Cedar Vale, OH 86484 today."


class FakePipeline:
    def __init__(self, items):
        self.items = items
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return self.items


def item(group, start, end, score=0.99):
    return {"entity_group": group, "start": start, "end": end, "score": score}


def test_label_mapping_and_stage():
    pipe = FakePipeline([item("PATIENT", 3, 14), item("LOC", 24, 41)])
    [name, geo] = NerRecognizer(pipeline=pipe).analyze(NOTE)
    assert name.type is EntityType.NAME
    assert name.text == "Alma Quimby"
    assert geo.type is EntityType.GEO_SUBDIVISION
    assert name.stage is Stage.RULES


def test_unmapped_labels_ignored():
    # Regex recognizers own dates/ages; the model's AGE/DATE labels must not leak in.
    pipe = FakePipeline([item("AGE", 0, 2), item("DATE", 66, 71), item("HOSP", 3, 14)])
    assert NerRecognizer(pipeline=pipe).analyze(NOTE) == []


def test_adjacent_same_type_spans_merge():
    # "4050 Mossbank Blvd" + "Cedar Vale" + "OH 86484" -> one address span.
    pipe = FakePipeline(
        [item("LOC", 24, 42), item("LOC", 44, 54), item("LOC", 56, 64, score=0.91)]
    )
    [geo] = NerRecognizer(pipeline=pipe).analyze(NOTE)
    assert geo.text == "4050 Mossbank Blvd, Cedar Vale, OH 86484"
    assert geo.confidence == pytest.approx(0.91)  # merged span keeps the weakest score


def test_different_types_do_not_merge():
    pipe = FakePipeline([item("PATIENT", 3, 14), item("LOC", 16, 21)])
    entities = NerRecognizer(pipeline=pipe).analyze(NOTE)
    assert len(entities) == 2


def test_distant_same_type_spans_do_not_merge():
    pipe = FakePipeline([item("LOC", 24, 42), item("LOC", 56, 64)])
    entities = NerRecognizer(pipeline=pipe).analyze(NOTE)
    assert len(entities) == 2


def test_no_merge_across_newlines_or_periods():
    # A name-in-email and the next paragraph's name must never merge into one
    # span; gluing them once cost EMAIL recall 1.00 -> 0.53 in the 50-note
    # eval. The email-shaped "name" is dropped outright (@ filter).
    text = "mail hollis.sablewood@example.org.\n\nQuentin Okafor signed."
    pipe = FakePipeline(
        [item("PATIENT", 5, 33), item("PATIENT", 36, 50)]
    )
    entities = NerRecognizer(pipeline=pipe).analyze(text)
    assert [e.text for e in entities] == ["Quentin Okafor"]


def test_single_model_group_split_at_newlines():
    # The pipeline itself can aggregate one label run across a paragraph
    # break; the recognizer must split it back into per-line spans (and the
    # email-shaped line is dropped by the @ filter).
    text = "mail hollis.sablewood@example.org.\n\nQuentin Okafor signed."
    pipe = FakePipeline([item("PATIENT", 5, 50)])
    [entity] = NerRecognizer(pipeline=pipe).analyze(text)
    assert entity.text == "Quentin Okafor"
    assert text[entity.start : entity.end] == entity.text


def test_email_beats_name_in_email_adjudication():
    # The model labels the email's local part as a patient name; the EMAIL
    # detection must still win the overlap and keep its type.
    text = "Send to hollis.sablewood@example.org today."
    pipe = FakePipeline([item("PATIENT", 8, 36)])
    recognizers = (*default_recognizers(), NerRecognizer(pipeline=pipe))
    result = scrub(text, recognizers=recognizers)
    [email] = [e for e in result.entities if e.text == "hollis.sablewood@example.org"]
    assert email.type is EntityType.EMAIL


def test_whitespace_trimmed_and_confidence_capped():
    text = "at  Cedar Vale  now"
    pipe = FakePipeline([{"entity_group": "LOC", "start": 2, "end": 16, "score": 0.9999}])
    [geo] = NerRecognizer(pipeline=pipe).analyze(text)
    assert geo.text == "Cedar Vale"
    assert text[geo.start : geo.end] == "Cedar Vale"
    assert geo.confidence == pytest.approx(0.95)


def test_low_score_lands_in_escalation_band():
    pipe = FakePipeline([item("PATIENT", 3, 14, score=0.6)])
    recognizers = (*default_recognizers(), NerRecognizer(pipeline=pipe))
    result = scrub(NOTE, recognizers=recognizers)
    assert any(esc.text == "Alma Quimby" for esc in result.escalations)
    assert "Alma Quimby" not in result.redacted  # fail-closed redaction


def test_integrates_with_default_recognizers():
    pipe = FakePipeline([item("PATIENT", 3, 14)])
    recognizers = (*default_recognizers(), NerRecognizer(pipeline=pipe))
    result = scrub(NOTE + " Contact test@example.com.", recognizers=recognizers)
    types = {e.type for e in result.entities}
    assert EntityType.NAME in types
    assert EntityType.EMAIL in types
