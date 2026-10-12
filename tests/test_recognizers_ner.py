import pytest

from localscrub.analyzer import scrub
from localscrub.entities import EntityType, Stage
from localscrub.recognizers import NerBackend, NerRecognizer, default_recognizers
from localscrub.recognizers.ner import onnx_cache_dir

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
    pipe = FakePipeline([item("AGE", 0, 2), item("DATE", 66, 71)])
    assert NerRecognizer(pipeline=pipe).analyze(NOTE) == []


def test_facility_and_employer_labels_map_to_name():
    # Safe Harbor (A) covers employer and facility names, not just the patient's.
    text = "Seen at Lakeside Family Clinic; retired from the fire department."
    pipe = FakePipeline([item("HOSP", 8, 30), item("PATORG", 49, 64)])
    [facility, employer] = NerRecognizer(pipeline=pipe).analyze(text)
    assert facility.type is EntityType.NAME
    assert facility.text == "Lakeside Family Clinic"
    assert employer.type is EntityType.NAME
    assert employer.text == "fire department"


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


class MappingPipeline:
    """Input-aware fake: the recognizer's spillover recheck sends a group's
    later lines back through the pipeline, so the fake must answer per-input
    the way the real model would."""

    def __init__(self, by_text):
        self.by_text = by_text
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return self.by_text.get(text, [])


def test_single_model_group_split_at_newlines():
    # The pipeline itself can aggregate one label run across a paragraph
    # break; the recognizer must split it back into per-line spans (the
    # email-shaped line is dropped by the @ filter, and the spillover line
    # survives because the model still labels it in isolation).
    text = "mail hollis.sablewood@example.org.\n\nQuentin Okafor signed."
    pipe = MappingPipeline(
        {
            text: [item("PATIENT", 5, 50)],
            "Quentin Okafor signed.": [item("PATIENT", 0, 14)],
        }
    )
    [entity] = NerRecognizer(pipeline=pipe).analyze(text)
    assert entity.text == "Quentin Okafor"
    assert text[entity.start : entity.end] == entity.text


def test_spillover_bleed_dropped_when_line_recheck_is_silent():
    # The tokenizer glues "\nMRN:" onto the name's last token; in isolation
    # the model does not label "MRN" a name, so the bled fragment must go.
    text = "Patient: Ann Cole\nMRN: 8842917"
    pipe = MappingPipeline(
        {
            text: [item("PATIENT", 9, 21)],  # 'Ann Cole\nMRN'
            "MRN: 8842917": [item("ID", 5, 12)],  # ID is unmapped anyway
        }
    )
    [entity] = NerRecognizer(pipeline=pipe).analyze(text)
    assert entity.text == "Ann Cole"
    assert "MRN: 8842917" in pipe.calls  # the recheck actually ran


def test_spillover_recheck_uses_the_lines_own_spans():
    # The recheck's isolated-line detection wins over the bled fragment's
    # boundaries: it may extend past the fragment to the full identifier.
    text = "Signed by Rex Ames\nDr. Ada Barr, attending"
    pipe = MappingPipeline(
        {
            text: [item("STAFF", 10, 26)],  # 'Rex Ames\nDr. Ada'
            "Dr. Ada Barr, attending": [item("STAFF", 4, 12)],  # 'Ada Barr'
        }
    )
    entities = NerRecognizer(pipeline=pipe).analyze(text)
    assert [e.text for e in entities] == ["Rex Ames", "Ada Barr"]


def test_middle_initial_joins_name_spans():
    # 'Marcus T.' + 'Whitfield' is one person; the period after a single
    # capital letter is an initial's, not a sentence boundary's.
    text = "Patient: Marcus T. Whitfield presented today."
    pipe = FakePipeline([item("PATIENT", 9, 18), item("PATIENT", 19, 28)])
    [name] = NerRecognizer(pipeline=pipe).analyze(text)
    assert name.text == "Marcus T. Whitfield"
    assert text[name.start : name.end] == name.text


def test_sentence_period_still_separates_names():
    # A period after a full word ends the sentence; the names on either
    # side are different people and must not merge.
    text = "Consulted Rex. Quimby will follow up."
    pipe = FakePipeline([item("PATIENT", 10, 14), item("PATIENT", 15, 21)])
    entities = NerRecognizer(pipeline=pipe).analyze(text)
    assert [e.text for e in entities] == ["Rex", "Quimby"]


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


def test_backend_is_validated():
    with pytest.raises(ValueError):
        NerRecognizer(pipeline=FakePipeline([]), backend="tensorrt")


def test_explicit_pipeline_skips_loading_for_any_backend():
    # No extra installed in CI; an injected pipeline must never trigger a load.
    rec = NerRecognizer(pipeline=FakePipeline([]), backend="onnx")
    assert rec.backend is NerBackend.ONNX
    assert rec.analyze(NOTE) == []


def test_onnx_cache_dir_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRUB_CACHE", str(tmp_path))
    assert onnx_cache_dir("obi/deid_roberta_i2b2") == (
        tmp_path / "onnx" / "obi--deid_roberta_i2b2-int8"
    )
