import json

import pytest

from localscrub.analyzer import apply_stage2, scrub
from localscrub.entities import Entity, EntityType, Stage
from localscrub.evaluation import evaluate
from localscrub.llm import OllamaError
from localscrub.stage2 import STAGE2_TYPES, LlmExtractor

NOTE = (
    "Pt Alma Quimby seen by Dr. Otto Fenwick. Alma Quimby lives at "
    "12 Alder St, Cedar Vale, OH 44120. Contact test@example.com."
)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


def reply_with(*entities):
    return json.dumps({"entities": [{"type": t, "text": s} for t, s in entities]})


def test_extractor_locates_all_occurrences():
    client = FakeClient(reply_with(("NAME", "Alma Quimby")))
    [first, second] = LlmExtractor(client).extract(NOTE)
    assert (first.start, first.text) == (3, "Alma Quimby")
    assert NOTE[second.start : second.end] == "Alma Quimby"
    assert first.stage is Stage.LLM


def test_extractor_prefers_longer_overlapping_snippet():
    client = FakeClient(reply_with(("NAME", "Otto Fenwick"), ("NAME", "Fenwick")))
    entities = LlmExtractor(client).extract(NOTE)
    assert [e.text for e in entities] == ["Otto Fenwick"]


def test_extractor_skips_absent_and_invalid_items():
    client = FakeClient(
        reply_with(
            ("NAME", "Nobody Here"),  # not in the text
            ("FULL_FACE_PHOTO", "Alma"),  # type not allowed in text
            ("BANANA", "Alma"),  # not a type at all
            ("NAME", ""),  # empty snippet
            ("GEO_SUBDIVISION", "Cedar Vale, OH 44120"),
        )
    )
    [entity] = LlmExtractor(client).extract(NOTE)
    assert entity.type is EntityType.GEO_SUBDIVISION


def test_extractor_raises_on_malformed_reply():
    with pytest.raises(OllamaError, match="schema-shaped"):
        LlmExtractor(FakeClient("not json")).extract(NOTE)


def test_truncated_reply_salvages_complete_items():
    # num_predict can cut a long extraction list mid-object; every complete
    # item before the cut must survive.
    truncated = (
        '{\n  "entities": [\n'
        '    {"type": "NAME", "text": "Alma Quimby"},\n'
        '    {"type": "NAME", "text": "Otto Fenwick"},\n'
        '    {"type": "GEO_SUBDIVISION", "text": "Cedar Va'
    )
    entities = LlmExtractor(FakeClient(truncated)).extract(NOTE)
    assert sorted({e.text for e in entities}) == ["Alma Quimby", "Otto Fenwick"]


def test_truncated_reply_with_no_complete_item_raises():
    with pytest.raises(OllamaError, match="schema-shaped"):
        LlmExtractor(FakeClient('{"entities": [{"type": "NAME", "te')).extract(NOTE)


def test_full_face_photo_excluded_from_stage2_types():
    assert EntityType.FULL_FACE_PHOTO not in STAGE2_TYPES


def test_prompt_includes_flagged_spans():
    client = FakeClient(reply_with())
    result = scrub("Reach me at 555-0142 sometime.")  # 7-digit local: escalates
    assert result.escalations
    LlmExtractor(client).extract(result.text, result.escalations)
    assert "555-0142" in client.prompts[0]
    assert "low_confidence" in client.prompts[0]


def test_scrub_cascade_adds_and_redacts_llm_entities():
    client = FakeClient(reply_with(("NAME", "Alma Quimby"), ("NAME", "Otto Fenwick")))
    result = scrub(NOTE, stage2=LlmExtractor(client))
    assert "Alma Quimby" not in result.redacted
    assert "Otto Fenwick" not in result.redacted
    assert "[EMAIL_1]" in result.redacted  # stage-1 detection intact
    stages = {e.stage for e in result.entities}
    assert stages == {Stage.RULES, Stage.LLM}


def test_stage1_wins_overlaps():
    client = FakeClient(reply_with(("OTHER_UNIQUE_ID", "test@example.com")))
    result = scrub(NOTE, stage2=LlmExtractor(client))
    [email] = [e for e in result.entities if e.text == "test@example.com"]
    assert email.type is EntityType.EMAIL
    assert email.stage is Stage.RULES


def test_confirmed_escalation_is_resolved():
    text = "Reach me at 555-0142 sometime."
    client = FakeClient(reply_with(("PHONE", "555-0142")))
    result = scrub(text, stage2=LlmExtractor(client))
    assert result.escalations == ()
    assert "555-0142" not in result.redacted


def test_unconfirmed_escalation_stays_escalated():
    text = "Reach me at 555-0142 sometime."
    client = FakeClient(reply_with())
    before = scrub(text)
    after = scrub(text, stage2=LlmExtractor(client))
    assert after.escalations == before.escalations
    assert "555-0142" not in after.redacted  # fail-closed redaction kept


def test_apply_stage2_no_extractions_returns_input():
    result = scrub(NOTE)
    assert apply_stage2(result, LlmExtractor(FakeClient(reply_with()))) is result


def test_evaluate_with_stage2_lifts_name_recall():
    # A perfect-oracle stage 2: echo the gold NAME/GEO spans back.
    from localscrub.corpus import generate_corpus

    notes = generate_corpus(10, seed=21)
    gold = {
        note.text: [
            (s.type, s.text)
            for s in note.spans
            if s.type in (EntityType.NAME, EntityType.GEO_SUBDIVISION)
        ]
        for note in notes
    }

    class Oracle:
        confidence = 0.85

        def extract(self, text, escalations=()):
            entities = []
            for etype, snippet in gold[text]:
                start = 0
                while (start := text.find(snippet, start)) != -1:
                    entities.append(
                        Entity(etype, start, start + len(snippet), snippet, 0.85, Stage.LLM)
                    )
                    start += len(snippet)
            deduped = []
            for e in sorted(entities, key=lambda e: (e.start - e.end, e.start)):
                if not any(e.start < a.end and a.start < e.end for a in deduped):
                    deduped.append(e)
            return tuple(sorted(deduped, key=lambda e: e.start))

    baseline = evaluate(notes)
    cascade = evaluate(notes, stage2=Oracle())
    assert baseline.relaxed.get(EntityType.NAME) is None or (
        baseline.relaxed[EntityType.NAME].recall == 0.0
    )
    assert cascade.relaxed[EntityType.NAME].recall == 1.0
    assert cascade.redaction_recall > baseline.redaction_recall


def test_evaluate_survives_llm_failure_and_counts_it():
    # An LLM that dies on every request must cost the cascade its stage-2
    # lift, never the whole run.
    from localscrub.corpus import generate_corpus

    class DeadLlm:
        confidence = 0.85

        def extract(self, text, escalations=()):
            raise OllamaError("boom")

    notes = generate_corpus(3, seed=21)
    with pytest.warns(RuntimeWarning, match="scored without it"):
        report = evaluate(notes, stage2=DeadLlm())
    assert report.llm_failures == 3
    assert report.notes == 3
    assert report.to_dict() == evaluate(notes).to_dict() | {"llm_failures": 3}
    assert "scored without stage 2" in report.format_table()


def test_evaluate_retries_transient_llm_failure():
    from localscrub.corpus import generate_corpus

    class FlakyLlm:
        confidence = 0.85

        def __init__(self):
            self.calls = 0

        def extract(self, text, escalations=()):
            self.calls += 1
            if self.calls == 1:
                raise OllamaError("transient")
            return ()

    report = evaluate(generate_corpus(3, seed=21), stage2=FlakyLlm())
    assert report.llm_failures == 0
