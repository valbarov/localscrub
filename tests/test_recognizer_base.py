import re
from typing import ClassVar

from localscrub.entities import EntityType, Stage
from localscrub.recognizers.base import Pattern, Recognizer, RegexRecognizer


class ToyRecognizer(RegexRecognizer):
    """Matches the literal token XYZ-123 for framework tests."""

    name: ClassVar[str] = "toy"
    entity_type: ClassVar[EntityType] = EntityType.OTHER_UNIQUE_ID
    patterns: ClassVar[tuple[Pattern, ...]] = (
        Pattern(re.compile(r"XYZ-\d{3}"), confidence=0.6),
    )
    context_words: ClassVar[frozenset[str]] = frozenset({"token"})


def test_satisfies_protocol():
    assert isinstance(ToyRecognizer(), Recognizer)


def test_basic_match_span_and_stage():
    text = "id XYZ-123 here"
    [entity] = ToyRecognizer().analyze(text)
    assert (entity.start, entity.end) == (3, 10)
    assert entity.text == "XYZ-123"
    assert entity.stage is Stage.RULES
    assert entity.confidence == 0.6


def test_context_word_boosts_confidence():
    [entity] = ToyRecognizer().analyze("token: XYZ-123")
    assert entity.confidence == 0.8


def test_confidence_capped_at_one():
    class HighToy(ToyRecognizer):
        patterns: ClassVar[tuple[Pattern, ...]] = (
            Pattern(re.compile(r"XYZ-\d{3}"), confidence=0.95),
        )

    [entity] = HighToy().analyze("token: XYZ-123")
    assert entity.confidence == 1.0


def test_validate_none_rejects_match():
    class PickyToy(ToyRecognizer):
        def validate(self, match):
            return None if match.group().endswith("000") else 0.0

    entities = PickyToy().analyze("XYZ-000 and XYZ-001")
    assert [e.text for e in entities] == ["XYZ-001"]


def test_validate_delta_adjusts_confidence():
    class BonusToy(ToyRecognizer):
        def validate(self, match):
            return 0.1

    [entity] = BonusToy().analyze("XYZ-123")
    assert entity.confidence == 0.7


def test_group_selects_entity_span():
    class GroupToy(ToyRecognizer):
        patterns: ClassVar[tuple[Pattern, ...]] = (
            Pattern(re.compile(r"id:\s*(?P<v>XYZ-\d{3})"), confidence=0.6, group="v"),
        )

    [entity] = GroupToy().analyze("id: XYZ-123")
    assert entity.text == "XYZ-123"
    assert entity.start == 4
