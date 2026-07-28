import re
from typing import ClassVar

from localscrub.analyzer import Analyzer, scrub
from localscrub.config import ScrubConfig
from localscrub.entities import EntityType
from localscrub.recognizers.base import Pattern, RegexRecognizer


def make_recognizer(etype, regex, conf, rec_name="toy"):
    # Distinct parameter names: a name assigned in a class body cannot also be
    # read from the enclosing function scope.
    class Toy(RegexRecognizer):
        name: ClassVar[str] = rec_name
        entity_type: ClassVar[EntityType] = etype
        patterns: ClassVar[tuple[Pattern, ...]] = (Pattern(re.compile(regex), confidence=conf),)

    return Toy()


def test_high_confidence_accepted_and_redacted():
    analyzer = Analyzer([make_recognizer(EntityType.SSN, r"AAA", 0.9)])
    result = analyzer.analyze("x AAA y")
    assert [e.type for e in result.entities] == [EntityType.SSN]
    assert result.redacted == "x [SSN_1] y"
    assert not result.escalated


def test_below_floor_dropped_silently():
    analyzer = Analyzer([make_recognizer(EntityType.SSN, r"AAA", 0.3)])
    result = analyzer.analyze("x AAA y")
    assert result.entities == ()
    assert result.escalations == ()
    assert result.redacted == "x AAA y"


def test_escalation_band_redacts_and_escalates():
    analyzer = Analyzer([make_recognizer(EntityType.PHONE, r"AAA", 0.6)])
    result = analyzer.analyze("x AAA y")
    assert result.redacted == "x [PHONE_1] y"
    assert result.escalated
    [esc] = result.escalations
    assert esc.reason == "low_confidence"
    assert esc.text == "AAA"


def test_redact_escalated_false_leaves_band_unredacted():
    analyzer = Analyzer(
        [make_recognizer(EntityType.PHONE, r"AAA", 0.6)],
        config=ScrubConfig(redact_escalated=False),
    )
    result = analyzer.analyze("x AAA y")
    assert result.redacted == "x AAA y"
    assert result.entities == ()
    assert [esc.reason for esc in result.escalations] == ["low_confidence"]


def test_overlap_higher_confidence_wins():
    analyzer = Analyzer(
        [
            make_recognizer(EntityType.URL, r"AAABBB", 0.9, "long"),
            make_recognizer(EntityType.EMAIL, r"BBB", 0.5, "short"),
        ]
    )
    result = analyzer.analyze("x AAABBB y")
    assert [e.type for e in result.entities] == [EntityType.URL]
    # margin 0.4 > ambiguity_margin: loser is discarded, not escalated
    assert result.escalations == ()


def test_overlap_close_call_escalates_ambiguous_type():
    analyzer = Analyzer(
        [
            make_recognizer(EntityType.ACCOUNT_NUMBER, r"AAABBB", 0.9, "acct"),
            make_recognizer(EntityType.PHONE, r"BBB", 0.85, "phone"),
        ]
    )
    result = analyzer.analyze("x AAABBB y")
    assert [e.type for e in result.entities] == [EntityType.ACCOUNT_NUMBER]
    [esc] = result.escalations
    assert esc.reason == "ambiguous_type"
    assert [c.type for c in esc.candidates] == [EntityType.ACCOUNT_NUMBER, EntityType.PHONE]


def test_same_type_overlap_resolved_silently():
    analyzer = Analyzer(
        [
            make_recognizer(EntityType.URL, r"AAABBB", 0.9, "long"),
            make_recognizer(EntityType.URL, r"BBB", 0.85, "short"),
        ]
    )
    result = analyzer.analyze("x AAABBB y")
    assert len(result.entities) == 1
    assert result.escalations == ()


def test_entities_ordered_by_position():
    analyzer = Analyzer(
        [
            make_recognizer(EntityType.SSN, r"AAA", 0.85, "a"),
            make_recognizer(EntityType.EMAIL, r"BBB", 0.95, "b"),
        ]
    )
    result = analyzer.analyze("AAA then BBB")
    assert [e.text for e in result.entities] == ["AAA", "BBB"]


def test_scrub_end_to_end_synthetic_note():
    note = (
        "Pt (MRN: 4859302) is a 92-year-old admitted 2024-03-15. "
        "Contact: jane.doe@example.com, phone (555) 015-0123, SSN 123-45-6789. "
        "Sent from 192.0.2.14."
    )
    result = scrub(note)
    types = {e.type for e in result.entities}
    assert {
        EntityType.MRN,
        EntityType.DATE,
        EntityType.EMAIL,
        EntityType.PHONE,
        EntityType.SSN,
        EntityType.IP_ADDRESS,
    } <= types
    for entity in result.entities:
        assert entity.text not in result.redacted
