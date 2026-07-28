import pytest

from localscrub.entities import EntityType
from localscrub.recognizers.dates import AgeOver89Recognizer, DateRecognizer


@pytest.mark.parametrize(
    ("text", "expected", "confidence"),
    [
        ("logged 2024-03-15 morning", "2024-03-15", 0.90),
        ("on March 15, 2024 the", "March 15, 2024", 0.90),
        ("seen Mar. 3 2021 at", "Mar. 3 2021", 0.90),
        ("dated 15 March 2024 by", "15 March 2024", 0.90),
        ("form 3/15/2024 filed", "3/15/2024", 0.70),
        ("eu style 15-3-24 ok", "15-3-24", 0.70),
    ],
)
def test_date_forms(text, expected, confidence):
    entities = DateRecognizer().analyze(text)
    [entity] = [e for e in entities if e.text == expected]
    assert entity.type is EntityType.DATE
    assert entity.confidence == pytest.approx(confidence)


def test_context_boost_lifts_numeric_date():
    [entity] = DateRecognizer().analyze("DOB: 3/15/1990")
    assert entity.confidence == pytest.approx(0.90)


@pytest.mark.parametrize(
    "text",
    [
        "bad 2024-02-30 date",  # not a calendar date
        "bp 120/80 reading",  # two components only
        "in 2024 overall",  # bare year is not PHI
        "ratio 25/25/25 mix",  # no valid month reading
    ],
)
def test_date_negatives(text):
    assert DateRecognizer().analyze(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a 92-year-old patient", "92-year-old"),
        ("she is 94 years old today", "94 years old"),
        ("male, 91 y/o, presents", "91 y/o"),
        ("age: 103 noted", "age: 103"),
    ],
)
def test_ages_over_89_detected(text, expected):
    [entity] = AgeOver89Recognizer().analyze(text)
    assert entity.type is EntityType.DATE
    assert entity.text == expected


@pytest.mark.parametrize("text", ["a 45-year-old patient", "age: 89 noted", "aged 12"])
def test_ages_89_and_under_not_phi(text):
    assert AgeOver89Recognizer().analyze(text) == []
