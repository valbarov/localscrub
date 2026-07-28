import pytest

from localscrub.entities import EntityType
from localscrub.recognizers.government import SsnRecognizer


def test_dashed_ssn_detected():
    [entity] = SsnRecognizer().analyze("patient ssn 123-45-6789 on file")
    assert entity.type is EntityType.SSN
    assert entity.text == "123-45-6789"
    assert entity.confidence == pytest.approx(1.0)  # 0.85 + context boost, capped


def test_dashed_ssn_without_context_still_strong():
    [entity] = SsnRecognizer().analyze("value 123-45-6789 noted")
    assert entity.confidence == pytest.approx(0.85)


@pytest.mark.parametrize(
    "text",
    [
        "000-45-6789",  # area 000
        "666-45-6789",  # area 666
        "923-45-6789",  # area 9xx
        "123-00-6789",  # group 00
        "123-45-0000",  # serial 0000
    ],
)
def test_structurally_invalid_ssn_rejected(text):
    assert SsnRecognizer().analyze(f"ssn {text} here") == []


def test_bare_nine_digits_is_noise_floor():
    [entity] = SsnRecognizer().analyze("ref 123456789 end")
    assert entity.confidence == pytest.approx(0.30)


def test_bare_nine_digits_with_context_reaches_escalation_band():
    [entity] = SsnRecognizer().analyze("SSN: 123456789")
    assert entity.confidence == pytest.approx(0.50)


def test_ten_digit_run_not_matched():
    assert SsnRecognizer().analyze("num 1234567890 end") == []
