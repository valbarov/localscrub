import pytest

from localscrub.entities import EntityType
from localscrub.recognizers.contact import EmailRecognizer, PhoneFaxRecognizer


@pytest.mark.parametrize(
    "text",
    [
        "write to jane.doe@example.com today",
        "cc: j+tag@sub.example.org",
        "J_D%test@example-mail.example.com",
    ],
)
def test_email_detected(text):
    [entity] = EmailRecognizer().analyze(text)
    assert entity.type is EntityType.EMAIL
    assert entity.confidence == 0.95
    assert "@" in entity.text


@pytest.mark.parametrize("text", ["no at sign here", "half@nodot", "just @ alone"])
def test_email_negatives(text):
    assert EmailRecognizer().analyze(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 555-015-0123 now", "555-015-0123"),
        ("(555) 015-0123", "(555) 015-0123"),
        ("+1 555 015 0123", "+1 555 015 0123"),
        ("555.015.0123", "555.015.0123"),
    ],
)
def test_phone_ten_digit_forms(text, expected):
    entities = PhoneFaxRecognizer().analyze(text)
    assert expected in [e.text for e in entities]
    assert all(e.type is EntityType.PHONE for e in entities)


def test_phone_context_boost():
    entities = PhoneFaxRecognizer().analyze("phone: (555) 015-0123")
    [entity] = [e for e in entities if e.text == "(555) 015-0123"]
    assert entity.confidence == pytest.approx(0.9)


def test_fax_context_switches_type():
    entities = PhoneFaxRecognizer().analyze("fax: (555) 015-0123")
    assert entities and all(e.type is EntityType.FAX for e in entities)


def test_seven_digit_local_is_escalation_band():
    entities = PhoneFaxRecognizer().analyze("try 015-0123")
    [entity] = [e for e in entities if e.text == "015-0123"]
    assert entity.confidence == 0.5


def test_ssn_shape_not_matched_as_phone():
    # 3-2-4 grouping is an SSN, not a NANP number
    assert all(
        e.text != "123-45-6789" for e in PhoneFaxRecognizer().analyze("ssn 123-45-6789")
    )
