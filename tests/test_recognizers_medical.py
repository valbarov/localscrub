import pytest

from localscrub.entities import EntityType
from localscrub.recognizers.medical import HealthPlanIdRecognizer, MrnRecognizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MRN: 4859302 admitted", "4859302"),
        ("medical record number 88-JK-4421 on", "88-JK-4421"),
        ("Record # A99871 closed", "A99871"),
        ("mrn#00-4417-B", "00-4417-B"),
    ],
)
def test_mrn_labeled_forms(text, expected):
    [entity] = MrnRecognizer().analyze(text)
    assert entity.type is EntityType.MRN
    assert entity.text == expected
    assert entity.confidence == 0.90


def test_unlabeled_number_not_an_mrn():
    assert MrnRecognizer().analyze("value 4859302 alone") == []


def test_mrn_label_without_digits_not_matched():
    assert MrnRecognizer().analyze("the medical record number follows") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("member ID: ZGP-4482017 active", "ZGP-4482017"),
        ("policy number 900-44-XA1", "900-44-XA1"),
        ("Subscriber # 77120-K", "77120-K"),
    ],
)
def test_health_plan_labeled_forms(text, expected):
    [entity] = HealthPlanIdRecognizer().analyze(text)
    assert entity.type is EntityType.HEALTH_PLAN_ID
    assert entity.text == expected
