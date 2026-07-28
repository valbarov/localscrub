import pytest

from localscrub.entities import EntityType
from localscrub.recognizers.financial import (
    AccountNumberRecognizer,
    CreditCardRecognizer,
    luhn_ok,
)


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1111 on file",
        "pay with 4111-1111-1111-1111",
        "amex 371449635398431 works",
    ],
)
def test_test_cards_detected(text):
    [entity] = CreditCardRecognizer().analyze(text)
    assert entity.type is EntityType.CREDIT_CARD
    # Luhn-valid + known IIN + context word in two cases; always above accept
    assert entity.confidence >= 0.95


def test_luhn_failure_rejects():
    assert CreditCardRecognizer().analyze("num 1234 5678 9012 3456") == []


def test_ten_digit_phone_not_a_card():
    assert CreditCardRecognizer().analyze("5550150123") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("account number: 8834-2291-X7 open", "8834-2291-X7"),
        ("acct #99120044", "99120044"),
        ("Account No. 55321-A88", "55321-A88"),
    ],
)
def test_labeled_account_number(text, expected):
    [entity] = AccountNumberRecognizer().analyze(text)
    assert entity.type is EntityType.ACCOUNT_NUMBER
    assert entity.text == expected


def test_account_prose_not_matched():
    assert AccountNumberRecognizer().analyze("the account balance is high") == []
