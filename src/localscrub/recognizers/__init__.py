"""Stage-1 recognizers: regex + dictionary + checksum detection.

NAME and GEO_SUBDIVISION are not regex-able; install the ``[ner]`` extra and
add ``NerRecognizer`` (or use stage 2) — see ROADMAP.md.
"""

from localscrub.recognizers.base import Pattern, Recognizer, RegexRecognizer
from localscrub.recognizers.contact import EmailRecognizer, PhoneFaxRecognizer
from localscrub.recognizers.dates import AgeOver89Recognizer, DateRecognizer
from localscrub.recognizers.financial import AccountNumberRecognizer, CreditCardRecognizer
from localscrub.recognizers.government import SsnRecognizer
from localscrub.recognizers.medical import HealthPlanIdRecognizer, MrnRecognizer
from localscrub.recognizers.ner import NerRecognizer
from localscrub.recognizers.network import IpAddressRecognizer, UrlRecognizer

__all__ = [
    "AccountNumberRecognizer",
    "AgeOver89Recognizer",
    "CreditCardRecognizer",
    "DateRecognizer",
    "EmailRecognizer",
    "HealthPlanIdRecognizer",
    "IpAddressRecognizer",
    "MrnRecognizer",
    "NerRecognizer",
    "Pattern",
    "PhoneFaxRecognizer",
    "Recognizer",
    "RegexRecognizer",
    "SsnRecognizer",
    "UrlRecognizer",
    "default_recognizers",
]


def default_recognizers() -> tuple[Recognizer, ...]:
    """The stage-1 recognizer set, one instance each (NER extra not included)."""
    return (
        EmailRecognizer(),
        PhoneFaxRecognizer(),
        UrlRecognizer(),
        IpAddressRecognizer(),
        CreditCardRecognizer(),
        AccountNumberRecognizer(),
        SsnRecognizer(),
        MrnRecognizer(),
        HealthPlanIdRecognizer(),
        DateRecognizer(),
        AgeOver89Recognizer(),
    )
