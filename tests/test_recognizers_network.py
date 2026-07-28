import pytest

from localscrub.entities import EntityType
from localscrub.recognizers.network import IpAddressRecognizer, UrlRecognizer


@pytest.mark.parametrize(
    ("text", "expected", "confidence"),
    [
        ("see https://example.com/page?q=1 there", "https://example.com/page?q=1", 0.90),
        ("plain http://example.org", "http://example.org", 0.90),
        ("at www.example.com/about", "www.example.com/about", 0.75),
    ],
)
def test_url_forms(text, expected, confidence):
    [entity] = UrlRecognizer().analyze(text)
    assert entity.type is EntityType.URL
    assert entity.text == expected
    assert entity.confidence == confidence


def test_bare_domain_not_matched():
    assert UrlRecognizer().analyze("visit example.com sometime") == []


@pytest.mark.parametrize("text", ["from 192.0.2.14 today", "host 203.0.113.9"])
def test_ipv4_detected(text):
    [entity] = IpAddressRecognizer().analyze(text)
    assert entity.type is EntityType.IP_ADDRESS
    assert entity.confidence >= 0.90


def test_ipv6_detected():
    [entity] = IpAddressRecognizer().analyze("addr 2001:db8::8a2e:370:7334 seen")
    assert entity.type is EntityType.IP_ADDRESS


@pytest.mark.parametrize("text", ["bad 999.1.1.1 quad", "time 12:30:45 stamp"])
def test_invalid_addresses_rejected(text):
    assert IpAddressRecognizer().analyze(text) == []


def test_version_string_demoted():
    [entity] = IpAddressRecognizer().analyze("running v1.2.3.4 build")
    assert entity.confidence == pytest.approx(0.70)
