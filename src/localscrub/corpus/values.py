"""Fabricated identifier values for the synthetic corpus.

Value policy
------------
Every value is synthetic, and drawn from a reserved range wherever one exists
(the same policy as ``tests/conftest.py``):

- phone/fax numbers: NANP 555-01XX (reserved fictional range)
- emails/URLs: example.com / example.org / example.net (RFC 2606)
- IP addresses: 192.0.2.x, 198.51.100.x, 203.0.113.x (RFC 5737 TEST-NET)
- card numbers: random Luhn-valid numbers on major-industry test prefixes
- SSNs: structurally valid random digit strings tied to no person
- names: random pairings from small built-in lists — combinations, not records
- streets/cities: invented; MRNs/accounts/plan IDs: random digits

No value is copied from any real record, clinical source, or person.
"""

from __future__ import annotations

import random
from datetime import date

from localscrub.recognizers.financial import luhn_ok

_FIRST_NAMES = (
    "Alma", "Bennett", "Camila", "Darius", "Elena", "Farid", "Greta", "Hollis",
    "Imani", "Jonas", "Katya", "Lionel", "Marisol", "Nadia", "Otto", "Priya",
    "Quentin", "Rosalind", "Silas", "Tamsin", "Ulysses", "Vera", "Wendell", "Yusuf",
)
_LAST_NAMES = (
    "Ashcombe", "Bellweather", "Cardenas", "Dremel", "Eastvale", "Fenwick",
    "Garrows", "Halloway", "Ivers", "Jastrow", "Kellerman", "Lindqvist",
    "Marchetti", "Norwood", "Okafor", "Pemberton", "Quimby", "Rasmussen",
    "Sablewood", "Tarrant", "Underhill", "Vestergaard", "Whitlock", "Yarrow",
)
_STREET_NAMES = (
    "Alder", "Birchwood", "Copperfield", "Dunmore", "Elmhurst", "Foxglove",
    "Grangemill", "Harrowgate", "Juniper", "Kestrel", "Larkspur", "Mossbank",
)
_STREET_SUFFIXES = ("St", "Ave", "Rd", "Ln", "Blvd", "Ct")
_CITIES = (
    "Fairmont Hollow", "Cedar Vale", "Marrow Springs", "Quarry Bend",
    "Lantern Hill", "Westhaven", "Bramblewood", "Sorrel Creek",
)
_STATES = ("OH", "PA", "MN", "CO", "NC", "OR", "VT", "NM")
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
_AREA_CODES = ("212", "303", "412", "503", "614", "704", "802", "919")
_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
_TEST_NETS = ("192.0.2", "198.51.100", "203.0.113")
_URL_HOSTS = ("portal.example.org", "www.example.com", "results.example.net")
_URL_PATHS = ("visit-summary", "lab-results", "patient/records", "billing/statement")
_CARD_PREFIXES = ("4", "51", "55", "6011")


class Fabricator:
    """Draws synthetic identifier values from a shared seeded ``random.Random``."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def name(self) -> str:
        return f"{self.rng.choice(_FIRST_NAMES)} {self.rng.choice(_LAST_NAMES)}"

    def street_address(self) -> str:
        number = self.rng.randint(10, 9899)
        street = self.rng.choice(_STREET_NAMES)
        suffix = self.rng.choice(_STREET_SUFFIXES)
        return f"{number} {street} {suffix}"

    def city_state_zip(self) -> str:
        city = self.rng.choice(_CITIES)
        state = self.rng.choice(_STATES)
        return f"{city}, {state} {self.rng.randint(10000, 99499)}"

    def date(self) -> str:
        d = date(self.rng.randint(2019, 2025), self.rng.randint(1, 12), self.rng.randint(1, 28))
        style = self.rng.randrange(4)
        if style == 0:
            return d.isoformat()
        if style == 1:
            return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"
        if style == 2:
            return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"
        return f"{d.month}/{d.day}/{d.year}"

    def age_over_89(self) -> str:
        age = self.rng.randint(90, 103)
        style = self.rng.randrange(3)
        if style == 0:
            return f"{age}-year-old"
        if style == 1:
            return f"{age} yo"
        return f"aged {age}"

    def phone(self) -> str:
        area = self.rng.choice(_AREA_CODES)
        line = f"01{self.rng.randint(0, 99):02d}"
        style = self.rng.randrange(3)
        if style == 0:
            return f"({area}) 555-{line}"
        if style == 1:
            return f"{area}-555-{line}"
        return f"+1 {area} 555 {line}"

    def email(self, name: str | None = None) -> str:
        if name is None:
            name = self.name()
        first, last = name.lower().split(" ", 1)
        local = f"{first}.{last.replace(' ', '')}"
        return f"{local}@{self.rng.choice(_EMAIL_DOMAINS)}"

    def ssn(self) -> str:
        area = self.rng.randint(1, 899)
        if area == 666:
            area = 667
        group = self.rng.randint(1, 99)
        serial = self.rng.randint(1, 9999)
        return f"{area:03d}-{group:02d}-{serial:04d}"

    def mrn(self) -> str:
        return str(self.rng.randint(1_000_000, 99_999_999))

    def health_plan_id(self) -> str:
        letters = "".join(self.rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
        return f"{letters}{self.rng.randint(10_000_000, 99_999_999)}"

    def account_number(self) -> str:
        return str(self.rng.randint(10_000_000, 9_999_999_999))

    def url(self) -> str:
        return f"https://{self.rng.choice(_URL_HOSTS)}/{self.rng.choice(_URL_PATHS)}"

    def ip_address(self) -> str:
        return f"{self.rng.choice(_TEST_NETS)}.{self.rng.randint(1, 254)}"

    def credit_card(self) -> str:
        prefix = self.rng.choice(_CARD_PREFIXES)
        body = prefix + "".join(str(self.rng.randrange(10)) for _ in range(15 - len(prefix)))
        check = next(d for d in "0123456789" if luhn_ok(body + d))
        digits = body + check
        if self.rng.randrange(2):
            return " ".join(digits[i : i + 4] for i in range(0, 16, 4))
        return digits
