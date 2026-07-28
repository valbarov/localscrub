import json

import pytest

from localscrub.analyzer import scrub
from localscrub.corpus import (
    NOTE_TYPES,
    GoldSpan,
    SyntheticNote,
    generate_corpus,
    read_jsonl,
    write_jsonl,
)
from localscrub.corpus.templates import ENTITY_SLOTS
from localscrub.entities import EntityType
from localscrub.recognizers.financial import luhn_ok

CORPUS = generate_corpus(200, seed=7)


def test_same_seed_same_corpus():
    assert generate_corpus(20, seed=7) == generate_corpus(20, seed=7)


def test_different_seed_different_corpus():
    texts_a = [n.text for n in generate_corpus(5, seed=1)]
    texts_b = [n.text for n in generate_corpus(5, seed=2)]
    assert texts_a != texts_b


def test_gold_spans_slice_exactly():
    for note in CORPUS:
        for span in note.spans:
            assert note.text[span.start : span.end] == span.text


def test_gold_spans_ordered_and_disjoint():
    for note in CORPUS:
        for prev, cur in zip(note.spans, note.spans[1:], strict=False):
            assert prev.end <= cur.start


def test_every_planted_type_appears():
    seen = {span.type for note in CORPUS for span in note.spans}
    assert seen == set(ENTITY_SLOTS.values())


def test_every_note_type_appears():
    assert {n.note_type for n in CORPUS} == set(NOTE_TYPES)


@pytest.mark.parametrize(
    ("etype", "check"),
    [
        # reserved fictional NANP range
        (EntityType.PHONE, lambda v: "555-01" in v or "555 01" in v),
        (EntityType.FAX, lambda v: "555-01" in v or "555 01" in v),
        # RFC 2606 domains only
        (EntityType.EMAIL, lambda v: v.rsplit("@", 1)[1].startswith("example.")),
        (EntityType.URL, lambda v: ".example." in v or "example.com" in v),
        # RFC 5737 TEST-NET blocks
        (
            EntityType.IP_ADDRESS,
            lambda v: v.startswith(("192.0.2.", "198.51.100.", "203.0.113.")),
        ),
        # industry prefixes, Luhn-valid
        (EntityType.CREDIT_CARD, lambda v: luhn_ok(v.replace(" ", ""))),
    ],
)
def test_values_stay_in_reserved_ranges(etype, check):
    values = [s.text for n in CORPUS for s in n.spans if s.type is etype]
    assert values
    assert all(check(v) for v in values)


def test_ssn_values_structurally_valid():
    for note in CORPUS:
        for span in note.spans:
            if span.type is EntityType.SSN:
                area, group, serial = span.text.split("-")
                assert area not in ("000", "666") and not area.startswith("9")
                assert group != "00"
                assert serial != "0000"


def test_jsonl_round_trip(tmp_path):
    notes = generate_corpus(10, seed=3)
    path = tmp_path / "corpus.jsonl"
    write_jsonl(notes, path)
    assert read_jsonl(path) == notes


def test_note_type_filter():
    notes = generate_corpus(30, seed=0, note_types=["billing_note"])
    assert {n.note_type for n in notes} == {"billing_note"}


def test_unknown_note_type_rejected():
    with pytest.raises(ValueError, match="unknown note type"):
        generate_corpus(1, note_types=["radiology"])


def test_gold_span_invariants():
    with pytest.raises(ValueError):
        GoldSpan(EntityType.NAME, 5, 5, "")
    with pytest.raises(ValueError):
        GoldSpan(EntityType.NAME, 0, 3, "toolong")


def test_stage1_redacts_planted_structured_identifiers():
    # High-signal, format-distinctive types stage 1 must catch in full.
    strong = {
        EntityType.EMAIL,
        EntityType.SSN,
        EntityType.IP_ADDRESS,
        EntityType.CREDIT_CARD,
    }
    for note in generate_corpus(30, seed=11):
        result = scrub(note.text)
        for span in note.spans:
            if span.type in strong:
                assert span.text not in result.redacted


def test_from_dict_round_trip():
    note = CORPUS[0]
    assert SyntheticNote.from_dict(json.loads(json.dumps(note.to_dict()))) == note
