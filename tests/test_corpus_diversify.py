import pytest

from localscrub.corpus import generate_corpus
from localscrub.corpus.diversify import _clean, _mask, _rebuild, diversify_corpus

NOTES = generate_corpus(3, seed=5)


def note_body(prompt):
    # The masked note is everything after the prompt's final "Note:" line.
    return prompt.split("Note:\n", 1)[1]


def test_mask_replaces_every_span_with_sentinel():
    note = NOTES[0]
    masked = _mask(note)
    for i, span in enumerate(note.spans):
        assert f"[[E{i}]]" in masked
        assert span.text not in masked


def test_accepted_rewrite_has_exact_spans():
    # A "model" that reorders prose but keeps all sentinels once each.
    def fake_generate(prompt, seed):
        return "Rewritten preamble.\n" + note_body(prompt)

    out, fallbacks = diversify_corpus(NOTES, generate=fake_generate)
    assert fallbacks == 0
    for original, rewritten in zip(NOTES, out, strict=True):
        assert rewritten.text != original.text
        assert len(rewritten.spans) == len(original.spans)
        for span in rewritten.spans:
            assert rewritten.text[span.start : span.end] == span.text
        assert sorted(s.text for s in rewritten.spans) == sorted(
            s.text for s in original.spans
        )


def test_dropped_sentinel_falls_back():
    def fake_generate(prompt, seed):
        return note_body(prompt).replace("[[E0]]", "", 1)

    out, fallbacks = diversify_corpus(NOTES, generate=fake_generate)
    assert fallbacks == len(NOTES)
    assert out == list(NOTES)


def test_duplicated_sentinel_falls_back():
    def fake_generate(prompt, seed):
        return note_body(prompt) + "\nAlso [[E0]] again."

    _, fallbacks = diversify_corpus(NOTES, generate=fake_generate)
    assert fallbacks == len(NOTES)


def test_invented_identifier_falls_back():
    # Sentinels intact, but the model leaked identifier-shaped prose.
    def fake_generate(prompt, seed):
        return note_body(prompt) + "\nAlternate contact: leaked.address@example.com"

    out, fallbacks = diversify_corpus(NOTES, generate=fake_generate)
    assert fallbacks == len(NOTES)
    assert out == list(NOTES)


def test_clean_strips_think_blocks_and_fences():
    assert _clean("<think>hmm\nplan</think>\nText here") == "Text here"
    assert _clean("```\nText here\n```") == "Text here"


def test_rebuild_rejects_unknown_sentinel():
    note = NOTES[0]
    assert _rebuild(_mask(note) + " [[E99]]", note.spans) is None


@pytest.mark.parametrize("permute", [False, True])
def test_rebuild_allows_reordered_sentinels(permute):
    note = NOTES[0]
    masked = _mask(note)
    if permute:
        masked = "[[E1]] then [[E0]] " + "".join(
            f"[[E{i}]] " for i in range(2, len(note.spans))
        )
    rebuilt = _rebuild(masked, note.spans)
    assert rebuilt is not None
    text, spans = rebuilt
    for span in spans:
        assert text[span.start : span.end] == span.text
