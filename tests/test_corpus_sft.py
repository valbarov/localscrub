import json

from localscrub.corpus import generate_corpus
from localscrub.corpus.sft import build_sft_examples
from localscrub.stage2 import extraction_prompt

NOTES = generate_corpus(5, seed=9)


def test_one_example_per_note_with_expected_keys():
    examples = build_sft_examples(NOTES)
    assert len(examples) == len(NOTES)
    for example, note in zip(examples, NOTES, strict=True):
        assert set(example) == {"id", "prompt", "completion"}
        assert example["id"] == note.id


def test_prompt_matches_inference_prompt_shape():
    [example] = build_sft_examples(NOTES[:1])
    assert NOTES[0].text in example["prompt"]
    # Same preamble as the inference-time prompt for this document.
    assert example["prompt"].startswith(extraction_prompt("x")[:80])


def test_completion_is_gold_in_reply_format():
    for example, note in zip(build_sft_examples(NOTES), NOTES, strict=True):
        entities = json.loads(example["completion"])["entities"]
        gold = {(s.type.value, s.text) for s in note.spans}
        assert {(e["type"], e["text"]) for e in entities} == gold
        # unique snippets only — repeated mentions collapse to one entry
        assert len(entities) == len({(e["type"], e["text"]) for e in entities})


def test_deterministic():
    assert build_sft_examples(NOTES) == build_sft_examples(NOTES)
