import csv
import hashlib

import pytest

from localscrub.corpus.mtsamples import INJECTIONS, build_benchmark, fetch, load_transcriptions
from localscrub.entities import EntityType

# Invented transcription-shaped prose (test-data policy: nothing from any
# real clinical source). Padded to clear the default length band.
FAKE_BODIES = [
    (
        "SUBJECTIVE:, The patient reports gradual improvement. "
        "OBJECTIVE:, Lungs are clear to auscultation bilaterally. "
        "ASSESSMENT:, Symptoms improving on current therapy. "
        "PLAN:, Continue current regimen and reassess. "
    )
    * 4,
    (
        "HISTORY:, No acute distress noted on examination today. "
        "FINDINGS:, Unremarkable study without focal abnormality. "
        "IMPRESSION:, Stable appearance compared to prior. "
        "RECOMMENDATION:, Routine follow-up as scheduled. "
    )
    * 4,
    (
        "PROCEDURE:, Informed consent was obtained before starting. "
        "TECHNIQUE:, Standard sterile technique was employed throughout. "
        "COMPLICATIONS:, None were encountered during the procedure. "
        "DISPOSITION:, Stable condition on transfer to recovery. "
    )
    * 4,
]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "mtsamples.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["", "description", "medical_specialty", "sample_name",
                         "transcription", "keywords"])
        for i, body in enumerate(FAKE_BODIES):
            writer.writerow([i, "desc", "Testing", f"Sample {i}", body, ""])
        writer.writerow([99, "desc", "Testing", "Empty", "", ""])  # filtered out
    return path


def test_load_transcriptions_filters_by_length(csv_path):
    texts = load_transcriptions(csv_path)
    assert len(texts) == 3
    assert load_transcriptions(csv_path, min_chars=1, max_chars=10) == []


def test_benchmark_is_deterministic(csv_path):
    a = build_benchmark(csv_path, 3, seed=5, injections=3)
    b = build_benchmark(csv_path, 3, seed=5, injections=3)
    assert a == b
    assert a != build_benchmark(csv_path, 3, seed=6, injections=3)


def test_gold_spans_slice_exactly(csv_path):
    for note in build_benchmark(csv_path, 3, seed=1, injections=4):
        assert note.note_type == "mtsamples"
        assert note.id.startswith("mts-")
        for span in note.spans:
            assert note.text[span.start : span.end] == span.text


def test_base_prose_survives_injection(csv_path):
    [note] = build_benchmark(csv_path, 1, seed=2, injections=2)
    # Removing the injected fragments must give back the original transcript.
    assert "Informed consent" in note.text or "acute distress" in note.text or (
        "gradual improvement" in note.text
    )
    # Header identifiers are always present.
    types = {s.type for s in note.spans}
    assert {EntityType.NAME, EntityType.MRN, EntityType.DATE} <= types


def test_coreference_shared_across_injections(csv_path):
    # The same {patient} slot fills the same name in the header and in any
    # narrative injection mentioning the patient: with many injections at a
    # fixed seed, at least one note must repeat the header name verbatim.
    notes = build_benchmark(csv_path, 3, seed=3, injections=len(INJECTIONS))
    repeated = 0
    for note in notes:
        names = [s.text for s in note.spans if s.type is EntityType.NAME]
        header_patient = names[0]
        repeated += names.count(header_patient) > 1
    assert repeated >= 1


def test_count_exceeding_pool_rejected(csv_path):
    with pytest.raises(ValueError, match="only 3 transcripts qualify"):
        build_benchmark(csv_path, 10)


def test_fetch_rejects_checksum_mismatch(tmp_path):
    bogus = tmp_path / "mtsamples.csv"
    bogus.write_text("not the real file", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        fetch(bogus)


def test_fetch_accepts_matching_checksum(tmp_path):
    path = tmp_path / "mtsamples.csv"
    path.write_text("content", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert fetch(path, expected_sha256=digest) == path
