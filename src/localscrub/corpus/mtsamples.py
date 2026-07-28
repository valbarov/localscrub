"""MTSamples-injection benchmark (ROADMAP M2).

Public medical-transcription samples (mtsamples.com) provide authentic
clinical prose; fully synthetic identifiers are injected at known offsets,
giving perfect ground truth on real-world text. Unlike the template corpus,
injected identifiers sit inside *narrative sentences with no labels* — this
is the harder test the template numbers cannot provide. The transcripts also
supply natural hard negatives for free: ages, bare years, drug doses.

Data handling: the CSV is downloaded from a pinned mirror into
``data/mtsamples/`` (gitignored) and verified by SHA-256; it is never
redistributed with localscrub. The benchmark itself is a pure function of
``(csv, count, seed, injections)`` — regenerate, don't archive.

Injection is at sentence boundaries (a naive splitter; an abbreviation like
"Dr." occasionally yields an awkward insertion point, which only makes the
prose harder — ground truth is exact either way).
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import urllib.request
from pathlib import Path

from localscrub.corpus.generator import GoldSpan, SyntheticNote, render_slots
from localscrub.corpus.values import Fabricator

MTSAMPLES_URL = (
    "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
)
MTSAMPLES_SHA256 = "a92cb78569f460606661ee7da50c92305b60d0fac2be4b06c8d2de0f234612d4"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

#: narrative injections — identifiers embedded in prose, no labels
INJECTIONS: tuple[str, ...] = (
    "{patient} tolerated the procedure well.",
    "{patient} verbalized understanding of the plan.",
    "Dr. {provider} was consulted during the encounter.",
    "The patient's daughter, {relative}, was present for the discussion.",
    "The patient may be reached at {phone} with any questions.",
    "Records were forwarded to {email} per the patient's request.",
    "The patient resides at {address}.",
    "Follow-up is scheduled for {followup_date}.",
    "Prior imaging from {visit_date} was reviewed and compared.",
    "The full report was uploaded to {url}.",
    "Identity was confirmed against SSN {ssn}.",
    "Coverage was verified under member ID {plan_id}.",
)

#: a short demographics header, as real transcriptions carry
_HEADER = "Patient: {patient}\nMRN: {mrn}\nDOB: {dob}\n\n"


def fetch(
    dest: Path = Path("data/mtsamples/mtsamples.csv"),
    *,
    url: str = MTSAMPLES_URL,
    expected_sha256: str = MTSAMPLES_SHA256,
    timeout: float = 120.0,
) -> Path:
    """Download the CSV if absent; verify the pinned checksum either way."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            dest.write_bytes(response.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{dest} does not match the pinned MTSamples checksum "
            f"(got {digest}); delete it and re-fetch, or update the pin"
        )
    return dest


def load_transcriptions(
    csv_path: Path, *, min_chars: int = 500, max_chars: int = 3500
) -> list[str]:
    """Transcription texts within the length band, in stable file order.

    The band keeps single-call model inference practical; the cut is a
    benchmark parameter, not a silent cap — state it when reporting numbers.
    """
    texts: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("transcription") or "").strip()
            if min_chars <= len(text) <= max_chars:
                texts.append(text)
    return texts


def build_benchmark(
    csv_path: Path,
    count: int = 100,
    seed: int = 0,
    injections: int = 4,
    *,
    min_chars: int = 500,
    max_chars: int = 3500,
) -> list[SyntheticNote]:
    """Inject synthetic PHI into ``count`` transcripts, deterministically."""
    pool = load_transcriptions(csv_path, min_chars=min_chars, max_chars=max_chars)
    if count > len(pool):
        raise ValueError(f"asked for {count} notes but only {len(pool)} transcripts qualify")
    rng = random.Random(seed)
    fab = Fabricator(rng)
    notes: list[SyntheticNote] = []
    for i, pool_index in enumerate(sorted(rng.sample(range(len(pool)), count))):
        base = pool[pool_index]
        values: dict[str, str] = {}

        text, spans = render_slots(_HEADER, fab, values)
        gold = list(spans)

        boundaries = [m.end() for m in _SENTENCE_BOUNDARY.finditer(base)]
        k = min(injections, len(boundaries))
        points = sorted(rng.sample(boundaries, k)) if k else []
        templates = [rng.choice(INJECTIONS) for _ in range(k)]

        cursor = 0
        for point, template in zip(points, templates, strict=True):
            text += base[cursor:point]
            fragment, fragment_spans = render_slots(template + " ", fab, values)
            offset = len(text)
            gold.extend(
                GoldSpan(s.type, s.start + offset, s.end + offset, s.text)
                for s in fragment_spans
            )
            text += fragment
            cursor = point
        text += base[cursor:]

        notes.append(
            SyntheticNote(
                id=f"mts-{i:05d}",
                note_type="mtsamples",
                text=text,
                spans=tuple(sorted(gold, key=lambda s: s.start)),
            )
        )
    return notes
