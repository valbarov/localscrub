"""Pretrained clinical NER recognizer — the optional ``[ner]`` extra.

Backed by a token-classification model fine-tuned for de-identification
(default: ``obi/deid_roberta_i2b2``, trained on the i2b2 2014 corpus), so
stage 1 gains the categories rules cannot see. CPU inference, milliseconds
per note. Install with ``pip install localscrub[ner]``.

Only labels the regex layer cannot cover are mapped — person, facility,
and employer names, and sub-state geography. Dates, phones, emails, and
IDs stay with the regex recognizers, whose boundaries and calibration are
already exact; emitting them here would only add boundary noise. Facility
and employer names map to NAME: Safe Harbor category (A) covers names of
relatives and employers, not just the patient's.

Adjacent same-type spans separated by short punctuation are merged
("4050 Mossbank Blvd" + "Cedar Vale" + "OH" + "44120" becomes one span):
a partially redacted address is still a leak, so span *completeness* is
the point of this recognizer. A period joins two NAME spans only when the
left one ends in a single capital letter ("Marcus T." + "Whitfield"):
elsewhere a period is a sentence boundary, and names on either side of it
are different people's.

The tokenizer glues a newline plus the next line's first word onto a
group's last token ("Whitfield\nMRN:"), so a group's spillover onto later
lines is usually boundary bleed, not detection. But it is not *always*
bleed — a real name can start the next line — so spillover segments are
re-checked: the pipeline runs once more on the spillover's own line, and
the segment survives only if the model still labels it without the
cross-line context that produced the bleed.

Two inference backends run the same model. ``torch`` (the default) is the
stock transformers pipeline. ``onnx`` runs an int8-quantized ONNX export
through onnxruntime (the ``[ner-onnx]`` extra): on the MTSamples benchmark
it is ~3x faster and uses half the RAM with identical scores. The export is
built once on first use and cached under ``~/.cache/localscrub/onnx``
(override with ``LOCALSCRUB_CACHE``); pass a directory as ``model`` to use
an export you built yourself.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from localscrub.entities import Entity, EntityType, Stage

DEFAULT_NER_MODEL = "obi/deid_roberta_i2b2"


class NerBackend(StrEnum):
    """How the token classifier is run."""

    TORCH = "torch"
    ONNX = "onnx"


#: model label -> our taxonomy; unlisted labels are ignored on purpose
_LABEL_MAP: dict[str, EntityType] = {
    "PATIENT": EntityType.NAME,
    "STAFF": EntityType.NAME,
    "HOSP": EntityType.NAME,  # facility names are Safe Harbor (A) names
    "PATORG": EntityType.NAME,  # so are employer names
    "LOC": EntityType.GEO_SUBDIVISION,
}

#: what may separate two spans that are really one identifier: spaces, commas,
#: hyphens — never newlines or periods, which end the identifier's context
_JOINABLE = re.compile(r"^[ ,\-–]{1,4}$")

#: the one period that does not end a name's context: a middle initial's
_INITIAL_GAP = re.compile(r"^\.\s{0,2}$")
_ENDS_WITH_INITIAL = re.compile(r"(?:^|\s)[A-Z]$")

#: sentence punctuation the model drags into a span is never the identifier's
_TRAILING_JUNK = re.compile(r"[\s.,;:]+$")

#: character windows sized well under the model's 512-token limit
_CHUNK_CHARS = 1500
_CHUNK_OVERLAP = 200


def _chunks(text: str) -> list[tuple[int, str]]:
    """(offset, window) pairs covering ``text`` with overlap, split at spaces."""
    if len(text) <= _CHUNK_CHARS:
        return [(0, text)]
    windows: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + _CHUNK_CHARS, len(text))
        if end < len(text):
            space = text.rfind(" ", start + _CHUNK_CHARS // 2, end)
            if space != -1:
                end = space
        windows.append((start, text[start:end]))
        if end == len(text):
            break
        start = max(end - _CHUNK_OVERLAP, start + 1)
    return windows


class TokenClassifier(Protocol):
    """The slice of a transformers token-classification pipeline we use."""

    def __call__(self, text: str) -> list[dict[str, Any]]: ...


def _load_pipeline(model: str) -> TokenClassifier:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the [ner] extra is not installed; run: pip install 'localscrub[ner]'"
        ) from exc
    return pipeline("token-classification", model=model, aggregation_strategy="first")


def onnx_cache_dir(model: str = DEFAULT_NER_MODEL) -> Path:
    """Where the int8 ONNX export of ``model`` lives (or will, after first use)."""
    root = Path(os.environ.get("LOCALSCRUB_CACHE") or Path.home() / ".cache" / "localscrub")
    return root / "onnx" / f"{model.replace('/', '--')}-int8"


def _export_onnx_int8(model: str, out: Path) -> None:
    """Export ``model`` to ONNX and dynamically quantize it to int8 into ``out``.

    Dynamic (weight-only) quantization needs no calibration data, which is
    what makes a one-shot export on the user's machine safe. The VNNI config
    keeps full int8 range; CPUs without AVX-512 VNNI still run it, just
    with less of the speed-up.
    """
    from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    # build next to the target and rename last, so an interrupted export never
    # leaves a half-written directory that later runs would trust
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    with tempfile.TemporaryDirectory() as fp32:
        ORTModelForTokenClassification.from_pretrained(model, export=True).save_pretrained(fp32)
        quantizer = ORTQuantizer.from_pretrained(fp32)
        quantizer.quantize(
            save_dir=staging,
            quantization_config=AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            ),
        )
    AutoTokenizer.from_pretrained(model).save_pretrained(staging)
    staging.replace(out)


def _load_onnx_pipeline(model: str) -> TokenClassifier:
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer, pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the [ner-onnx] extra is not installed; run: pip install 'localscrub[ner-onnx]'"
        ) from exc
    path = Path(model)
    if not path.is_dir():
        path = onnx_cache_dir(model)
        if not path.is_dir():
            _export_onnx_int8(model, path)
    onnx_files = sorted(path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"no .onnx model in {path}")
    # prefer the quantized file when an export dir holds both
    quantized = [f for f in onnx_files if "quantized" in f.name]
    onnx_file = (quantized or onnx_files)[0].name
    ort_model = ORTModelForTokenClassification.from_pretrained(path, file_name=onnx_file)
    return pipeline(
        "token-classification",
        model=ort_model,
        tokenizer=AutoTokenizer.from_pretrained(path),
        aggregation_strategy="first",
    )


_LOADERS = {NerBackend.TORCH: _load_pipeline, NerBackend.ONNX: _load_onnx_pipeline}


class NerRecognizer:
    """NAME/GEO detection with a pretrained de-identification model.

    ``confidence_cap`` bounds the model score before it enters the
    calibration bands: scores below the accept threshold escalate
    fail-closed like any other recognizer's.

    ``backend`` picks the inference engine (see module docstring); an
    explicit ``pipeline`` bypasses loading altogether.
    """

    name: ClassVar[str] = "ner"

    def __init__(
        self,
        model: str = DEFAULT_NER_MODEL,
        *,
        pipeline: TokenClassifier | None = None,
        backend: NerBackend | str = NerBackend.TORCH,
        confidence_cap: float = 0.95,
    ) -> None:
        self.backend = NerBackend(backend)
        self._pipeline = pipeline if pipeline is not None else _LOADERS[self.backend](model)
        self.confidence_cap = confidence_cap

    def analyze(self, text: str) -> list[Entity]:
        raw: list[tuple[EntityType, int, int, float]] = []
        seen: set[tuple[EntityType, int, int]] = set()
        line_cache: dict[tuple[int, int], list[tuple[EntityType, int, int, float]]] = {}
        for chunk_offset, window in _chunks(text):
            self._analyze_window(text, window, chunk_offset, raw, seen, line_cache)

        raw.sort(key=lambda r: r[1])
        merged: list[tuple[EntityType, int, int, float]] = []
        for etype, start, end, score in raw:
            if merged:
                prev_type, prev_start, prev_end, prev_score = merged[-1]
                gap = text[prev_end:start]
                joinable = _JOINABLE.match(gap) or (
                    etype is EntityType.NAME
                    and _INITIAL_GAP.match(gap)
                    and _ENDS_WITH_INITIAL.search(text[prev_start:prev_end])
                )
                if etype is prev_type and prev_end <= start and joinable:
                    merged[-1] = (prev_type, prev_start, end, min(prev_score, score))
                    continue
            merged.append((etype, start, end, score))

        return [
            Entity(
                type=etype,
                start=start,
                end=end,
                text=text[start:end],
                confidence=min(score, self.confidence_cap),
                stage=Stage.RULES,
            )
            for etype, start, end, score in merged
        ]

    def _analyze_window(
        self,
        text: str,
        window: str,
        chunk_offset: int,
        raw: list[tuple[EntityType, int, int, float]],
        seen: set[tuple[EntityType, int, int]],
        line_cache: dict[tuple[int, int], list[tuple[EntityType, int, int, float]]],
    ) -> None:
        for item in self._pipeline(window):
            entity_type = _LABEL_MAP.get(item.get("entity_group", ""))
            if entity_type is None:
                continue
            start = chunk_offset + int(item["start"])
            end = chunk_offset + int(item["end"])
            score = float(item["score"])
            # The pipeline can aggregate one label run across a paragraph
            # break ("name@example.org.\n\nNext Name"); no identifier spans a
            # line break, so split model groups at newlines first. The first
            # segment is where the model anchored the group; later segments
            # are usually tokenizer boundary bleed and must be re-earned.
            for index, line in enumerate(re.finditer(r"[^\n]+", text[start:end])):
                segment = _TRAILING_JUNK.sub("", line.group())
                trimmed = segment.lstrip()
                # A person or place never contains "@": that's the model
                # labeling the human name inside an email address.
                if not trimmed or "@" in trimmed:
                    continue
                line_start = start + line.start() + (len(segment) - len(trimmed))
                if index > 0:
                    self._recheck_spillover(
                        text, line_start, line_start + len(trimmed), raw, seen, line_cache
                    )
                    continue
                record = (entity_type, line_start, line_start + len(trimmed))
                if record not in seen:  # overlapping windows re-find spans
                    seen.add(record)
                    raw.append((*record, score))

    def _recheck_spillover(
        self,
        text: str,
        frag_start: int,
        frag_end: int,
        raw: list[tuple[EntityType, int, int, float]],
        seen: set[tuple[EntityType, int, int]],
        line_cache: dict[tuple[int, int], list[tuple[EntityType, int, int, float]]],
    ) -> None:
        """Keep a group's spillover segment only if its own line re-earns it.

        The pipeline runs on the spillover's full line in isolation; whatever
        it labels *there* — with its own spans, types, and scores — replaces
        the bled fragment, but only where it overlaps the fragment (the rest
        of the line was already analyzed in its window).
        """
        line_start = text.rfind("\n", 0, frag_start) + 1
        newline = text.find("\n", frag_start)
        line_end = len(text) if newline == -1 else newline
        key = (line_start, line_end)
        if key not in line_cache:
            found: list[tuple[EntityType, int, int, float]] = []
            for item in self._pipeline(text[line_start:line_end]):
                entity_type = _LABEL_MAP.get(item.get("entity_group", ""))
                if entity_type is None:
                    continue
                start, end = int(item["start"]), int(item["end"])
                if not 0 <= start < end <= line_end - line_start:
                    continue
                found.append(
                    (entity_type, line_start + start, line_start + end, float(item["score"]))
                )
            line_cache[key] = found
        for entity_type, start, end, score in line_cache[key]:
            if not (start < frag_end and frag_start < end):
                continue
            segment = _TRAILING_JUNK.sub("", text[start:end])
            trimmed = segment.lstrip()
            if not trimmed or "@" in trimmed:
                continue
            span_start = start + (len(segment) - len(trimmed))
            record = (entity_type, span_start, span_start + len(trimmed))
            if record not in seen:
                seen.add(record)
                raw.append((*record, score))
