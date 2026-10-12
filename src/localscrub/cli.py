"""Command-line interface.

Exit codes: 0 success; 1 input/IO error; 2 usage error; 3 entities or
escalations found under ``--fail-on-detect`` (for CI gates over log samples
and fixtures).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from localscrub.analyzer import scrub as scrub_text
from localscrub.config import ScrubConfig
from localscrub.corpus import generate_corpus, read_jsonl, write_jsonl
from localscrub.corpus.diversify import diversify_corpus
from localscrub.corpus.mtsamples import build_benchmark, fetch
from localscrub.corpus.sft import build_sft_examples
from localscrub.evaluation import evaluate
from localscrub.llm import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient, OllamaError
from localscrub.recognizers import NerBackend, NerRecognizer, Recognizer, default_recognizers
from localscrub.recognizers.ner import DEFAULT_NER_MODEL, onnx_cache_dir
from localscrub.report import build_report, write_audit
from localscrub.stage2 import LlmExtractor

app = typer.Typer(name="localscrub", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Local-first PHI/PII de-identification. Your data never leaves your machine."""


def _recognizers_with_ner(backend: NerBackend) -> tuple[Recognizer, ...]:
    """Default set plus the pretrained NER model; exit 1 if the extra is missing."""
    if backend is NerBackend.ONNX and not onnx_cache_dir(DEFAULT_NER_MODEL).is_dir():
        typer.secho(
            f"first --ner-backend onnx run: exporting {DEFAULT_NER_MODEL} to int8 ONNX "
            f"in {onnx_cache_dir(DEFAULT_NER_MODEL)} (one-time, a few minutes)",
            err=True,
        )
    try:
        ner = NerRecognizer(backend=backend)
    except ImportError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    return (*default_recognizers(), ner)


_NER_BACKEND_HELP = (
    "Inference engine for --ner: torch (default) or onnx — an int8 ONNX export, "
    "~3x faster on CPU with the same model (requires the [ner-onnx] extra)."
)


def _read_input(path: Path | None) -> tuple[str, str]:
    """Return (text, source label) from a file path or stdin."""
    if path is None or str(path) == "-":
        if sys.stdin.isatty():
            typer.secho(
                "no input: pass a file path or pipe text on stdin",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        return sys.stdin.read(), "<stdin>"
    try:
        return path.read_text(encoding="utf-8"), str(path)
    except OSError as exc:
        typer.secho(f"cannot read {path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc


@app.command()
def scrub(
    path: Annotated[
        Path | None,
        typer.Argument(help="Input file; omit or use '-' to read stdin."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write redacted text here instead of stdout."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write the JSON entity report to this file."),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Print the JSON report to stdout instead of redacted text."),
    ] = False,
    report_only: Annotated[
        bool,
        typer.Option("--report-only", help="Detect and report without rewriting the text."),
    ] = False,
    audit: Annotated[
        Path | None,
        typer.Option("--audit", help="Append a JSONL audit log (offsets only, no text) here."),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", min=0.0, max=1.0, help="Accept threshold (default 0.8)."),
    ] = 0.8,
    fail_on_detect: Annotated[
        bool,
        typer.Option("--fail-on-detect", help="Exit 3 if anything was detected or escalated."),
    ] = False,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm",
            help="Run stage 2: a local Ollama model extracts context-dependent PHI "
            "(names, places, narrative identifiers) and adjudicates escalations.",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", help="Ollama model for --llm."),
    ] = DEFAULT_MODEL,
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama server for --llm."),
    ] = DEFAULT_HOST,
    ner: Annotated[
        bool,
        typer.Option(
            "--ner",
            help="Add the pretrained clinical NER recognizer for names/places "
            "(requires the [ner] extra).",
        ),
    ] = False,
    ner_backend: Annotated[
        NerBackend, typer.Option("--ner-backend", help=_NER_BACKEND_HELP)
    ] = NerBackend.TORCH,
) -> None:
    """Scrub PHI/PII from a text file: redacted text + JSON entity report."""
    text, source = _read_input(path)
    config = ScrubConfig(accept_threshold=threshold)
    stage2 = LlmExtractor(OllamaClient(model=model, host=ollama_host)) if llm else None
    recognizers = _recognizers_with_ner(ner_backend) if ner else None
    try:
        result = scrub_text(text, config, stage2=stage2, recognizers=recognizers)
    except OllamaError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    report_data = build_report(result, source=source)

    if report is not None:
        report.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    if audit is not None:
        write_audit(result, source, audit)
    if out is not None and not report_only:
        out.write_text(result.redacted, encoding="utf-8")

    if json_out or report_only:
        typer.echo(json.dumps(report_data, indent=2))
    elif out is None:
        typer.echo(result.redacted, nl=False)

    if fail_on_detect and (result.entities or result.escalations):
        raise typer.Exit(3)


@app.command()
def synth(
    count: Annotated[
        int,
        typer.Option("--count", "-n", min=1, help="Number of notes to generate."),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option("--seed", help="RNG seed; the same seed always yields the same corpus."),
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write JSONL here instead of stdout."),
    ] = None,
    note_type: Annotated[
        list[str] | None,
        typer.Option("--note-type", help="Restrict to these note types (repeatable)."),
    ] = None,
    diversify: Annotated[
        bool,
        typer.Option(
            "--diversify",
            help="Paraphrase prose with a local Ollama model (needs a running GPU model); "
            "gold spans stay exact, rejected rewrites fall back to template prose.",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", help="Ollama model for --diversify."),
    ] = DEFAULT_MODEL,
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama server for --diversify."),
    ] = DEFAULT_HOST,
) -> None:
    """Generate a synthetic clinical-note corpus with gold PHI spans (JSONL).

    Every identifier is fabricated from reserved/fictional ranges — no real
    PHI is involved at any point. One JSON object per line: id, note_type,
    text, and the exact gold entities (type, start, end, text).
    """
    try:
        notes = generate_corpus(count, seed=seed, note_types=note_type)
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    if diversify:

        def _progress(done: int, total: int) -> None:
            typer.secho(f"diversify {done}/{total}", err=True)

        try:
            notes, fallbacks = diversify_corpus(
                notes, model=model, host=ollama_host, seed=seed, on_progress=_progress
            )
        except OllamaError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        if fallbacks:
            typer.secho(
                f"{fallbacks}/{len(notes)} rewrites rejected; those notes keep template prose",
                err=True,
                fg=typer.colors.YELLOW,
            )

    if out is not None:
        try:
            write_jsonl(notes, out)
        except OSError as exc:
            typer.secho(f"cannot write {out}: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(1) from exc
    else:
        # ensure_ascii guards against non-UTF-8 console encodings on Windows;
        # --out writes real UTF-8 instead.
        for note in notes:
            typer.echo(json.dumps(note.to_dict(), ensure_ascii=True))


@app.command()
def mtsamples(
    csv_path: Annotated[
        Path,
        typer.Option("--csv", help="Path to the MTSamples CSV."),
    ] = Path("data/mtsamples/mtsamples.csv"),
    count: Annotated[
        int,
        typer.Option("--count", "-n", min=1, help="Number of benchmark notes."),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option("--seed", help="RNG seed; same CSV + seed, same benchmark."),
    ] = 0,
    injections: Annotated[
        int,
        typer.Option("--injections", min=0, help="Narrative PHI injections per note."),
    ] = 4,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write JSONL here instead of stdout."),
    ] = None,
    do_fetch: Annotated[
        bool,
        typer.Option("--fetch", help="Download the public CSV first if it is missing."),
    ] = False,
) -> None:
    """Build the MTSamples-injection benchmark (JSONL with exact gold spans).

    Synthetic identifiers are injected into authentic public
    medical-transcription prose at known offsets — narrative sentences, no
    labels. Score it with `localscrub eval`.
    """
    try:
        if do_fetch:
            fetch(csv_path)
        elif not csv_path.exists():
            typer.secho(
                f"{csv_path} not found; pass --fetch to download the public MTSamples CSV",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        notes = build_benchmark(csv_path, count, seed=seed, injections=injections)
    except (OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    if out is not None:
        try:
            write_jsonl(notes, out)
        except OSError as exc:
            typer.secho(f"cannot write {out}: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(1) from exc
    else:
        for note in notes:
            typer.echo(json.dumps(note.to_dict(), ensure_ascii=True))


@app.command()
def sft(
    count: Annotated[
        int,
        typer.Option("--count", "-n", min=1, help="Number of training examples."),
    ] = 2000,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Corpus seed; keep disjoint from eval seeds."),
    ] = 7,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write JSONL here instead of stdout."),
    ] = None,
) -> None:
    """Build the stage-2 SFT dataset for the LoRA fine-tune recipe.

    One example per synthetic note: the exact stage-2 extraction prompt
    (stage-1 escalation hints included) paired with the gold answer in the
    extractor's reply format. See docs/finetune.md.
    """
    examples = build_sft_examples(generate_corpus(count, seed=seed))
    if out is not None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8", newline="\n") as fh:
                for example in examples:
                    fh.write(json.dumps(example, ensure_ascii=False))
                    fh.write("\n")
        except OSError as exc:
            typer.secho(f"cannot write {out}: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(1) from exc
    else:
        for example in examples:
            typer.echo(json.dumps(example, ensure_ascii=True))


@app.command("eval")
def eval_corpus(
    corpus: Annotated[
        Path,
        typer.Argument(help="Gold-labeled JSONL corpus (as written by `localscrub synth`)."),
    ],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Print the full JSON report instead of the table."),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option("--threshold", min=0.0, max=1.0, help="Accept threshold (default 0.8)."),
    ] = 0.8,
    llm: Annotated[
        bool,
        typer.Option("--llm", help="Score the full cascade: stage 1 + stage-2 Ollama pass."),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", help="Ollama model for --llm."),
    ] = DEFAULT_MODEL,
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama server for --llm."),
    ] = DEFAULT_HOST,
    ner: Annotated[
        bool,
        typer.Option(
            "--ner",
            help="Add the pretrained clinical NER recognizer (requires the [ner] extra).",
        ),
    ] = False,
    ner_backend: Annotated[
        NerBackend, typer.Option("--ner-backend", help=_NER_BACKEND_HELP)
    ] = NerBackend.TORCH,
) -> None:
    """Score the analyzer against a gold corpus: per-entity precision/recall.

    Prints a markdown results table (relaxed = type + overlap, strict =
    type + exact boundaries) plus redaction recall — the fraction of gold
    spans fully removed from the output, escalations included.
    """
    try:
        notes = read_jsonl(corpus)
    except (OSError, ValueError, KeyError) as exc:
        typer.secho(f"cannot read corpus {corpus}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    if not notes:
        typer.secho(f"corpus {corpus} is empty", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    stage2 = LlmExtractor(OllamaClient(model=model, host=ollama_host)) if llm else None
    recognizers = _recognizers_with_ner(ner_backend) if ner else None
    try:
        report = evaluate(
            notes,
            config=ScrubConfig(accept_threshold=threshold),
            stage2=stage2,
            recognizers=recognizers,
        )
    except OllamaError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    if json_out:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(report.format_table())
        typer.echo(
            f"\n{report.notes} notes, {report.total_gold} gold spans; "
            f"redaction recall {report.redaction_recall:.3f}"
        )
