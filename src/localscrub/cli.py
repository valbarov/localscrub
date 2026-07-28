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
from localscrub.corpus import generate_corpus, write_jsonl
from localscrub.corpus.diversify import DEFAULT_HOST, DEFAULT_MODEL, OllamaError, diversify_corpus
from localscrub.report import build_report, write_audit

app = typer.Typer(name="localscrub", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Local-first PHI/PII de-identification. Your data never leaves your machine."""


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
) -> None:
    """Scrub PHI/PII from a text file: redacted text + JSON entity report."""
    text, source = _read_input(path)
    config = ScrubConfig(accept_threshold=threshold)
    result = scrub_text(text, config)
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
