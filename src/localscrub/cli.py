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
