import json

from typer.testing import CliRunner

from localscrub.cli import app

runner = CliRunner()

NOTE = "Contact test@example.com about MRN: 4859302."


def test_scrub_file_to_stdout(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text(NOTE, encoding="utf-8")
    result = runner.invoke(app, ["scrub", str(src)])
    assert result.exit_code == 0
    assert "[EMAIL_1]" in result.output
    assert "test@example.com" not in result.output


def test_scrub_stdin():
    result = runner.invoke(app, ["scrub"], input=NOTE)
    assert result.exit_code == 0
    assert "[EMAIL_1]" in result.output


def test_json_flag_prints_report():
    result = runner.invoke(app, ["scrub", "--json"], input=NOTE)
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["schema_version"] == 1
    assert report["source"] == "<stdin>"
    assert {e["type"] for e in report["entities"]} == {"EMAIL", "MRN"}


def test_out_and_report_files(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text(NOTE, encoding="utf-8")
    out = tmp_path / "clean.txt"
    report = tmp_path / "report.json"
    result = runner.invoke(
        app, ["scrub", str(src), "--out", str(out), "--report", str(report)]
    )
    assert result.exit_code == 0
    assert "[MRN_1]" in out.read_text(encoding="utf-8")
    assert json.loads(report.read_text(encoding="utf-8"))["source"] == str(src)


def test_report_only_does_not_write_out(tmp_path):
    out = tmp_path / "clean.txt"
    result = runner.invoke(app, ["scrub", "--report-only", "--out", str(out)], input=NOTE)
    assert result.exit_code == 0
    assert not out.exists()
    assert json.loads(result.output)["escalated"] is False


def test_audit_file_written(tmp_path):
    audit = tmp_path / "audit.jsonl"
    result = runner.invoke(app, ["scrub", "--audit", str(audit)], input=NOTE)
    assert result.exit_code == 0
    assert audit.read_text(encoding="utf-8").count("\n") == 2


def test_fail_on_detect_exit_3():
    result = runner.invoke(app, ["scrub", "--fail-on-detect"], input=NOTE)
    assert result.exit_code == 3


def test_fail_on_detect_clean_input_exit_0():
    result = runner.invoke(app, ["scrub", "--fail-on-detect"], input="nothing sensitive")
    assert result.exit_code == 0


def test_missing_file_exit_1(tmp_path):
    result = runner.invoke(app, ["scrub", str(tmp_path / "absent.txt")])
    assert result.exit_code == 1
