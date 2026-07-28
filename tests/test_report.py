import json

from localscrub import __version__, scrub
from localscrub.report import build_report, write_audit

NOTE = "Contact test@example.com or SSN: 123456789 today"


def test_report_shape():
    result = scrub(NOTE)
    report = build_report(result, source="note.txt")
    assert report["schema_version"] == 1
    assert report["tool"] == {"name": "localscrub", "version": __version__}
    assert report["source"] == "note.txt"
    assert report["stats"]["chars"] == len(NOTE)
    assert report["stats"]["counts"]["EMAIL"] == 1
    # round-trips through JSON
    assert json.loads(json.dumps(report)) == report


def test_report_marks_escalated_entities():
    result = scrub(NOTE)
    report = build_report(result)
    assert report["escalated"] is True
    by_type = {e["type"]: e for e in report["entities"]}
    assert by_type["EMAIL"]["escalated"] is False
    assert by_type["EMAIL"]["placeholder"] == "[EMAIL_1]"
    assert by_type["SSN"]["escalated"] is True
    [esc] = report["escalations"]
    assert esc["reason"] == "low_confidence"
    assert esc["candidates"][0]["type"] == "SSN"


def test_audit_log_appends_jsonl_without_identifier_text(tmp_path):
    result = scrub(NOTE)
    audit = tmp_path / "audit.jsonl"
    write_audit(result, "note.txt", audit)
    write_audit(result, "note.txt", audit)  # append, not overwrite

    lines = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 * (len(result.entities) + len(result.escalations))
    assert {line["action"] for line in lines} == {"redact", "escalate"}
    content = audit.read_text(encoding="utf-8")
    assert "test@example.com" not in content
    assert "123456789" not in content
