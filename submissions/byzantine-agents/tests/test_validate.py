"""Tests for validate.py entrypoint behavior."""

from pathlib import Path

import json

import validate


def test_validate_fails_with_clear_message_when_artifacts_missing(
    tmp_path: Path,
    capsys,
):
    rc = validate.main(["--results-dir", str(tmp_path / "results")])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Missing required artifact(s)" in out
    assert "results.json" in out
    assert "report.md" in out


def test_validate_checks_for_missing_report_even_if_results_json_exists(
    tmp_path: Path,
    capsys,
):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    payload = {
        "metadata": {
            "total_configs": 1,
            "elapsed_seconds": 0.1,
        },
        "raw_results": [],
        "summaries": [],
        "derived_metrics": [],
        "amplifications": [],
    }
    (results_dir / "results.json").write_text(json.dumps(payload))

    rc = validate.main(["--results-dir", str(results_dir)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "report.md" in out
