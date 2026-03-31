"""Tests for runtime helpers used by top-level scripts."""

from pathlib import Path

import pytest

from src.runtime import ensure_submission_cwd


def test_ensure_submission_cwd_accepts_submission_directory(tmp_path, monkeypatch):
    """The helper should accept execution from the script directory."""
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    script_path = submission_dir / "run.py"
    script_path.write_text("pass\n")

    monkeypatch.chdir(submission_dir)

    assert ensure_submission_cwd(script_path) == submission_dir.resolve()


def test_ensure_submission_cwd_rejects_other_directory(tmp_path, monkeypatch, capsys):
    """The helper should reject execution from an unrelated directory."""
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    script_path = submission_dir / "validate.py"
    script_path.write_text("pass\n")
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()

    monkeypatch.chdir(other_dir)

    with pytest.raises(SystemExit) as excinfo:
        ensure_submission_cwd(script_path)

    assert excinfo.value.code == 1
    captured = capsys.readouterr().out
    assert "validate.py must be executed from the submission directory." in captured
    assert str(submission_dir.resolve()) in captured
