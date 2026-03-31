"""Tests for run.py entrypoint behavior."""

from pathlib import Path

import run


def test_ensure_submission_dir_switches_cwd(monkeypatch, tmp_path):
    """Entry point should switch to script directory when launched elsewhere."""
    script_dir = tmp_path / "submissions" / "feature-attribution"
    script_file = script_dir / "run.py"
    original_cwd = tmp_path / "outside"

    chdir_calls = []
    monkeypatch.setattr(run.os, "getcwd", lambda: str(original_cwd))
    monkeypatch.setattr(run.os, "chdir", lambda path: chdir_calls.append(path))
    monkeypatch.setattr(run.sys, "path", [])

    resolved = run._ensure_submission_dir(str(script_file))

    assert resolved == str(script_dir.resolve())
    assert chdir_calls == [str(script_dir.resolve())]
    assert str(script_dir.resolve()) in run.sys.path
