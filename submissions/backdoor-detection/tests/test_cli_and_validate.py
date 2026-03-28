"""Tests for CLI execution guards and validation thesis checks."""

from pathlib import Path

import pytest

import validate
from src.cli import ensure_submission_cwd


def test_ensure_submission_cwd_accepts_submission_directory():
    script_path = Path(__file__).resolve().parents[1] / "run.py"
    assert ensure_submission_cwd(script_path, cwd=script_path.parent) == script_path.parent


def test_ensure_submission_cwd_rejects_other_directories():
    script_path = Path(__file__).resolve().parents[1] / "run.py"
    with pytest.raises(RuntimeError, match="run.py must be executed from"):
        ensure_submission_cwd(script_path, cwd=script_path.parent.parent)


def test_strong_trigger_thesis_check_ignores_five_percent_poison_configs():
    results = [
        {"config": {"poison_fraction": 0.05, "trigger_strength": 10.0}, "detection_auc": 0.20},
        {"config": {"poison_fraction": 0.10, "trigger_strength": 10.0}, "detection_auc": 0.95},
        {"config": {"poison_fraction": 0.30, "trigger_strength": 10.0}, "detection_auc": 1.00},
        {"config": {"poison_fraction": 0.30, "trigger_strength": 5.0}, "detection_auc": 0.30},
    ]

    passed, total = validate.strong_trigger_thesis_check(results)

    assert (passed, total) == (2, 2)
