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


def test_check_config_grid_coverage_flags_missing_duplicate_and_unexpected():
    metadata = {
        "poison_fractions": [0.1, 0.2],
        "trigger_strengths": [10.0],
        "hidden_dims": [64],
    }
    results = [
        {"config": {"poison_fraction": 0.1, "trigger_strength": 10.0, "hidden_dim": 64}},
        {"config": {"poison_fraction": 0.1, "trigger_strength": 10.0, "hidden_dim": 64}},
        {"config": {"poison_fraction": 0.3, "trigger_strength": 10.0, "hidden_dim": 64}},
    ]

    missing, unexpected, duplicates = validate.check_config_grid_coverage(results, metadata)

    assert missing == {(0.2, 10.0, 64)}
    assert unexpected == {(0.3, 10.0, 64)}
    assert duplicates == {(0.1, 10.0, 64): 2}


def test_thesis_requirement_satisfied_rejects_empty_subset():
    results = [
        {"config": {"poison_fraction": 0.05, "trigger_strength": 5.0}, "detection_auc": 0.2},
        {"config": {"poison_fraction": 0.10, "trigger_strength": 5.0}, "detection_auc": 0.3},
    ]

    ok, passed, total = validate.thesis_requirement_satisfied(results)

    assert not ok
    assert passed == 0
    assert total == 0
