# tests/test_validate_cli.py
"""Integration tests for validate.py reproducibility checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


SUBMISSION_DIR = Path(__file__).resolve().parents[1]


def _run_validate(work_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "validate.py"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _setup_validation_bundle(tmp_path: Path, payload: dict) -> Path:
    work_dir = tmp_path / "submission"
    results_dir = work_dir / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)

    shutil.copy2(SUBMISSION_DIR / "validate.py", work_dir / "validate.py")
    shutil.copy2(SUBMISSION_DIR / "results" / "report.md", results_dir / "report.md")
    shutil.copy2(
        SUBMISSION_DIR / "results" / "figures" / "memorization_curve.png",
        figures_dir / "memorization_curve.png",
    )
    shutil.copy2(
        SUBMISSION_DIR / "results" / "figures" / "threshold_comparison.png",
        figures_dir / "threshold_comparison.png",
    )

    with (results_dir / "results.json").open("w") as f:
        json.dump(payload, f, indent=2)

    return work_dir


def test_validate_requires_run_metadata(tmp_path):
    """Validation should fail when reproducibility metadata is missing."""
    with (SUBMISSION_DIR / "results" / "results.json").open() as f:
        payload = json.load(f)
    payload.pop("run_metadata", None)

    work_dir = _setup_validation_bundle(tmp_path, payload)
    result = _run_validate(work_dir)
    assert result.returncode != 0
    assert "run_metadata" in result.stdout


def test_validate_accepts_run_metadata(tmp_path):
    """Validation should pass when run_metadata is present and complete."""
    with (SUBMISSION_DIR / "results" / "results.json").open() as f:
        payload = json.load(f)

    payload["run_metadata"] = {
        "seeds": [42, 43, 44],
        "hidden_dims": [5, 10, 20, 40, 80, 160, 320, 640],
        "n_train": 200,
        "n_test": 50,
        "d": 20,
        "n_classes": 10,
        "max_epochs": 5000,
        "lr": 0.001,
        "python_version": "3.13.5",
        "dependency_versions": {"torch": "2.6.0", "numpy": "2.2.4", "scipy": "1.15.2"},
        "execution": {
            "start_utc": "2026-03-28 20:00:00 UTC",
            "end_utc": "2026-03-28 20:01:00 UTC",
            "plots_generated": True,
        },
    }

    work_dir = _setup_validation_bundle(tmp_path, payload)
    result = _run_validate(work_dir)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Validation passed." in result.stdout
