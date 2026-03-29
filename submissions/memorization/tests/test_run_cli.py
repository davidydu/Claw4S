# tests/test_run_cli.py
"""Integration tests for run.py CLI and reproducibility metadata."""

import json
import subprocess
import sys
from pathlib import Path


SUBMISSION_DIR = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run command in submission directory and capture output."""
    return subprocess.run(
        cmd,
        cwd=SUBMISSION_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_run_help_includes_cli_options():
    """run.py should expose CLI help with key configurability flags."""
    result = _run([sys.executable, "run.py", "--help"], timeout=15)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "--seeds" in result.stdout
    assert "--hidden-dims" in result.stdout
    assert "--no-plots" in result.stdout
    assert "--output-dir" in result.stdout


def test_run_small_config_writes_run_metadata(tmp_path):
    """Small custom run should complete and write reproducibility metadata."""
    out_dir = tmp_path / "mini_results"
    result = _run(
        [
            sys.executable,
            "run.py",
            "--seeds",
            "7",
            "--hidden-dims",
            "5",
            "--n-train",
            "20",
            "--n-test",
            "10",
            "--d",
            "4",
            "--n-classes",
            "2",
            "--max-epochs",
            "5",
            "--no-plots",
            "--output-dir",
            str(out_dir),
        ],
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    results_json = out_dir / "results.json"
    assert results_json.exists()

    with results_json.open() as f:
        payload = json.load(f)

    run_metadata = payload.get("run_metadata")
    assert isinstance(run_metadata, dict), "results.json missing run_metadata"
    assert run_metadata["seeds"] == [7]
    assert run_metadata["hidden_dims"] == [5]
    assert run_metadata["n_train"] == 20
    assert run_metadata["n_test"] == 10
    assert run_metadata["d"] == 4
    assert run_metadata["n_classes"] == 2
    assert run_metadata["max_epochs"] == 5
    assert "python_version" in run_metadata
    assert "dependency_versions" in run_metadata
    assert "execution" in run_metadata
    assert "start_utc" in run_metadata["execution"]
    assert "end_utc" in run_metadata["execution"]
