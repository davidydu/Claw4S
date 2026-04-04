"""Tests for output artifact specification utilities."""

from pathlib import Path


def test_primary_artifact_paths_are_stable():
    from src.output_spec import PRIMARY_ARTIFACTS

    assert PRIMARY_ARTIFACTS == (
        "results/results.json",
        "results/report.md",
    )


def test_clear_primary_artifacts_removes_only_existing(tmp_path: Path):
    from src.output_spec import clear_primary_artifacts

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    existing = results_dir / "results.json"
    existing.write_text("{}")

    removed = clear_primary_artifacts(base_dir=tmp_path)
    assert removed == [existing]
    assert not existing.exists()
