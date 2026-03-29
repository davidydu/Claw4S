"""Tests for run/validate artifact freshness and completeness checks."""

from pathlib import Path

from run import cleanup_previous_outputs
from src.output_spec import FIGURE_OUTPUTS, TOP_LEVEL_OUTPUTS
from validate import find_missing_artifacts


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok")


def test_find_missing_artifacts_when_outputs_complete(tmp_path):
    """Validation should report no missing artifacts for complete outputs."""
    results_dir = tmp_path / "results"

    for filename in TOP_LEVEL_OUTPUTS:
        _touch(results_dir / filename)
    for filename in FIGURE_OUTPUTS:
        _touch(results_dir / "figures" / filename)

    assert find_missing_artifacts(results_dir) == []


def test_find_missing_artifacts_reports_missing_files(tmp_path):
    """Validation should identify specific missing output artifacts."""
    results_dir = tmp_path / "results"
    _touch(results_dir / "results.json")
    _touch(results_dir / "figures" / "collusion_heatmap.png")

    missing = find_missing_artifacts(results_dir)

    assert "report.md" in missing
    assert "statistical_tests.json" in missing
    assert "figures/memory_effect.png" in missing
    assert "figures/auditor_agreement.png" in missing


def test_cleanup_previous_outputs_removes_only_expected_artifacts(tmp_path):
    """Cleanup should remove prior output artifacts but keep unrelated files."""
    results_dir = tmp_path / "results"
    for filename in TOP_LEVEL_OUTPUTS:
        _touch(results_dir / filename)
    for filename in FIGURE_OUTPUTS:
        _touch(results_dir / "figures" / filename)
    _touch(results_dir / "custom_note.txt")

    removed = cleanup_previous_outputs(results_dir)

    assert set(removed) == {
        "results.json",
        "report.md",
        "statistical_tests.json",
        "figures/collusion_heatmap.png",
        "figures/memory_effect.png",
        "figures/auditor_agreement.png",
    }
    assert not (results_dir / "results.json").exists()
    assert not (results_dir / "figures" / "collusion_heatmap.png").exists()
    assert (results_dir / "custom_note.txt").exists()
