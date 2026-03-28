"""Tests for statistical summaries and checksum manifests."""

from pathlib import Path

from src.summary import (
    compute_paired_delta_summary,
    compute_sha256,
    write_checksum_manifest,
)


def _entry(model_label: str, layer_name: str, ks_statistic: float) -> dict:
    return {
        "model_label": model_label,
        "layer_name": layer_name,
        "ks_statistic": ks_statistic,
    }


def test_compute_paired_delta_summary_counts_and_ordering():
    """Summary should correctly count positive/negative/tied paired deltas."""
    trained = [
        _entry("m1", "fc1", 0.30),
        _entry("m1", "fc2", 0.10),
        _entry("m2", "fc1", 0.20),
        _entry("m2", "fc2", 0.50),
    ]
    untrained = [
        _entry("m1", "fc1", 0.10),  # +0.20
        _entry("m1", "fc2", 0.20),  # -0.10
        _entry("m2", "fc1", 0.20),  # +0.00
        _entry("m2", "fc2", 0.20),  # +0.30
    ]

    summary = compute_paired_delta_summary(trained, untrained, bootstrap_samples=500, seed=7)

    assert summary["n_pairs"] == 4
    assert summary["n_positive"] == 2
    assert summary["n_negative"] == 1
    assert summary["n_ties"] == 1
    assert summary["positive_fraction"] == 0.5
    assert summary["avg_delta"] == 0.1
    assert summary["median_delta"] == 0.1
    assert summary["bootstrap_ci_low"] <= summary["avg_delta"] <= summary["bootstrap_ci_high"]
    assert 0.0 <= summary["sign_test_pvalue"] <= 1.0


def test_compute_paired_delta_summary_bootstrap_is_deterministic_with_seed():
    """Bootstrap confidence bounds should be stable for a fixed seed."""
    trained = [_entry("m1", "fc1", 0.40), _entry("m1", "fc2", 0.25)]
    untrained = [_entry("m1", "fc1", 0.10), _entry("m1", "fc2", 0.05)]

    s1 = compute_paired_delta_summary(trained, untrained, bootstrap_samples=400, seed=123)
    s2 = compute_paired_delta_summary(trained, untrained, bootstrap_samples=400, seed=123)

    assert s1["bootstrap_ci_low"] == s2["bootstrap_ci_low"]
    assert s1["bootstrap_ci_high"] == s2["bootstrap_ci_high"]


def test_write_checksum_manifest_uses_relative_paths(tmp_path: Path):
    """Manifest lines should use relative filenames and correct SHA256 values."""
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    results_path.write_text('{"ok": true}\n')
    report_path.write_text("# report\n")

    manifest_path = tmp_path / "checksums.sha256"
    checksums = write_checksum_manifest(
        paths=[results_path, report_path],
        manifest_path=manifest_path,
        base_dir=tmp_path,
    )

    assert checksums["results.json"] == compute_sha256(results_path)
    assert checksums["report.md"] == compute_sha256(report_path)

    manifest_lines = manifest_path.read_text().strip().splitlines()
    assert manifest_lines[0].endswith("  report.md") or manifest_lines[0].endswith("  results.json")
    assert any(line.endswith("  results.json") for line in manifest_lines)
    assert any(line.endswith("  report.md") for line in manifest_lines)
