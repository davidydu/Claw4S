"""Tests for src/report.py -- markdown report generation."""

from functools import lru_cache

from src.analysis import run_full_analysis
from src.config import MSI_ARTIFACT_THRESHOLD
from src.report import generate_report


@lru_cache(maxsize=1)
def _get_report():
    """Helper: generate report from full analysis."""
    results = run_full_analysis(seed=42)
    return generate_report(results)


def test_report_contains_title():
    """Report contains the analysis title."""
    report = _get_report()
    assert "Emergent Abilities" in report


def test_report_contains_findings():
    """Report contains a findings section."""
    report = _get_report()
    assert "Finding" in report or "Result" in report or "finding" in report


def test_report_contains_methodology():
    """Report contains methodology description."""
    report = _get_report()
    assert "metric" in report.lower()
    assert "sigmoid" in report.lower() or "linear" in report.lower()


def test_report_contains_limitations():
    """Report contains a limitations section."""
    report = _get_report()
    assert "limitation" in report.lower() or "caveat" in report.lower()


def test_report_contains_msi():
    """Report mentions Metric Sensitivity Index."""
    report = _get_report()
    assert "MSI" in report or "Metric Sensitivity" in report


def test_report_marks_single_token_msi_as_definitional():
    """Single-token tasks are flagged as non-interpretable for MSI verdicts."""
    report = _get_report()
    assert "Sports Understanding | 1.00" in report
    assert "N/A (n_tokens=1)" in report


def test_report_avoids_wall_clock_timestamps():
    """Report stays deterministic by avoiding wall-clock timestamps."""
    report = _get_report()
    assert "Generated:" not in report
    assert "Generated deterministically from hardcoded benchmark data" in report


def test_report_mentions_schaeffer():
    """Report references Schaeffer et al."""
    report = _get_report()
    assert "Schaeffer" in report


def test_report_mentions_bootstrap_uncertainty():
    """Report includes uncertainty language for MSI interpretation."""
    report = _get_report()
    assert "bootstrap" in report.lower()
    assert "95% CI" in report


def test_report_uses_configured_msi_threshold():
    """Report text reflects the configured MSI threshold."""
    report = _get_report()
    assert f"MSI > {MSI_ARTIFACT_THRESHOLD:.1f}" in report
