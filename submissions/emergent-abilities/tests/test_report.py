"""Tests for src/report.py -- markdown report generation."""

from src.analysis import run_full_analysis
from src.report import generate_report


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


def test_report_mentions_schaeffer():
    """Report references Schaeffer et al."""
    report = _get_report()
    assert "Schaeffer" in report
