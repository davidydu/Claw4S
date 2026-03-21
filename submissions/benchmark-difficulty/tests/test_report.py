"""Tests for report generation module."""

import pytest
from src.analysis import run_full_analysis
from src.report import generate_report


@pytest.fixture
def results():
    """Run analysis on hardcoded data for report tests."""
    return run_full_analysis(use_hardcoded=True, seed=42)


def test_generate_report_returns_string(results):
    """generate_report returns a non-empty string."""
    report = generate_report(results)
    assert isinstance(report, str)
    assert len(report) > 100


def test_report_contains_key_sections(results):
    """Report contains required sections."""
    report = generate_report(results)
    assert "Summary" in report or "summary" in report.lower()
    assert "Correlation" in report or "correlation" in report.lower()
    assert "Feature Importance" in report or "feature importance" in report.lower()
    assert "Cross-Validation" in report or "cross-validation" in report.lower()


def test_report_contains_metrics(results):
    """Report contains key metric values."""
    report = generate_report(results)
    assert "R-squared" in report or "R²" in report or "r_squared" in report.lower()
    assert "MAE" in report or "mae" in report.lower()
    assert "Spearman" in report or "spearman" in report.lower()


def test_report_contains_num_questions(results):
    """Report states the number of questions analyzed."""
    report = generate_report(results)
    assert str(results["num_questions"]) in report
