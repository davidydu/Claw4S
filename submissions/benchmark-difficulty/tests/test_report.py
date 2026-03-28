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


def test_report_contains_baseline_and_significance(results):
    """Report documents baseline comparison and permutation testing."""
    report = generate_report(results)
    assert "Baseline Comparison" in report
    assert "Permutation test" in report


def test_report_flags_low_r_squared_as_insufficient():
    """Low cross-validated R-squared should be framed as insufficient signal."""
    results = {
        "num_questions": 1172,
        "model_metrics": {"r_squared": 0.5102, "mae": 0.1552},
        "cv_metrics": {
            "mean_r_squared": 0.0071,
            "std_r_squared": 0.0100,
            "mean_mae": 0.2228,
            "std_mae": 0.0065,
            "mean_spearman": 0.1273,
            "std_spearman": 0.0231,
            "oof_spearman": 0.1200,
            "fold_scores": [
                {"r_squared": 0.0012, "mae": 0.2209, "spearman_rho": 0.1107},
            ],
        },
        "correlations": {
            "negation_count": {"rho": 0.0708, "pvalue": 0.0153},
            "answer_entropy": {"rho": -0.0374, "pvalue": 0.2008},
        },
        "ranked_features": [("flesch_kincaid_grade", 0.1564)],
        "feature_importances": {"flesch_kincaid_grade": 1.0},
        "baseline_metrics": {
            "mean_r_squared": -0.005,
            "std_r_squared": 0.002,
            "mean_mae": 0.230,
            "std_mae": 0.004,
            "mean_spearman": 0.0,
            "std_spearman": 0.0,
            "oof_spearman": 0.0,
            "fold_scores": [
                {"r_squared": -0.005, "mae": 0.230, "spearman_rho": 0.0},
            ],
        },
        "significance": {
            "oof_spearman": 0.12,
            "permutation_pvalue": 0.02,
            "n_permutations": 200,
        },
        "data_provenance": {
            "source": "huggingface",
            "dataset_name": "furonghuang-lab/Easy2Hard-Bench",
            "config": "E2H-ARC",
            "split": "eval",
            "revision": "55bc0d2fb10954151e669d2026b87fa896f2fa26",
            "is_fallback": False,
        },
        "predictions": [0.5],
        "difficulties": [0.5],
        "seed": 42,
    }

    report = generate_report(results)

    assert "insufficient" in report.lower()
