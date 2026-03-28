"""Tests for src/analysis.py -- core emergence analyses."""

import numpy as np
from src.analysis import (
    infer_per_token_accuracy,
    compute_metric_comparison,
    compute_nonlinearity_scores,
    generate_synthetic_demo,
    run_full_analysis,
)
from src.config import NONLINEARITY_BOOTSTRAP_SAMPLES


def test_infer_per_token_accuracy():
    """Infer per-token accuracy from exact-match accuracy."""
    # p^n = exact_match -> p = exact_match^(1/n)
    p = infer_per_token_accuracy(exact_match=0.81, n_tokens=2)
    assert abs(p - 0.9) < 1e-6, f"Expected ~0.9, got {p}"


def test_infer_per_token_accuracy_zero():
    """Zero exact match -> zero per-token accuracy."""
    p = infer_per_token_accuracy(exact_match=0.0, n_tokens=4)
    assert p == 0.0


def test_infer_per_token_accuracy_one():
    """Perfect exact match -> perfect per-token accuracy."""
    p = infer_per_token_accuracy(exact_match=1.0, n_tokens=4)
    assert abs(p - 1.0) < 1e-10


def test_metric_comparison_returns_both_types():
    """Metric comparison returns both discontinuous and continuous metrics."""
    result = compute_metric_comparison("2_digit_multiplication")
    assert "task" in result
    assert "entries" in result
    assert len(result["entries"]) > 0
    for entry in result["entries"]:
        assert "exact_match" in entry
        assert "partial_credit" in entry
        assert "token_edit_distance" in entry
        assert "params_b" in entry


def test_metric_comparison_continuous_smoother():
    """Continuous metrics should be smoother than discontinuous for the same data."""
    result = compute_metric_comparison("2_digit_multiplication")
    entries = result["entries"]
    if len(entries) < 3:
        return  # Skip if insufficient data

    # Check that partial credit values are >= exact match values
    for entry in entries:
        assert entry["partial_credit"] >= entry["exact_match"] or abs(
            entry["partial_credit"] - entry["exact_match"]
        ) < 0.01


def test_nonlinearity_detection_returns_scores():
    """Nonlinearity detection returns MSI and fit comparison for each task."""
    scores = compute_nonlinearity_scores()
    assert len(scores) > 0
    for task_name, task_scores in scores.items():
        assert "msi" in task_scores, f"Missing 'msi' for {task_name}"
        assert "linear_r2_continuous" in task_scores
        assert "sigmoid_r2_continuous" in task_scores
        assert "linear_r2_discontinuous" in task_scores
        assert "sigmoid_r2_discontinuous" in task_scores


def test_nonlinearity_scores_include_task_metadata():
    """Scores include task metadata needed for interpretation."""
    scores = compute_nonlinearity_scores()
    sports = scores["sports_understanding"]
    assert sports["n_tokens"] == 1
    assert sports["metric_type"] == "multiple_choice"


def test_nonlinearity_scores_include_uncertainty_fields():
    """Scores include deterministic bootstrap uncertainty metadata."""
    scores = compute_nonlinearity_scores()
    for task_name, task_scores in scores.items():
        assert "msi_ci_lower" in task_scores, f"Missing CI lower for {task_name}"
        assert "msi_ci_upper" in task_scores, f"Missing CI upper for {task_name}"
        assert "artifact_probability" in task_scores
        assert "artifact_threshold" in task_scores
        assert "n_bootstrap" in task_scores
        assert task_scores["n_bootstrap"] == NONLINEARITY_BOOTSTRAP_SAMPLES

        assert 0.0 <= task_scores["artifact_probability"] <= 1.0
        assert task_scores["msi_ci_lower"] <= task_scores["msi_ci_upper"]

        if np.isfinite(task_scores["msi"]):
            assert np.isfinite(task_scores["msi_ci_lower"])
            assert np.isfinite(task_scores["msi_ci_upper"])


def test_synthetic_demo_shows_divergence():
    """Synthetic demo: exact match << partial credit at low per-token accuracy."""
    demo = generate_synthetic_demo(seed=42)
    assert "per_token_acc" in demo
    assert "exact_match" in demo
    assert "partial_credit" in demo

    # At low per-token accuracy, exact match should be much lower
    low_idx = 0  # lowest per-token accuracy
    assert demo["exact_match"][low_idx] < demo["partial_credit"][low_idx]


def test_synthetic_demo_deterministic():
    """Same seed -> same synthetic results."""
    demo1 = generate_synthetic_demo(seed=42)
    demo2 = generate_synthetic_demo(seed=42)
    assert np.allclose(demo1["exact_match"], demo2["exact_match"])
    assert np.allclose(demo1["partial_credit"], demo2["partial_credit"])


def test_full_analysis_returns_complete_results():
    """Full analysis returns all expected sections."""
    results = run_full_analysis(seed=42)
    assert "metric_comparisons" in results
    assert "nonlinearity_scores" in results
    assert "synthetic_demo" in results
    assert "mmlu_analysis" in results


def test_full_analysis_deterministic():
    """Same seed -> same analysis results."""
    r1 = run_full_analysis(seed=42)
    r2 = run_full_analysis(seed=42)
    # Check that nonlinearity scores match
    for task in r1["nonlinearity_scores"]:
        s1 = r1["nonlinearity_scores"][task]
        s2 = r2["nonlinearity_scores"][task]
        assert abs(s1["msi"] - s2["msi"]) < 1e-10
        assert abs(s1["msi_ci_lower"] - s2["msi_ci_lower"]) < 1e-10
        assert abs(s1["msi_ci_upper"] - s2["msi_ci_upper"]) < 1e-10
        assert abs(s1["artifact_probability"] - s2["artifact_probability"]) < 1e-10
