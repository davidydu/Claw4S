"""Tests for statistical analysis and reporting."""

import os
import json
import tempfile
import numpy as np
from src.analysis import compute_correlations, generate_report, save_results


def _make_mock_results():
    """Create mock results for testing analysis functions."""
    return [
        {
            "hidden_width": 16,
            "n_params": 261,
            "mean_attack_auc": 0.52,
            "std_attack_auc": 0.01,
            "mean_attack_accuracy": 0.51,
            "std_attack_accuracy": 0.01,
            "mean_overfit_gap": 0.05,
            "std_overfit_gap": 0.01,
            "mean_train_acc": 0.85,
            "mean_test_acc": 0.80,
            "repeats": [],
        },
        {
            "hidden_width": 64,
            "n_params": 1029,
            "mean_attack_auc": 0.60,
            "std_attack_auc": 0.02,
            "mean_attack_accuracy": 0.58,
            "std_attack_accuracy": 0.02,
            "mean_overfit_gap": 0.15,
            "std_overfit_gap": 0.02,
            "mean_train_acc": 0.95,
            "mean_test_acc": 0.80,
            "repeats": [],
        },
        {
            "hidden_width": 256,
            "n_params": 4101,
            "mean_attack_auc": 0.70,
            "std_attack_auc": 0.03,
            "mean_attack_accuracy": 0.65,
            "std_attack_accuracy": 0.03,
            "mean_overfit_gap": 0.30,
            "std_overfit_gap": 0.03,
            "mean_train_acc": 0.99,
            "mean_test_acc": 0.69,
            "repeats": [],
        },
    ]


def test_compute_correlations_keys():
    """Correlations contain expected keys."""
    results = _make_mock_results()
    corrs = compute_correlations(results)
    assert "auc_vs_log_params" in corrs
    assert "auc_vs_overfit_gap" in corrs
    assert "gap_vs_log_params" in corrs


def test_compute_correlations_values():
    """Correlation r-values are in [-1, 1]."""
    results = _make_mock_results()
    corrs = compute_correlations(results)
    for key, c in corrs.items():
        assert -1.0 <= c["r"] <= 1.0, f"{key}: r={c['r']}"
        assert 0.0 <= c["p"] <= 1.0, f"{key}: p={c['p']}"


def test_compute_correlations_positive():
    """Mock data should show positive correlations."""
    results = _make_mock_results()
    corrs = compute_correlations(results)
    # AUC increases with gap in our mock data
    assert corrs["auc_vs_overfit_gap"]["r"] > 0


def test_generate_report_creates_file():
    """Report generation creates a markdown file."""
    results = _make_mock_results()
    corrs = compute_correlations(results)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.md")
        generate_report(results, corrs, path)
        assert os.path.isfile(path)
        with open(path) as f:
            content = f.read()
        assert "Membership Inference" in content
        assert "Attack AUC" in content


def test_generate_report_uses_cautious_language_for_nonsignificant_predictors():
    """Report should not overclaim when both AUC correlations are non-significant."""
    results = _make_mock_results()
    corrs = {
        "auc_vs_log_params": {
            "r": 0.7434,
            "p": 0.1498,
            "description": "Attack AUC vs log2(parameter count)",
        },
        "auc_vs_overfit_gap": {
            "r": 0.7821,
            "p": 0.1181,
            "description": "Attack AUC vs overfitting gap",
        },
        "gap_vs_log_params": {
            "r": 0.9579,
            "p": 0.0103,
            "description": "Overfitting gap vs log2(parameter count)",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.md")
        generate_report(results, corrs, path)
        with open(path) as f:
            content = f.read()

    assert "neither association is statistically significant" in content
    assert "supports the thesis" not in content


def test_save_results_creates_valid_json():
    """Results are saved as valid JSON with expected structure."""
    results = _make_mock_results()
    corrs = compute_correlations(results)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.json")
        save_results(results, corrs, path)
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert "results" in data
        assert "correlations" in data
        assert "config" in data
        assert len(data["results"]) == 3


def test_compute_correlations_single_width_is_safe():
    """Single-width input should not crash Pearson correlation computation."""
    single = _make_mock_results()[:1]
    corrs = compute_correlations(single)
    for key in ("auc_vs_log_params", "auc_vs_overfit_gap", "gap_vs_log_params"):
        assert corrs[key]["r"] == 0.0
        assert corrs[key]["p"] == 1.0


def test_save_results_uses_supplied_config():
    """save_results should preserve run-specific config instead of hardcoded defaults."""
    results = _make_mock_results()
    corrs = compute_correlations(results)
    custom_config = {
        "n_samples": 600,
        "n_features": 12,
        "n_classes": 3,
        "hidden_widths": [16, 64, 256],
        "n_shadow_models": 5,
        "n_repeats": 7,
        "seed": 123,
        "train_fraction": 0.4,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.json")
        save_results(results, corrs, path, config=custom_config)
        with open(path) as f:
            data = json.load(f)

    assert data["config"] == custom_config
