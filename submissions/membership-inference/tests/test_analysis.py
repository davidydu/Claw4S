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
