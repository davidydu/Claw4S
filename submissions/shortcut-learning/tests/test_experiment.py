"""Tests for the experiment runner (fast, minimal configs)."""

from src.experiment import _compute_findings, run_single


def test_run_single_returns_all_keys():
    """run_single should return a dict with all required metric keys."""
    result = run_single(hidden_dim=32, weight_decay=0.0, seed=42)
    required_keys = [
        "hidden_dim", "weight_decay", "seed",
        "train_acc", "test_acc_with_shortcut",
        "test_acc_without_shortcut", "shortcut_reliance",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_run_single_accuracy_ranges():
    """All accuracies should be in [0, 1]."""
    result = run_single(hidden_dim=32, weight_decay=0.0, seed=42)
    for key in ["train_acc", "test_acc_with_shortcut", "test_acc_without_shortcut"]:
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"


def test_shortcut_reliance_is_consistent():
    """shortcut_reliance should equal test_with - test_without."""
    result = run_single(hidden_dim=64, weight_decay=0.0, seed=42)
    expected = round(result["test_acc_with_shortcut"] - result["test_acc_without_shortcut"], 4)
    assert result["shortcut_reliance"] == expected, \
        f"Reliance {result['shortcut_reliance']} != {expected}"


def test_compute_findings_reports_tied_best_configurations():
    """The best-reliance finding should not pretend a tie has a single winner."""
    aggregates = [
        {
            "hidden_dim": 32,
            "weight_decay": 0.0,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.5,
        },
        {
            "hidden_dim": 64,
            "weight_decay": 0.0,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.5,
        },
        {
            "hidden_dim": 128,
            "weight_decay": 0.0,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.5,
        },
        {
            "hidden_dim": 32,
            "weight_decay": 0.001,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.5,
        },
        {
            "hidden_dim": 64,
            "weight_decay": 0.001,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.5,
        },
        {
            "hidden_dim": 128,
            "weight_decay": 0.001,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.5,
        },
        {
            "hidden_dim": 32,
            "weight_decay": 1.0,
            "train_acc_mean": 0.51,
            "test_acc_with_mean": 0.5,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.0,
        },
        {
            "hidden_dim": 64,
            "weight_decay": 1.0,
            "train_acc_mean": 0.51,
            "test_acc_with_mean": 0.5,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.0,
        },
        {
            "hidden_dim": 128,
            "weight_decay": 1.0,
            "train_acc_mean": 0.51,
            "test_acc_with_mean": 0.5,
            "test_acc_without_mean": 0.5,
            "shortcut_reliance_mean": 0.0,
        },
        {
            "hidden_dim": 32,
            "weight_decay": 0.01,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.49,
            "shortcut_reliance_mean": 0.51,
        },
        {
            "hidden_dim": 64,
            "weight_decay": 0.01,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.49,
            "shortcut_reliance_mean": 0.51,
        },
        {
            "hidden_dim": 128,
            "weight_decay": 0.01,
            "train_acc_mean": 1.0,
            "test_acc_with_mean": 1.0,
            "test_acc_without_mean": 0.49,
            "shortcut_reliance_mean": 0.51,
        },
        {
            "hidden_dim": 32,
            "weight_decay": 0.1,
            "train_acc_mean": 0.99,
            "test_acc_with_mean": 0.99,
            "test_acc_without_mean": 0.62,
            "shortcut_reliance_mean": 0.37,
        },
        {
            "hidden_dim": 64,
            "weight_decay": 0.1,
            "train_acc_mean": 0.99,
            "test_acc_with_mean": 0.99,
            "test_acc_without_mean": 0.62,
            "shortcut_reliance_mean": 0.37,
        },
        {
            "hidden_dim": 128,
            "weight_decay": 0.1,
            "train_acc_mean": 0.99,
            "test_acc_with_mean": 0.99,
            "test_acc_without_mean": 0.62,
            "shortcut_reliance_mean": 0.37,
        },
    ]

    findings = _compute_findings(aggregates)

    tied_best = [f for f in findings if "Lowest shortcut reliance" in f]
    assert tied_best, "Expected a finding about the minimum shortcut reliance"
    assert "tied across 3 configurations" in tied_best[0]
    assert "(hidden_dim=32, weight_decay=1.0)" in tied_best[0]
    assert "(hidden_dim=64, weight_decay=1.0)" in tied_best[0]
    assert "(hidden_dim=128, weight_decay=1.0)" in tied_best[0]
