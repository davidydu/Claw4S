"""Tests for report generation."""

from src.report import generate_report


def test_generate_report_warns_that_low_reliance_can_reflect_nonlearning():
    """Interpretation should explain that near-zero reliance is not always success."""
    results = {
        "metadata": {
            "n_configs": 1,
            "hidden_dims": [32],
            "weight_decays": [1.0],
            "seeds": [42],
            "n_genuine_features": 10,
            "n_total_features": 11,
            "n_train": 2000,
            "n_test": 1000,
            "elapsed_seconds": 1.0,
        },
        "aggregates": [
            {
                "hidden_dim": 32,
                "weight_decay": 1.0,
                "train_acc_mean": 0.506,
                "train_acc_std": 0.008,
                "test_acc_with_mean": 0.498,
                "test_acc_with_std": 0.015,
                "test_acc_without_mean": 0.498,
                "test_acc_without_std": 0.015,
                "shortcut_reliance_mean": 0.0,
                "shortcut_reliance_std": 0.0,
            }
        ],
        "findings": [
            "Weight decay=1.0 eliminates shortcut reliance but also prevents learning."
        ],
    }

    report = generate_report(results)

    assert "Near-zero shortcut reliance can also arise when the model fails to learn anything" in report
