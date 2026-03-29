"""Tests for result-structure validation helpers."""

from src.validation import (
    EXPECTED_HIDDEN_DIMS,
    EXPECTED_SEEDS,
    EXPECTED_WEIGHT_DECAYS,
    collect_validation_errors,
)


def _build_valid_results() -> dict:
    runs = []
    for hidden_dim in EXPECTED_HIDDEN_DIMS:
        for weight_decay in EXPECTED_WEIGHT_DECAYS:
            for seed in EXPECTED_SEEDS:
                runs.append(
                    {
                        "hidden_dim": hidden_dim,
                        "weight_decay": weight_decay,
                        "seed": seed,
                        "train_acc": 0.9,
                        "test_acc_with_shortcut": 0.8,
                        "test_acc_without_shortcut": 0.6,
                        "shortcut_reliance": 0.2,
                    }
                )

    aggregates = []
    for hidden_dim in EXPECTED_HIDDEN_DIMS:
        for weight_decay in EXPECTED_WEIGHT_DECAYS:
            aggregates.append(
                {
                    "hidden_dim": hidden_dim,
                    "weight_decay": weight_decay,
                    "n_seeds": len(EXPECTED_SEEDS),
                    "train_acc_mean": 0.9,
                    "train_acc_std": 0.05,
                    "test_acc_with_mean": 0.8,
                    "test_acc_with_std": 0.04,
                    "test_acc_without_mean": 0.6,
                    "test_acc_without_std": 0.03,
                    "shortcut_reliance_mean": 0.2,
                    "shortcut_reliance_std": 0.02,
                }
            )

    return {
        "metadata": {
            "n_configs": len(runs),
            "hidden_dims": EXPECTED_HIDDEN_DIMS,
            "weight_decays": EXPECTED_WEIGHT_DECAYS,
            "seeds": EXPECTED_SEEDS,
            "n_genuine_features": 10,
            "n_total_features": 11,
            "n_train": 2000,
            "n_test": 1000,
            "epochs": 100,
            "lr": 0.01,
            "batch_size": 128,
            "elapsed_seconds": 1.0,
        },
        "individual_runs": runs,
        "aggregates": aggregates,
        "findings": ["finding 1", "finding 2"],
    }


def test_collect_validation_errors_accepts_consistent_payload():
    """A complete, non-duplicated payload should have zero validation errors."""
    data = _build_valid_results()
    errors = collect_validation_errors(data)
    assert errors == []


def test_collect_validation_errors_reports_missing_run_configuration():
    """Validation should fail when a required (hidden_dim, wd, seed) triple is missing."""
    data = _build_valid_results()
    data["individual_runs"].pop()
    errors = collect_validation_errors(data)
    assert any("Missing individual run configurations" in e for e in errors)


def test_collect_validation_errors_reports_duplicate_aggregate_configuration():
    """Validation should fail when aggregate (hidden_dim, wd) pairs are duplicated."""
    data = _build_valid_results()
    data["aggregates"].append(dict(data["aggregates"][0]))
    errors = collect_validation_errors(data)
    assert any("Duplicate aggregate configurations" in e for e in errors)
