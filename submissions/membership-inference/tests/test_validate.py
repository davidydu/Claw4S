"""Tests for validate.py data consistency checks."""

from validate import validate_results_payload


def _payload(hidden_widths, n_repeats):
    results = []
    for width in hidden_widths:
        repeats = [
            {
                "repeat": rep,
                "hidden_width": width,
                "train_acc": 0.8,
                "test_acc": 0.7,
                "overfit_gap": 0.1,
                "attack_auc": 0.55,
                "attack_accuracy": 0.52,
            }
            for rep in range(n_repeats)
        ]
        results.append(
            {
                "hidden_width": width,
                "n_params": (10 * width + width) + (width * 5 + 5),
                "mean_attack_auc": 0.55,
                "std_attack_auc": 0.01,
                "mean_attack_accuracy": 0.52,
                "std_attack_accuracy": 0.01,
                "mean_overfit_gap": 0.1,
                "std_overfit_gap": 0.01,
                "mean_train_acc": 0.8,
                "mean_test_acc": 0.7,
                "repeats": repeats,
            }
        )

    correlations = {
        "auc_vs_log_params": {"r": 0.5, "p": 0.2, "description": "AUC vs size"},
        "auc_vs_overfit_gap": {"r": 0.6, "p": 0.1, "description": "AUC vs gap"},
        "gap_vs_log_params": {"r": 0.7, "p": 0.05, "description": "Gap vs size"},
    }
    config = {
        "n_samples": 500,
        "n_features": 10,
        "n_classes": 5,
        "hidden_widths": hidden_widths,
        "n_shadow_models": 3,
        "n_repeats": n_repeats,
        "seed": 42,
        "train_fraction": 0.5,
    }
    return {"results": results, "correlations": correlations, "config": config}


def test_validate_results_payload_accepts_custom_widths_and_repeats():
    """Validation should follow values from config instead of hardcoded defaults."""
    payload = _payload(hidden_widths=[8, 16], n_repeats=2)
    errors = validate_results_payload(payload)
    assert errors == []


def test_validate_results_payload_rejects_repeat_count_mismatch():
    """Validation should fail if per-width repeats do not match config."""
    payload = _payload(hidden_widths=[8, 16], n_repeats=2)
    payload["results"][0]["repeats"] = payload["results"][0]["repeats"][:1]
    errors = validate_results_payload(payload)
    assert any("expected 2 repeats, got 1" in err for err in errors)
