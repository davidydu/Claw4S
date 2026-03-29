"""Tests for src/analysis.py."""

from src.analysis import compute_sparsity_generalization_correlation


def _mock_experiment(task: str, hidden_dim: int, initial_zero: float,
                     final_zero: float, final_train_acc: float, final_test_acc: float) -> dict:
    """Build a minimal experiment dict compatible with analysis helpers."""
    return {
        "task": task,
        "hidden_dim": hidden_dim,
        "history": {
            "dead_neuron_fraction": [0.0, 0.0],
            "near_dead_fraction": [0.2, 0.2],
            "zero_fraction": [initial_zero, final_zero],
            "activation_entropy": [1.5, 1.4],
            "mean_activation_magnitude": [0.8, 0.9],
            "train_acc": [0.5, final_train_acc],
            "test_acc": [0.4, final_test_acc],
        },
    }


def test_correlation_includes_uncertainty_metadata():
    """Pooled correlation stats include sample size and bootstrap CI bounds."""
    experiments = [
        _mock_experiment("modular_addition_mod97", 32, 0.60, 0.45, 1.00, 0.55),
        _mock_experiment("modular_addition_mod97", 64, 0.55, 0.40, 1.00, 0.25),
        _mock_experiment("modular_addition_mod97", 128, 0.52, 0.42, 1.00, 0.35),
        _mock_experiment("modular_addition_mod97", 256, 0.50, 0.43, 1.00, 0.72),
        _mock_experiment("nonlinear_regression", 32, 0.54, 0.60, 0.97, 0.92),
        _mock_experiment("nonlinear_regression", 64, 0.50, 0.57, 0.94, 0.84),
        _mock_experiment("nonlinear_regression", 128, 0.48, 0.55, 0.92, 0.78),
        _mock_experiment("nonlinear_regression", 256, 0.46, 0.53, 0.90, 0.74),
    ]

    result = compute_sparsity_generalization_correlation(experiments)
    corr = result["correlations"]["zero_frac_vs_gen_gap"]

    assert corr["n"] == 8
    assert "ci_low" in corr
    assert "ci_high" in corr
    assert corr["ci_low"] <= corr["rho"] <= corr["ci_high"]


def test_correlation_reports_task_stratified_stats():
    """Task-specific correlations are emitted alongside pooled metrics."""
    experiments = [
        _mock_experiment("modular_addition_mod97", 32, 0.60, 0.45, 1.00, 0.55),
        _mock_experiment("modular_addition_mod97", 64, 0.55, 0.40, 1.00, 0.25),
        _mock_experiment("modular_addition_mod97", 128, 0.52, 0.42, 1.00, 0.35),
        _mock_experiment("modular_addition_mod97", 256, 0.50, 0.43, 1.00, 0.72),
        _mock_experiment("nonlinear_regression", 32, 0.54, 0.60, 0.97, 0.92),
        _mock_experiment("nonlinear_regression", 64, 0.50, 0.57, 0.94, 0.84),
        _mock_experiment("nonlinear_regression", 128, 0.48, 0.55, 0.92, 0.78),
        _mock_experiment("nonlinear_regression", 256, 0.46, 0.53, 0.90, 0.74),
    ]

    result = compute_sparsity_generalization_correlation(experiments)
    by_task = result["correlations_by_task"]

    assert set(by_task) == {"modular_addition_mod97", "nonlinear_regression"}
    assert by_task["modular_addition_mod97"]["zero_frac_vs_test_acc"]["n"] == 4
    assert by_task["nonlinear_regression"]["zero_frac_vs_test_acc"]["n"] == 4
