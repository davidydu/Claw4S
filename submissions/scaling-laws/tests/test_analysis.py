# tests/test_analysis.py
import numpy as np
from src.analysis import (
    fit_bounded_power_law,
    fit_sigmoid,
    detect_breakpoint,
    run_loss_scaling,
    run_task_scaling,
    run_cross_metric_correlation,
    run_extrapolation_risk,
    run_cross_family_transfer,
    run_full_analysis,
)


# --- Synthetic unit tests (fast, test fitting mechanics) ---

def test_fit_bounded_power_law_recovers_params():
    """Bounded power-law acc(N) = 1 - a*N^(-alpha) should recover known params."""
    np.random.seed(42)
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    true_a, true_alpha = 50.0, 0.05
    y = 1.0 - true_a * np.power(n, -true_alpha)
    y += np.random.normal(0, 0.005, len(y))
    result = fit_bounded_power_law(n, y)
    assert result["converged"]
    assert abs(result["params"]["alpha"] - true_alpha) < 0.02


def test_fit_sigmoid_recovers_params():
    """Sigmoid acc(N) = L / (1 + exp(-k*(log(N) - x0))) should recover shape."""
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    log_n = np.log(n)
    y = 0.8 / (1.0 + np.exp(-1.5 * (log_n - np.log(1e9))))
    result = fit_sigmoid(n, y)
    assert result["converged"]
    assert 0 < result["params"]["L"] < 1.0


def test_detect_breakpoint_finds_planted_break():
    """Breakpoint detection should find a change in slope at the planted location."""
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    log_n = np.log(n)
    y = np.where(log_n < np.log(5e9), 0.01 * log_n, 0.05 * log_n - 0.5)
    result = detect_breakpoint(n, y)
    assert "breakpoint_idx" in result
    assert result["breakpoint_idx"] in [3, 4]


# --- Integration tests (use real data, slower) ---

def test_run_loss_scaling_returns_expected_structure():
    """Loss scaling should return fits for all three formulations."""
    from src.data import CEREBRAS_GPT
    result = run_loss_scaling(CEREBRAS_GPT, n_bootstrap=50, seed=42)
    assert "kaplan" in result
    assert "chinchilla" in result
    assert "corrected" in result
    for name, fit in result.items():
        assert "params" in fit
        assert "adj_r_squared" in fit
        assert "aic" in fit
        assert "bic" in fit


def test_run_task_scaling_returns_per_task_results():
    """Task scaling should return results for each benchmark."""
    from src.data import CEREBRAS_GPT
    result = run_task_scaling(CEREBRAS_GPT, n_bootstrap=50, seed=42)
    assert len(result) >= 5
    for task_name, task_result in result.items():
        assert "bounded_power_law" in task_result
        assert "sigmoid" in task_result
        assert "breakpoint" in task_result


def test_run_full_analysis_returns_all_phases():
    """Full analysis should return results for all 5 phases."""
    result = run_full_analysis(n_bootstrap=50, seed=42)
    assert "loss_scaling" in result
    assert "task_scaling" in result
    assert "cross_metric" in result
    assert "extrapolation" in result
    assert "cross_family" in result
    assert "metadata" in result
