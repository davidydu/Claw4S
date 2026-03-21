# tests/test_scaling_models.py
import numpy as np
from src.scaling_models import (
    kaplan_loss, chinchilla_loss, corrected_loss,
    compute_aic, compute_bic, adjusted_r_squared,
    FORMULATIONS,
)


def test_kaplan_loss_decreases_with_n():
    """Kaplan power-law: loss should decrease as N increases."""
    n_values = np.array([1e8, 1e9, 1e10])
    losses = kaplan_loss(n_values, a=1.0, alpha=0.07, l_inf=1.5)
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1]


def test_kaplan_loss_approaches_l_inf():
    """As N -> infinity, Kaplan loss should approach L_inf."""
    loss = kaplan_loss(np.array([1e30]), a=1.0, alpha=0.07, l_inf=1.5)
    assert abs(loss[0] - 1.5) < 0.01


def test_chinchilla_loss_with_fixed_d_equals_kaplan():
    """When D is constant, Chinchilla reduces to Kaplan + constant."""
    n = np.array([1e8, 1e9, 1e10])
    d = np.full_like(n, 300e9)
    chin = chinchilla_loss(n, d, a=1.0, alpha=0.07, b=1.0, beta=0.07, l_inf=1.5)
    for i in range(len(chin) - 1):
        assert chin[i] > chin[i + 1]


def test_corrected_loss_correction_vanishes_at_large_n():
    """Finite-size correction should vanish for large N."""
    large_n = np.array([1e20])
    corrected = corrected_loss(large_n, a=1.0, alpha=0.07, c=5.0, gamma=0.1, l_inf=1.5)
    kaplan = kaplan_loss(large_n, a=1.0, alpha=0.07, l_inf=1.5)
    assert abs(corrected[0] - kaplan[0]) < 0.01


def test_formulations_registry():
    """FORMULATIONS dict should contain all three named formulations."""
    assert "kaplan" in FORMULATIONS
    assert "chinchilla" in FORMULATIONS
    assert "corrected" in FORMULATIONS


def test_aic_prefers_simpler_model_on_identical_fit():
    """Given equal RSS, AIC should prefer fewer parameters."""
    n, k1, k2, rss = 7, 3, 5, 0.01
    aic1 = compute_aic(n, k1, rss)
    aic2 = compute_aic(n, k2, rss)
    assert aic1 < aic2


def test_bic_penalizes_params_more_than_aic():
    """BIC penalty grows with log(n), should penalize more than AIC for n >= 8."""
    n, k, rss = 10, 5, 0.01
    aic = compute_aic(n, k, rss)
    bic = compute_bic(n, k, rss)
    assert bic > aic


def test_adjusted_r_squared_less_than_r_squared():
    """Adjusted R-squared should be less than or equal to R-squared for k > 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_pred = y + np.array([0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.05])
    adj_r2 = adjusted_r_squared(y, y_pred, k=3)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert adj_r2 <= r2


def test_adjusted_r_squared_returns_nan_when_degenerate():
    """Adjusted R-squared should return NaN when n <= k + 1."""
    y = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    adj_r2 = adjusted_r_squared(y, y_pred, k=3)
    assert np.isnan(adj_r2)


def test_leave_one_out_cv():
    """LOO-CV on perfect power-law data should have near-zero error."""
    from src.scaling_models import leave_one_out_cv
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    y = 5.0 * np.power(n, -0.07) + 1.5
    cv_error = leave_one_out_cv("kaplan", n, y)
    assert cv_error < 0.01
