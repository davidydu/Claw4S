"""Scaling law formulations and model selection utilities.

Three functional forms for L(N):
  - Kaplan power-law: L = a * N^(-alpha) + L_inf
  - Chinchilla two-term: L = a * N^(-alpha) + b * D^(-beta) + L_inf
  - Corrected power-law: L = a * N^(-alpha) * (1 + c * N^(-gamma)) + L_inf
"""
from __future__ import annotations

import numpy as np


# --- Scaling law formulations ---

def kaplan_loss(n: np.ndarray, a: float, alpha: float, l_inf: float) -> np.ndarray:
    """Kaplan power-law: L(N) = a * N^(-alpha) + L_inf."""
    return a * np.power(n, -alpha) + l_inf


def chinchilla_loss(
    n: np.ndarray, d: np.ndarray,
    a: float, alpha: float, b: float, beta: float, l_inf: float,
) -> np.ndarray:
    """Chinchilla two-term: L(N,D) = a * N^(-alpha) + b * D^(-beta) + L_inf."""
    return a * np.power(n, -alpha) + b * np.power(d, -beta) + l_inf


def corrected_loss(
    n: np.ndarray, a: float, alpha: float, c: float, gamma: float, l_inf: float,
) -> np.ndarray:
    """Power-law with finite-size correction:
    L(N) = a * N^(-alpha) * (1 + c * N^(-gamma)) + L_inf."""
    return a * np.power(n, -alpha) * (1.0 + c * np.power(n, -gamma)) + l_inf


# Registry of formulations for automated iteration
FORMULATIONS: dict[str, dict] = {
    "kaplan": {
        "func": kaplan_loss,
        "param_names": ["a", "alpha", "l_inf"],
        "n_params": 3,
        "needs_d": False,
    },
    "chinchilla": {
        "func": chinchilla_loss,
        "param_names": ["a", "alpha", "b", "beta", "l_inf"],
        "n_params": 5,
        "needs_d": True,
    },
    "corrected": {
        "func": corrected_loss,
        "param_names": ["a", "alpha", "c", "gamma", "l_inf"],
        "n_params": 5,
        "needs_d": False,
    },
}


# --- Model selection utilities ---

def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion. Lower is better.
    AIC = n * ln(RSS/n) + 2k
    """
    return n * np.log(rss / n) + 2 * k


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion. Lower is better.
    BIC = n * ln(RSS/n) + k * ln(n)
    """
    return n * np.log(rss / n) + k * np.log(n)


def adjusted_r_squared(y: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Adjusted R-squared that penalizes number of parameters k.
    Returns NaN if n <= k + 1 (degenerate case).
    """
    n = len(y)
    if n <= k + 1:
        return float("nan")
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)


def leave_one_out_cv(name: str, n: np.ndarray, y: np.ndarray, d: np.ndarray | None = None) -> float:
    """Leave-one-out cross-validation error for a scaling law formulation.
    Returns mean absolute prediction error across all LOO folds.
    """
    from src.fitting import fit_scaling_law
    errors = []
    for i in range(len(n)):
        n_train = np.delete(n, i)
        y_train = np.delete(y, i)
        d_train = np.delete(d, i) if d is not None else None
        result = fit_scaling_law(name, n_train, y_train, d=d_train)
        if result.converged:
            formulation = FORMULATIONS[name]
            if formulation["needs_d"] and d is not None:
                y_pred = formulation["func"](np.array([n[i]]), np.array([d[i]]), *result.param_values)
            else:
                y_pred = formulation["func"](np.array([n[i]]), *result.param_values)
            errors.append(abs(y_pred[0] - y[i]))
    return np.mean(errors) if errors else float("nan")
