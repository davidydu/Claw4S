"""Scoring functions and curve fitting for emergence analysis.

Implements the core insight from Schaeffer et al. (2023): discontinuous
metrics (exact match) can create apparent phase transitions from smooth
underlying per-token improvement. Continuous metrics (partial credit,
token edit distance) reveal the smooth improvement.
"""

import math
import numpy as np
from scipy.optimize import curve_fit


# ── Scoring functions ────────────────────────────────────────────────────────

def exact_match_from_token_accuracy(per_token_acc: float, n_tokens: int) -> float:
    """Compute exact-match accuracy from per-token accuracy.

    Assumes token-level independence: P(all correct) = p^n.
    This is the discontinuous metric that creates apparent emergence.

    Args:
        per_token_acc: Probability of each token being correct, in [0, 1].
        n_tokens: Number of tokens in the answer.

    Returns:
        Exact-match accuracy in [0, 1].
    """
    return per_token_acc ** n_tokens


def partial_credit_from_token_accuracy(per_token_acc: float, n_tokens: int) -> float:
    """Compute partial-credit score from per-token accuracy.

    Under token independence, partial credit = per-token accuracy.
    This is a continuous metric that reveals smooth improvement.

    Args:
        per_token_acc: Probability of each token being correct, in [0, 1].
        n_tokens: Number of tokens in the answer (unused but kept for API symmetry).

    Returns:
        Partial-credit score in [0, 1].
    """
    return per_token_acc


def token_edit_distance(per_token_acc: float, n_tokens: int) -> float:
    """Compute expected token edit distance from per-token accuracy.

    Expected edit distance = n * (1 - p), where p is per-token accuracy.
    Lower is better. This is a continuous metric.

    Args:
        per_token_acc: Probability of each token being correct, in [0, 1].
        n_tokens: Number of tokens in the answer.

    Returns:
        Expected token edit distance (0 = perfect, n = worst).
    """
    return n_tokens * (1.0 - per_token_acc)


def brier_score(predicted_prob: float, true_label: int) -> float:
    """Compute Brier score for a single prediction.

    Brier score = (predicted_prob - true_label)^2.
    Lower is better. 0 = perfect, 1 = worst.

    Args:
        predicted_prob: Predicted probability of positive class, in [0, 1].
        true_label: True label (0 or 1).

    Returns:
        Brier score in [0, 1].
    """
    return (predicted_prob - true_label) ** 2


# ── Curve fitting ────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Logistic sigmoid: y = L / (1 + exp(-k * (x - x0))) + b."""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def sigmoid_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[dict, float, np.ndarray]:
    """Fit a logistic sigmoid to (x, y) data.

    Args:
        x: Independent variable (e.g., log(parameters)).
        y: Dependent variable (e.g., accuracy).

    Returns:
        (params_dict, r_squared, residuals) where params_dict has keys
        'L', 'k', 'x0', 'b'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guesses
    L0 = float(np.max(y) - np.min(y))
    k0 = 1.0
    x0_guess = float(np.median(x))
    b0 = float(np.min(y))

    try:
        popt, _ = curve_fit(
            _sigmoid, x, y,
            p0=[L0, k0, x0_guess, b0],
            maxfev=10000,
            bounds=(
                [0, -10, float(np.min(x)) - 10, -1],
                [2, 10, float(np.max(x)) + 10, 1],
            ),
        )
    except RuntimeError:
        # Fallback: return poor fit
        popt = [L0, k0, x0_guess, b0]

    y_pred = _sigmoid(x, *popt)
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    params = {"L": popt[0], "k": popt[1], "x0": popt[2], "b": popt[3]}
    return params, r_squared, residuals


def linear_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[dict, float, np.ndarray]:
    """Fit a linear model y = slope * x + intercept.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        (params_dict, r_squared, residuals) where params_dict has keys
        'slope', 'intercept'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]

    y_pred = slope * x + intercept
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    params = {"slope": float(slope), "intercept": float(intercept)}
    return params, r_squared, residuals


# ── Model comparison ─────────────────────────────────────────────────────────

def compute_aic(n: int, rss: float, k: int) -> float:
    """Compute Akaike Information Criterion.

    AIC = n * ln(RSS/n) + 2k

    Args:
        n: Number of data points.
        rss: Residual sum of squares.
        k: Number of parameters.

    Returns:
        AIC value (lower is better).
    """
    return n * math.log(rss / n) + 2 * k


def compute_bic(n: int, rss: float, k: int) -> float:
    """Compute Bayesian Information Criterion.

    BIC = n * ln(RSS/n) + k * ln(n)

    Args:
        n: Number of data points.
        rss: Residual sum of squares.
        k: Number of parameters.

    Returns:
        BIC value (lower is better).
    """
    return n * math.log(rss / n) + k * math.log(n)
