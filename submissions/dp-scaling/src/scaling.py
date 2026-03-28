"""Power-law curve fitting for scaling law analysis.

Fits the parametric form L(N) = a * N^(-alpha) + L_inf to observed
(parameter_count, test_loss) data, and computes goodness-of-fit metrics.
"""

import numpy as np
from scipy.optimize import curve_fit


def power_law(n: np.ndarray, a: float, alpha: float, l_inf: float) -> np.ndarray:
    """Power law model: L(N) = a * N^(-alpha) + L_inf.

    Args:
        n: Array of model sizes (parameter counts).
        a: Scaling coefficient.
        alpha: Scaling exponent (positive means loss decreases with size).
        l_inf: Irreducible loss (asymptotic floor).

    Returns:
        Predicted loss values.
    """
    return a * np.power(n, -alpha) + l_inf


def fit_scaling_law(
    param_counts: np.ndarray,
    losses: np.ndarray,
) -> dict:
    """Fit a power law L(N) = a * N^(-alpha) + L_inf to data.

    Uses scipy's trust-region reflective solver with explicit bounds and
    carefully chosen initial guesses to ensure convergence.

    Args:
        param_counts: Array of model sizes (parameter counts).
        losses: Array of corresponding test losses.

    Returns:
        Dictionary with keys:
            'a': Scaling coefficient.
            'alpha': Scaling exponent.
            'l_inf': Irreducible loss floor.
            'r_squared': Coefficient of determination.
            'residuals': Array of (predicted - observed) residuals.
    """
    # Initial guesses
    p0 = [
        float(np.max(losses) - np.min(losses)),  # a: range of losses
        0.3,  # alpha: typical scaling exponent
        float(np.min(losses)),  # l_inf: minimum observed loss
    ]

    # Bounds: a > 0, alpha > 0, l_inf >= 0
    bounds = ([0.0, 0.0, 0.0], [np.inf, 5.0, np.inf])

    try:
        popt, _pcov = curve_fit(
            power_law,
            param_counts,
            losses,
            p0=p0,
            bounds=bounds,
            method="trf",
            maxfev=10000,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Power law fitting failed to converge: {e}. "
            f"Data: N={param_counts.tolist()}, L={losses.tolist()}"
        ) from e

    a, alpha, l_inf = popt
    predicted = power_law(param_counts, a, alpha, l_inf)
    residuals = predicted - losses

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((losses - np.mean(losses)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": float(a),
        "alpha": float(alpha),
        "l_inf": float(l_inf),
        "r_squared": float(r_squared),
        "residuals": residuals.tolist(),
    }
