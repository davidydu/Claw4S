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


def bootstrap_alpha_ci(
    param_counts: np.ndarray,
    losses_by_size: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Estimate uncertainty of alpha via nonparametric bootstrap.

    For each bootstrap replicate, this samples one loss value (with replacement)
    for every model size and refits the scaling law. The resulting alpha samples
    are summarized with mean/std and a 95% percentile interval.

    Args:
        param_counts: Array of model sizes with shape (n_sizes,).
        losses_by_size: Matrix of observed losses with shape (n_sizes, n_trials),
            where each row contains repeated runs for one model size.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for deterministic resampling.

    Returns:
        Dictionary with bootstrap statistics for alpha.

    Raises:
        ValueError: If inputs have invalid shapes.
        RuntimeError: If too few bootstrap fits converge.
    """
    param_counts = np.asarray(param_counts, dtype=np.float64)
    losses_by_size = np.asarray(losses_by_size, dtype=np.float64)

    if losses_by_size.ndim != 2:
        raise ValueError(
            f"losses_by_size must be 2D (n_sizes, n_trials), got shape {losses_by_size.shape}"
        )
    if param_counts.ndim != 1:
        raise ValueError(
            f"param_counts must be 1D (n_sizes,), got shape {param_counts.shape}"
        )
    if losses_by_size.shape[0] != param_counts.shape[0]:
        raise ValueError(
            "Mismatch between number of sizes in param_counts and losses_by_size: "
            f"{param_counts.shape[0]} vs {losses_by_size.shape[0]}"
        )
    if losses_by_size.shape[1] < 1:
        raise ValueError("losses_by_size must include at least one trial per size")
    if n_bootstrap < 10:
        raise ValueError("n_bootstrap must be at least 10")

    rng = np.random.RandomState(seed)
    n_sizes, n_trials = losses_by_size.shape
    alpha_samples = []

    for _ in range(n_bootstrap):
        sampled = np.empty(n_sizes, dtype=np.float64)
        sampled_indices = rng.randint(0, n_trials, size=n_sizes)
        for size_idx in range(n_sizes):
            sampled[size_idx] = losses_by_size[size_idx, sampled_indices[size_idx]]

        try:
            fit = fit_scaling_law(param_counts, sampled)
            alpha_samples.append(fit["alpha"])
        except RuntimeError:
            # Skip occasional non-convergent replicates.
            continue

    if len(alpha_samples) < 10:
        raise RuntimeError(
            "Too few successful bootstrap fits for alpha interval estimation "
            f"({len(alpha_samples)} successful out of {n_bootstrap})."
        )

    alpha_arr = np.array(alpha_samples, dtype=np.float64)
    ci_low, ci_high = np.percentile(alpha_arr, [2.5, 97.5])

    return {
        "alpha_mean": float(np.mean(alpha_arr)),
        "alpha_std": float(np.std(alpha_arr)),
        "alpha_ci95_low": float(ci_low),
        "alpha_ci95_high": float(ci_high),
        "n_bootstrap": int(len(alpha_arr)),
    }
