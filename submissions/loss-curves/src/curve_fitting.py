"""Curve fitting: fit parameterized functions to loss curves."""

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Functional forms
# ---------------------------------------------------------------------------


def power_law(t: np.ndarray, a: float, beta: float, L_inf: float) -> np.ndarray:
    """L(t) = a * t^(-beta) + L_inf"""
    return a * np.power(t, -beta) + L_inf


def exponential(t: np.ndarray, a: float, lam: float, L_inf: float) -> np.ndarray:
    """L(t) = a * exp(-lambda * t) + L_inf"""
    return a * np.exp(-lam * t) + L_inf


def stretched_exponential(
    t: np.ndarray, a: float, tau: float, gamma: float, L_inf: float
) -> np.ndarray:
    """L(t) = a * exp(-(t/tau)^gamma) + L_inf"""
    return a * np.exp(-np.power(t / tau, gamma)) + L_inf


def log_power(t: np.ndarray, a: float, beta: float, L_inf: float) -> np.ndarray:
    """L(t) = a * (log(t))^(-beta) + L_inf"""
    return a * np.power(np.log(t), -beta) + L_inf


# ---------------------------------------------------------------------------
# Model definitions for fitting
# ---------------------------------------------------------------------------

FUNCTIONAL_FORMS: dict[str, dict] = {
    "power_law": {
        "func": power_law,
        "p0": [1.0, 0.5, 0.01],
        "bounds": ([0, 0, -np.inf], [np.inf, 10, np.inf]),
        "n_params": 3,
        "param_names": ["a", "beta", "L_inf"],
    },
    "exponential": {
        "func": exponential,
        "p0": [1.0, 0.001, 0.01],
        "bounds": ([0, 0, -np.inf], [np.inf, 1, np.inf]),
        "n_params": 3,
        "param_names": ["a", "lambda", "L_inf"],
    },
    "stretched_exp": {
        "func": stretched_exponential,
        "p0": [1.0, 500.0, 0.5, 0.01],
        "bounds": ([0, 1, 0.01, -np.inf], [np.inf, 1e6, 5, np.inf]),
        "n_params": 4,
        "param_names": ["a", "tau", "gamma", "L_inf"],
    },
    "log_power": {
        "func": log_power,
        "p0": [1.0, 1.0, 0.01],
        "bounds": ([0, 0, -np.inf], [np.inf, 20, np.inf]),
        "n_params": 3,
        "param_names": ["a", "beta", "L_inf"],
    },
}


# ---------------------------------------------------------------------------
# Fitting + model selection
# ---------------------------------------------------------------------------


def compute_aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion.

    AIC = n * ln(RSS/n) + 2k
    """
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def compute_bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion.

    BIC = n * ln(RSS/n) + k * ln(n)
    """
    if rss <= 0:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def fit_single_curve(
    t: np.ndarray,
    losses: np.ndarray,
    form_name: str,
    max_retries: int = 3,
) -> dict:
    """Fit a single functional form to a loss curve.

    Returns a dict with: form, params, param_names, rss, aic, bic, converged.
    """
    form = FUNCTIONAL_FORMS[form_name]
    func = form["func"]
    n = len(t)
    k = form["n_params"]

    # Try fitting with multiple initial guesses
    best_result = None
    best_rss = np.inf

    # Generate varied initial guesses
    p0_base = np.array(form["p0"], dtype=float)
    rng = np.random.default_rng(42)

    for attempt in range(max_retries):
        if attempt == 0:
            p0 = p0_base.copy()
        else:
            # Perturb initial guess
            p0 = p0_base * (1 + 0.5 * rng.standard_normal(len(p0_base)))
            p0 = np.clip(
                p0,
                np.array(form["bounds"][0]) + 1e-8,
                np.array(form["bounds"][1]) - 1e-8,
            )

        try:
            popt, _ = curve_fit(
                func,
                t,
                losses,
                p0=p0,
                bounds=form["bounds"],
                maxfev=10000,
            )
            predicted = func(t, *popt)
            rss = float(np.sum((losses - predicted) ** 2))

            if rss < best_rss:
                best_rss = rss
                best_result = popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return {
            "form": form_name,
            "params": {},
            "param_names": form["param_names"],
            "rss": np.inf,
            "aic": np.inf,
            "bic": np.inf,
            "converged": False,
        }

    param_dict = dict(zip(form["param_names"], [float(v) for v in best_result]))

    return {
        "form": form_name,
        "params": param_dict,
        "param_names": form["param_names"],
        "rss": best_rss,
        "aic": compute_aic(n, k, best_rss),
        "bic": compute_bic(n, k, best_rss),
        "converged": True,
    }


def fit_all_forms(
    epochs: list[int],
    losses: list[float],
    skip_epochs: int = 10,
) -> list[dict]:
    """Fit all functional forms to a loss curve (after initial transient).

    Returns a list of fit result dicts, sorted by AIC (best first).
    """
    t = np.array(epochs[skip_epochs:], dtype=float)
    y = np.array(losses[skip_epochs:], dtype=float)

    results = []
    for form_name in FUNCTIONAL_FORMS:
        result = fit_single_curve(t, y, form_name)
        results.append(result)

    # Sort by AIC (lowest = best)
    results.sort(key=lambda r: r["aic"])
    return results
