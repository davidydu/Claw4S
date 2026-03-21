"""Non-linear least squares fitting for scaling laws with bootstrap CIs.

Provides:
  - fit_scaling_law: multi-restart NLS via scipy.optimize.curve_fit
  - parametric_bootstrap: residual-based bootstrap for confidence intervals
  - FitResult: dataclass holding fit diagnostics (params, AIC, BIC, adj-R²)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from src.scaling_models import (
    FORMULATIONS,
    compute_aic,
    compute_bic,
    adjusted_r_squared,
)


@dataclass
class FitResult:
    """Container for a single scaling-law fit."""

    name: str
    params: dict[str, float]
    param_values: np.ndarray
    residuals: np.ndarray
    y_pred: np.ndarray
    adj_r_squared: float
    aic: float
    bic: float
    converged: bool


# Reasonable sampling ranges for initial guesses, keyed by parameter name.
_INIT_RANGES: dict[str, tuple[float, float]] = {
    "a": (0.1, 100.0),
    "alpha": (0.01, 0.5),
    "l_inf": (0.5, 3.0),
    "b": (0.1, 100.0),
    "beta": (0.01, 0.5),
    "c": (0.1, 50.0),
    "gamma": (0.01, 0.5),
}


def _sample_p0(param_names: list[str], rng: np.random.RandomState) -> list[float]:
    """Sample one set of initial parameter guesses."""
    return [rng.uniform(*_INIT_RANGES[p]) for p in param_names]


def fit_scaling_law(
    name: str,
    n: np.ndarray,
    y: np.ndarray,
    d: np.ndarray | None = None,
    n_restarts: int = 10,
    seed: int = 42,
) -> FitResult:
    """Fit a named scaling-law formulation to (n, y) data.

    Uses multiple random restarts and keeps the fit with lowest RSS.
    Returns FitResult with converged=False if all restarts fail.
    """
    formulation = FORMULATIONS[name]
    func = formulation["func"]
    param_names = formulation["param_names"]
    needs_d = formulation["needs_d"]
    k = formulation["n_params"]

    # For Chinchilla (needs_d), wrap the function so curve_fit only sees n
    # as the independent variable, with d bound via closure.
    if needs_d:
        if d is None:
            raise ValueError(f"Formulation '{name}' requires d but d=None")

        def fit_func(n_arg, *params):
            return func(n_arg, d, *params)
    else:
        fit_func = func

    rng = np.random.RandomState(seed)
    best_popt = None
    best_rss = np.inf

    for _ in range(n_restarts):
        p0 = _sample_p0(param_names, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                popt, _ = curve_fit(fit_func, n, y, p0=p0, maxfev=10000)
            y_pred = fit_func(n, *popt)
            rss = float(np.sum((y - y_pred) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_popt = popt
        except (RuntimeError, OptimizeWarning):
            continue

    # All restarts failed — return an unconverged placeholder.
    if best_popt is None:
        print(f"  WARNING: All {n_restarts} restarts failed for '{name}'. Returning unconverged result.")
        return FitResult(
            name=name,
            params={p: float("nan") for p in param_names},
            param_values=np.full(k, np.nan),
            residuals=np.full(len(y), np.nan),
            y_pred=np.full(len(y), np.nan),
            adj_r_squared=float("nan"),
            aic=float("nan"),
            bic=float("nan"),
            converged=False,
        )

    y_pred = fit_func(n, *best_popt)
    residuals = y - y_pred
    n_obs = len(y)

    return FitResult(
        name=name,
        params=dict(zip(param_names, best_popt.tolist())),
        param_values=best_popt,
        residuals=residuals,
        y_pred=y_pred,
        adj_r_squared=adjusted_r_squared(y, y_pred, k),
        aic=compute_aic(n_obs, k, best_rss),
        bic=compute_bic(n_obs, k, best_rss),
        converged=True,
    )


def parametric_bootstrap(
    name: str,
    n: np.ndarray,
    fit_result: FitResult,
    n_bootstrap: int = 1000,
    seed: int = 42,
    d: np.ndarray | None = None,
) -> dict[str, tuple[float, float] | float]:
    """Parametric bootstrap for confidence intervals on fitted parameters.

    Generates synthetic data by adding N(0, residual_std) noise to the fitted
    curve, refits each synthetic dataset, and computes 95% CIs (2.5th-97.5th
    percentiles).  Returns a dict mapping each parameter name to (lower, upper)
    plus a ``"convergence_rate"`` key.
    """
    formulation = FORMULATIONS[name]
    param_names = formulation["param_names"]

    rng = np.random.RandomState(seed)
    residual_std = float(np.std(fit_result.residuals, ddof=1))
    # Guard against zero residual_std (perfect fit): use a tiny noise floor.
    if residual_std == 0 or np.isnan(residual_std):
        residual_std = 1e-10

    # Collect bootstrap parameter estimates.
    boot_params: list[np.ndarray] = []
    n_converged = 0

    for i in range(n_bootstrap):
        noise = rng.normal(0, residual_std, len(n))
        y_synth = fit_result.y_pred + noise
        boot_seed = seed + i + 1
        result = fit_scaling_law(name, n, y_synth, d=d, n_restarts=5, seed=boot_seed)
        if not result.converged:
            continue
        n_converged += 1
        # Filter degenerate fits.
        pdict = result.params
        alpha_val = pdict.get("alpha", 0.5)
        l_inf_val = pdict.get("l_inf", 1.0)
        if alpha_val < 0 or alpha_val > 1 or l_inf_val < 0:
            continue
        boot_params.append(result.param_values)

    convergence_rate = n_converged / n_bootstrap if n_bootstrap > 0 else 0.0

    if len(boot_params) < 2:
        # Not enough valid bootstrap fits for CIs.
        ci: dict[str, tuple[float, float] | float] = {
            p: (float("nan"), float("nan")) for p in param_names
        }
        ci["convergence_rate"] = convergence_rate
        return ci

    boot_matrix = np.array(boot_params)  # (n_valid, n_params)
    ci = {}
    for j, pname in enumerate(param_names):
        lower = float(np.percentile(boot_matrix[:, j], 2.5))
        upper = float(np.percentile(boot_matrix[:, j], 97.5))
        ci[pname] = (lower, upper)
    ci["convergence_rate"] = convergence_rate
    return ci
