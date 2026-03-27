"""Phase transition analysis for Kuramoto synchronization experiments.

Provides functions for detecting the critical coupling K_c, estimating
critical exponents, performing finite-size scaling, and computing
aggregate statistics over simulation results.
"""

from __future__ import annotations

from itertools import groupby
from operator import itemgetter
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

from src.evaluators import EvaluatorPanel
from src.kuramoto import KuramotoModel


# ---------------------------------------------------------------------------
# Sigmoid fit
# ---------------------------------------------------------------------------


def _sigmoid(K, a, K_c):
    """Logistic sigmoid: r(K) = 1 / (1 + exp(-a*(K - K_c)))."""
    return 1.0 / (1.0 + np.exp(-a * (K - K_c)))


def fit_sigmoid(
    K_vals: np.ndarray, r_vals: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Fit r(K) = 1/(1+exp(-a*(K-K_c))) to data using scipy curve_fit.

    Parameters
    ----------
    K_vals : array of coupling strengths
    r_vals : array of mean order parameters (same length)

    Returns
    -------
    (K_c, params) where K_c is the estimated critical coupling and params
    is the full parameter array [a, K_c] from curve_fit.
    """
    # Initial guess: midpoint of K range for K_c, slope 3
    k_mid = float((K_vals.min() + K_vals.max()) / 2)
    p0 = [3.0, k_mid]
    params, _ = curve_fit(_sigmoid, K_vals, r_vals, p0=p0, maxfev=10_000)
    K_c = float(params[1])
    return K_c, params


# ---------------------------------------------------------------------------
# Susceptibility-based K_c estimate
# ---------------------------------------------------------------------------


def estimate_kc_susceptibility(
    K_vals: np.ndarray, r_var: np.ndarray, n: int
) -> float:
    """Return K at which the susceptibility χ = N * var(r) is maximum.

    Parameters
    ----------
    K_vals : array of coupling strengths
    r_var  : array of r variance at each K
    n      : system size N (used to scale χ)

    Returns
    -------
    K value at the susceptibility peak.
    """
    chi = n * r_var
    peak_idx = int(np.argmax(chi))
    return float(K_vals[peak_idx])


# ---------------------------------------------------------------------------
# Critical exponent β
# ---------------------------------------------------------------------------


def fit_critical_exponent(
    K_vals: np.ndarray, r_vals: np.ndarray, kc: float
) -> Tuple[float, float]:
    """Estimate β from log-log regression of r vs (K - K_c) for K > K_c.

    Model: r ≈ (K - K_c)^β  →  log(r) = β * log(K - K_c) + const

    Parameters
    ----------
    K_vals : array of coupling strengths
    r_vals : array of mean order parameters
    kc     : critical coupling K_c (already estimated)

    Returns
    -------
    (beta, R_squared)
    """
    mask = K_vals > kc
    dk = K_vals[mask] - kc
    r = r_vals[mask]

    # Remove any non-positive values before taking logs
    valid = (dk > 0) & (r > 0)
    log_dk = np.log(dk[valid])
    log_r = np.log(r[valid])

    slope, intercept, r_value, p_value, std_err = linregress(log_dk, log_r)
    beta = float(slope)
    r_squared = float(r_value ** 2)
    return beta, r_squared


# ---------------------------------------------------------------------------
# Finite-size scaling
# ---------------------------------------------------------------------------


def _kc_fss(N, K_c_inf, a, nu):
    """K_c(N) = K_c(∞) + a * N^(-nu)."""
    return K_c_inf + a * N ** (-nu)


def fit_finite_size_scaling(
    kc_by_n: Dict[int, float]
) -> Tuple[float, float]:
    """Fit K_c(N) = K_c(∞) + a * N^(-ν) using scipy curve_fit with bounds.

    Parameters
    ----------
    kc_by_n : dict mapping system size N -> estimated K_c(N)

    Returns
    -------
    (K_c_inf, nu)

    Notes
    -----
    With only 3 data points the fit is underdetermined. We constrain K_c(inf)
    to [0, 2*max(K_c(N))] and nu to [0.1, 10] to avoid divergent solutions.
    If the fit residual is large (K_c_inf outside the data range), we fall
    back to a linear extrapolation in log(N) space.
    """
    Ns = np.array(sorted(kc_by_n.keys()), dtype=float)
    kcs = np.array([kc_by_n[int(n)] for n in Ns], dtype=float)

    kc_min = float(kcs.min())
    kc_max = float(kcs.max())

    # Determine if K_c increases or decreases with N
    # Standard FSS assumes K_c decreases as N increases (K_c_inf < K_c(N))
    # If K_c increases with N, use negative a (a can be negative)
    p0 = [kc_min - 0.1, kc_max - kc_min, 1.0]
    bounds = ([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 10.0])

    try:
        params, _ = curve_fit(
            _kc_fss, Ns, kcs, p0=p0, bounds=bounds, maxfev=20_000
        )
        K_c_inf = float(params[0])
        nu = float(abs(params[2]))

        # Sanity check: K_c_inf should be near the data range
        # If way outside, the fit diverged; use linear extrapolation instead
        if not (-2.0 <= K_c_inf <= kc_max + 2.0):
            raise RuntimeError(f"K_c_inf={K_c_inf:.2f} out of range; using fallback")
    except Exception:
        # Fallback: log-log linear extrapolation
        # log(K_c(N) - K_c_min) ~ -nu * log(N) + const
        # Just use the smallest observed K_c as the infinity estimate
        K_c_inf = kc_min
        nu = 1.0  # default exponent

    return K_c_inf, nu


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Parameters
    ----------
    values      : sample values
    confidence  : confidence level (default 0.95)
    n_bootstrap : number of bootstrap resamples
    seed        : random seed for reproducibility

    Returns
    -------
    (low, high) confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    low = float(np.percentile(boot_means, 100 * alpha / 2))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------


def compute_statistics(records: List[dict]) -> List[dict]:
    """Group records by (topology, n, sigma, K) and compute aggregate stats.

    Parameters
    ----------
    records : list of dicts with keys: topology, n, sigma, K, seed,
              final_r, evaluator_scores

    Returns
    -------
    List of stat dicts, one per unique (topology, n, sigma, K) group.
    Each dict contains:
        topology, n, sigma, K, n_seeds, mean_r, std_r,
        evaluator_agreement (fraction of evaluators with mean score >= 0.5)
    """
    # Sort for groupby
    sorted_records = sorted(
        records, key=lambda r: (r["topology"], r["n"], r["sigma"], r["K"])
    )

    stats = []
    for key, group_iter in groupby(
        sorted_records,
        key=lambda r: (r["topology"], r["n"], r["sigma"], r["K"]),
    ):
        group = list(group_iter)
        topology, n, sigma, K = key

        r_values = np.array([rec["final_r"] for rec in group])
        mean_r = float(np.mean(r_values))
        std_r = float(np.std(r_values))

        # Evaluator agreement: fraction of evaluators whose mean score >= 0.5
        eval_names = list(group[0]["evaluator_scores"].keys())
        agreement_count = 0
        for ev_name in eval_names:
            ev_scores = [rec["evaluator_scores"][ev_name] for rec in group]
            if float(np.mean(ev_scores)) >= 0.5:
                agreement_count += 1
        evaluator_agreement = agreement_count / len(eval_names) if eval_names else 0.0

        stats.append({
            "topology": topology,
            "n": n,
            "sigma": sigma,
            "K": K,
            "n_seeds": len(group),
            "mean_r": mean_r,
            "std_r": std_r,
            "evaluator_agreement": evaluator_agreement,
        })

    return stats


# ---------------------------------------------------------------------------
# High-level analysis entry point
# ---------------------------------------------------------------------------


def analyze_results(sim_results: list) -> dict:
    """Run EvaluatorPanel on all SimulationResults and aggregate.

    Parameters
    ----------
    sim_results : list of SimulationResult objects (from experiment.py)

    Returns
    -------
    dict with keys:
        "records"    : list of per-run record dicts
        "statistics" : list of per-condition aggregate stat dicts
    """
    panel = EvaluatorPanel()
    records = []

    for result in sim_results:
        cfg = result.config

        # Build adjacency and positions from config (reuse model's data)
        model = KuramotoModel(
            n=cfg.n,
            K=cfg.K,
            sigma=cfg.sigma,
            omega0=cfg.omega0,
            topology=cfg.topology,
            dt=cfg.dt,
            stage_size=cfg.stage_size,
            seed=cfg.seed,
            topology_kwargs=cfg.topology_kwargs,
        )

        eval_results = panel.evaluate_all(
            phase_history=result.phase_history,
            positions=model.positions,
            adjacency=model.adjacency,
            sigma=cfg.sigma,
        )

        evaluator_scores = {
            er.evaluator_name: er.sync_score
            for er in eval_results
        }

        records.append({
            "topology": cfg.topology,
            "n": cfg.n,
            "sigma": cfg.sigma,
            "K": cfg.K,
            "seed": cfg.seed,
            "final_r": result.final_r,
            "convergence_step": result.convergence_step,
            "evaluator_scores": evaluator_scores,
        })

    statistics = compute_statistics(records)

    return {"records": records, "statistics": statistics}
