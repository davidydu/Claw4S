"""Five-phase analysis pipeline for neural scaling laws.

Phases:
  1. Loss scaling — fit Kaplan, Chinchilla, corrected formulations
  2. Task scaling — bounded power-law, sigmoid, breakpoint per benchmark
  3. Cross-metric correlation — loss vs. accuracy deltas
  4. Extrapolation risk — train on small, predict large
  5. Cross-family transfer — fit on one family, predict another
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning

from src.data import (
    CEREBRAS_GPT,
    PYTHIA,
    get_family_data,
    get_training_tokens,
    get_benchmark_keys,
)
from src.scaling_models import FORMULATIONS, adjusted_r_squared
from src.fitting import fit_scaling_law, parametric_bootstrap


# ---------------------------------------------------------------------------
# Task-level fitting helpers
# ---------------------------------------------------------------------------

def fit_bounded_power_law(n: np.ndarray, y: np.ndarray) -> dict:
    """Fit acc(N) = 1 - a * N^(-alpha).

    Returns dict with keys: params, adj_r_squared, converged, y_pred.
    """
    def _model(x, a, alpha):
        return 1.0 - a * np.power(x, -alpha)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                _model, n, y,
                p0=[50.0, 0.05],
                bounds=([0.0, 0.0], [np.inf, 1.0]),
                maxfev=10000,
            )
        y_pred = _model(n, *popt)
        k = 2
        return {
            "params": {"a": float(popt[0]), "alpha": float(popt[1])},
            "adj_r_squared": float(adjusted_r_squared(y, y_pred, k)),
            "converged": True,
            "y_pred": y_pred,
        }
    except (RuntimeError, ValueError, OptimizeWarning):
        return {
            "params": {"a": float("nan"), "alpha": float("nan")},
            "adj_r_squared": float("nan"),
            "converged": False,
            "y_pred": np.full(len(y), np.nan),
        }


def fit_sigmoid(n: np.ndarray, y: np.ndarray) -> dict:
    """Fit acc(N) = L / (1 + exp(-k * (log(N) - x0))).

    Returns dict with keys: params, adj_r_squared, converged, y_pred.
    """
    log_n = np.log(n)

    def _model(x, L, k, x0):
        return L / (1.0 + np.exp(-k * (np.log(x) - x0)))

    # Initial guess: L near max(y), k moderate, x0 near midpoint of log(n)
    L0 = min(float(np.max(y)) * 1.2, 1.0)
    k0 = 1.0
    x0_init = float(np.mean(log_n))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                _model, n, y,
                p0=[L0, k0, x0_init],
                bounds=([0.0, 0.0, -np.inf], [1.0, np.inf, np.inf]),
                maxfev=10000,
            )
        y_pred = _model(n, *popt)
        k_params = 3
        return {
            "params": {"L": float(popt[0]), "k": float(popt[1]), "x0": float(popt[2])},
            "adj_r_squared": float(adjusted_r_squared(y, y_pred, k_params)),
            "converged": True,
            "y_pred": y_pred,
        }
    except (RuntimeError, ValueError, OptimizeWarning):
        return {
            "params": {"L": float("nan"), "k": float("nan"), "x0": float("nan")},
            "adj_r_squared": float("nan"),
            "converged": False,
            "y_pred": np.full(len(y), np.nan),
        }


def detect_breakpoint(n: np.ndarray, y: np.ndarray) -> dict:
    """Find breakpoint via piecewise-linear regression in log-space.

    Tests every possible split point (indices 2..len(n)-2) ensuring
    at least 2 data points per segment. Returns the split with minimum
    total RSS.
    """
    log_n = np.log(n)
    best_idx = None
    best_rss = np.inf
    best_left_slope = 0.0
    best_right_slope = 0.0

    for i in range(2, len(n) - 1):
        # Left segment: indices [0, i)
        x_left = log_n[:i]
        y_left = y[:i]
        # Right segment: indices [i, end)
        x_right = log_n[i:]
        y_right = y[i:]

        if len(x_left) < 2 or len(x_right) < 2:
            continue

        # Fit linear regression to each segment
        sl, il, _, _, _ = stats.linregress(x_left, y_left)
        sr, ir, _, _, _ = stats.linregress(x_right, y_right)

        rss_left = float(np.sum((y_left - (sl * x_left + il)) ** 2))
        rss_right = float(np.sum((y_right - (sr * x_right + ir)) ** 2))
        total_rss = rss_left + rss_right

        if total_rss < best_rss:
            best_rss = total_rss
            best_idx = i
            best_left_slope = float(sl)
            best_right_slope = float(sr)

    return {
        "breakpoint_idx": best_idx,
        "total_rss": best_rss,
        "left_slope": best_left_slope,
        "right_slope": best_right_slope,
    }


# ---------------------------------------------------------------------------
# Phase 1: Loss scaling
# ---------------------------------------------------------------------------

def run_loss_scaling(
    family: dict, n_bootstrap: int = 1000, seed: int = 42
) -> dict:
    """Fit Kaplan, Chinchilla, and corrected scaling laws to loss data."""
    n_params, losses = get_family_data(family, "pile_test_loss")
    d = get_training_tokens(family)

    results = {}
    for form_name, form_spec in FORMULATIONS.items():
        d_arg = d if form_spec["needs_d"] else None
        fit = fit_scaling_law(form_name, n_params, losses, d=d_arg, seed=seed)
        ci = parametric_bootstrap(
            form_name, n_params, fit,
            n_bootstrap=n_bootstrap, seed=seed, d=d_arg,
        )
        results[form_name] = {
            "params": fit.params,
            "ci": ci,
            "adj_r_squared": fit.adj_r_squared,
            "aic": fit.aic,
            "bic": fit.bic,
        }
    return results


# ---------------------------------------------------------------------------
# Phase 2: Task scaling
# ---------------------------------------------------------------------------

def run_task_scaling(
    family: dict, n_bootstrap: int = 1000, seed: int = 42
) -> dict:
    """Fit bounded power-law, sigmoid, and breakpoint detection per benchmark."""
    benchmarks = get_benchmark_keys(family)
    results = {}

    for bench in benchmarks:
        n_params, acc = get_family_data(family, bench)
        bpl = fit_bounded_power_law(n_params, acc)
        sig = fit_sigmoid(n_params, acc)
        bp = detect_breakpoint(n_params, acc)

        # Bootstrap CIs for bounded power-law
        if bpl["converged"]:
            residuals = acc - bpl["y_pred"]
            res_std = float(np.std(residuals, ddof=1))
            if res_std == 0 or np.isnan(res_std):
                res_std = 1e-10
            rng = np.random.RandomState(seed)
            boot_a, boot_alpha = [], []
            for i in range(n_bootstrap):
                noise = rng.normal(0, res_std, len(n_params))
                y_synth = bpl["y_pred"] + noise
                boot_fit = fit_bounded_power_law(n_params, y_synth)
                if boot_fit["converged"]:
                    boot_a.append(boot_fit["params"]["a"])
                    boot_alpha.append(boot_fit["params"]["alpha"])
            if len(boot_a) >= 2:
                bpl["ci"] = {
                    "a": (float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5))),
                    "alpha": (float(np.percentile(boot_alpha, 2.5)), float(np.percentile(boot_alpha, 97.5))),
                }
            else:
                bpl["ci"] = {
                    "a": (float("nan"), float("nan")),
                    "alpha": (float("nan"), float("nan")),
                }

        results[bench] = {
            "bounded_power_law": bpl,
            "sigmoid": sig,
            "breakpoint": bp,
        }
    return results


# ---------------------------------------------------------------------------
# Phase 3: Cross-metric correlation
# ---------------------------------------------------------------------------

def run_cross_metric_correlation(family: dict) -> dict:
    """Correlate delta-loss with delta-accuracy between consecutive sizes."""
    # Only works for families with pile_test_loss
    n_params, losses = get_family_data(family, "pile_test_loss")
    benchmarks = get_benchmark_keys(family)

    # Compute delta-loss (negative so that "better" is positive)
    delta_loss = -np.diff(losses)  # loss decreases, so negate for direction

    # Average delta-accuracy across all benchmarks
    delta_accs = []
    for bench in benchmarks:
        _, acc = get_family_data(family, bench)
        delta_accs.append(np.diff(acc))
    delta_acc = np.mean(delta_accs, axis=0)

    n_pairs = len(delta_loss)
    if n_pairs < 3:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "n_pairs": n_pairs,
        }

    pr, pp = stats.pearsonr(delta_loss, delta_acc)
    sr, sp = stats.spearmanr(delta_loss, delta_acc)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n_pairs": int(n_pairs),
    }


# ---------------------------------------------------------------------------
# Phase 4: Extrapolation risk
# ---------------------------------------------------------------------------

def run_extrapolation_risk(
    family: dict, n_train: int = 4, n_bootstrap: int = 1000, seed: int = 42
) -> dict:
    """Fit on the n_train smallest models; predict the rest. Measure MAPE."""
    # Loss extrapolation (Kaplan formulation)
    n_params, losses = get_family_data(family, "pile_test_loss")
    n_tr = n_params[:n_train]
    y_tr = losses[:n_train]
    n_test = n_params[n_train:]
    y_test = losses[n_train:]

    kaplan_fit = fit_scaling_law("kaplan", n_tr, y_tr, seed=seed)
    if kaplan_fit.converged:
        from src.scaling_models import kaplan_loss
        y_pred_test = kaplan_loss(n_test, *kaplan_fit.param_values)
        loss_mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)
        loss_predictions = {
            str(int(n_test[i])): {"actual": float(y_test[i]), "predicted": float(y_pred_test[i])}
            for i in range(len(n_test))
        }
    else:
        loss_mape = float("nan")
        loss_predictions = {}

    # Task accuracy extrapolation
    benchmarks = get_benchmark_keys(family)
    task_mapes = []
    task_predictions = {}
    for bench in benchmarks:
        np_arr, acc = get_family_data(family, bench)
        np_tr_b = np_arr[:n_train]
        acc_tr = acc[:n_train]
        np_test_b = np_arr[n_train:]
        acc_test = acc[n_train:]

        bpl = fit_bounded_power_law(np_tr_b, acc_tr)
        if bpl["converged"]:
            def _bpl_model(x, a, alpha):
                return 1.0 - a * np.power(x, -alpha)

            acc_pred = _bpl_model(np_test_b, bpl["params"]["a"], bpl["params"]["alpha"])
            # Avoid division by zero in MAPE
            nonzero = acc_test != 0
            if np.any(nonzero):
                mape = float(np.mean(np.abs((acc_test[nonzero] - acc_pred[nonzero]) / acc_test[nonzero])) * 100)
            else:
                mape = float("nan")
            task_mapes.append(mape)
            task_predictions[bench] = {
                str(int(np_test_b[i])): {"actual": float(acc_test[i]), "predicted": float(acc_pred[i])}
                for i in range(len(np_test_b))
            }

    task_mape_avg = float(np.mean(task_mapes)) if task_mapes else float("nan")

    return {
        "loss_mape": loss_mape,
        "task_mape_avg": task_mape_avg,
        "loss_predictions": loss_predictions,
        "task_predictions": task_predictions,
    }


# ---------------------------------------------------------------------------
# Phase 5: Cross-family transfer
# ---------------------------------------------------------------------------

def run_cross_family_transfer(
    primary: dict, secondary: dict,
    n_bootstrap: int = 1000, seed: int = 42,
) -> dict:
    """Fit bounded power-law on primary family, predict secondary family accuracy."""
    primary_benchmarks = set(get_benchmark_keys(primary))
    secondary_benchmarks = set(get_benchmark_keys(secondary))
    overlapping = sorted(primary_benchmarks & secondary_benchmarks)

    transfer_errors = {}
    for bench in overlapping:
        n_prim, acc_prim = get_family_data(primary, bench)
        n_sec, acc_sec = get_family_data(secondary, bench)

        bpl = fit_bounded_power_law(n_prim, acc_prim)
        if not bpl["converged"]:
            transfer_errors[bench] = float("nan")
            continue

        def _bpl_model(x, a, alpha):
            return 1.0 - a * np.power(x, -alpha)

        acc_pred = _bpl_model(n_sec, bpl["params"]["a"], bpl["params"]["alpha"])
        nonzero = acc_sec != 0
        if np.any(nonzero):
            mape = float(np.mean(np.abs((acc_sec[nonzero] - acc_pred[nonzero]) / acc_sec[nonzero])) * 100)
        else:
            mape = float("nan")
        transfer_errors[bench] = mape

    valid_errors = [v for v in transfer_errors.values() if not (isinstance(v, float) and np.isnan(v))]
    avg_error = float(np.mean(valid_errors)) if valid_errors else float("nan")

    return {
        "overlapping_tasks": overlapping,
        "transfer_errors": transfer_errors,
        "avg_transfer_error": avg_error,
    }


# ---------------------------------------------------------------------------
# JSON serializer for numpy types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Run all 5 analysis phases. Save results to results/results.json."""
    import scipy

    os.makedirs("results/figures", exist_ok=True)

    results: dict = {}

    # Phase 1
    print("[1/5] Fitting loss scaling laws...")
    try:
        results["loss_scaling"] = run_loss_scaling(CEREBRAS_GPT, n_bootstrap=n_bootstrap, seed=seed)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        import traceback
        print(f"  ERROR in phase: {e}")
        traceback.print_exc()
        results["loss_scaling"] = {"error": str(e)}

    # Phase 2
    print("[2/5] Fitting task-level scaling curves...")
    try:
        results["task_scaling"] = run_task_scaling(CEREBRAS_GPT, n_bootstrap=n_bootstrap, seed=seed)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        import traceback
        print(f"  ERROR in phase: {e}")
        traceback.print_exc()
        results["task_scaling"] = {"error": str(e)}

    # Phase 3
    print("[3/5] Computing cross-metric correlations...")
    try:
        results["cross_metric"] = run_cross_metric_correlation(CEREBRAS_GPT)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        import traceback
        print(f"  ERROR in phase: {e}")
        traceback.print_exc()
        results["cross_metric"] = {"error": str(e)}

    # Phase 4
    print("[4/5] Evaluating extrapolation risk...")
    try:
        results["extrapolation"] = run_extrapolation_risk(CEREBRAS_GPT, n_bootstrap=n_bootstrap, seed=seed)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        import traceback
        print(f"  ERROR in phase: {e}")
        traceback.print_exc()
        results["extrapolation"] = {"error": str(e)}

    # Phase 5
    print("[5/5] Testing cross-family transfer...")
    try:
        results["cross_family"] = run_cross_family_transfer(
            CEREBRAS_GPT, PYTHIA, n_bootstrap=n_bootstrap, seed=seed,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        import traceback
        print(f"  ERROR in phase: {e}")
        traceback.print_exc()
        results["cross_family"] = {"error": str(e)}

    # Metadata
    results["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }

    # Save
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print("Results saved to results/results.json")

    return results
