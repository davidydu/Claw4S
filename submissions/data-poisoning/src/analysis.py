"""Analysis: aggregate results, fit sigmoid curves, and export results."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit

from src.experiment import ExperimentConfig, RunResult


@dataclass
class AggregatedPoint:
    """Aggregated metrics for one (poison_fraction, hidden_width) pair."""

    poison_fraction: float
    hidden_width: int
    test_acc_mean: float
    test_acc_std: float
    train_acc_mean: float
    train_acc_std: float
    train_clean_acc_mean: float
    train_clean_acc_std: float
    gen_gap_mean: float
    gen_gap_std: float
    n_seeds: int


@dataclass
class SigmoidFit:
    """Parameters of fitted sigmoid: acc = L / (1 + exp(k * (x - x0))) + b"""

    hidden_width: int
    L: float       # curve maximum
    k: float       # steepness
    x0: float      # midpoint (critical threshold)
    b: float       # baseline offset
    r_squared: float
    threshold_midpoint: float  # poison fraction where accuracy drops to midpoint of clean and chance


def aggregate_results(results: list[RunResult]) -> list[AggregatedPoint]:
    """Aggregate run results by (poison_fraction, hidden_width), computing mean and std.

    Args:
        results: List of individual run results.

    Returns:
        List of AggregatedPoint objects, sorted by (hidden_width, poison_fraction).
    """
    groups: dict[tuple[float, int], list[RunResult]] = {}
    for r in results:
        key = (r.poison_fraction, r.hidden_width)
        groups.setdefault(key, []).append(r)

    agg = []
    for (pf, hw), runs in sorted(groups.items()):
        test_accs = [r.test_accuracy for r in runs]
        train_accs = [r.train_accuracy for r in runs]
        train_clean_accs = [r.train_clean_accuracy for r in runs]
        gen_gaps = [r.generalization_gap for r in runs]

        agg.append(AggregatedPoint(
            poison_fraction=pf,
            hidden_width=hw,
            test_acc_mean=float(np.mean(test_accs)),
            test_acc_std=float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
            train_acc_mean=float(np.mean(train_accs)),
            train_acc_std=float(np.std(train_accs, ddof=1)) if len(train_accs) > 1 else 0.0,
            train_clean_acc_mean=float(np.mean(train_clean_accs)),
            train_clean_acc_std=float(np.std(train_clean_accs, ddof=1)) if len(train_clean_accs) > 1 else 0.0,
            gen_gap_mean=float(np.mean(gen_gaps)),
            gen_gap_std=float(np.std(gen_gaps, ddof=1)) if len(gen_gaps) > 1 else 0.0,
            n_seeds=len(runs),
        ))

    return agg


def _sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Descending sigmoid: y = L / (1 + exp(k * (x - x0))) + b"""
    return L / (1.0 + np.exp(k * (x - x0))) + b


def fit_sigmoid_curve(
    agg_points: list[AggregatedPoint],
    hidden_width: int,
    chance_acc: float = 0.2,
    threshold_search_max: float | None = None,
) -> SigmoidFit:
    """Fit a descending sigmoid to test accuracy vs. poison fraction for one model size.

    Args:
        agg_points: Aggregated data points.
        hidden_width: Which model size to fit.
        chance_acc: Chance-level accuracy for the classification task.
        threshold_search_max: Optional max poison fraction for threshold search.

    Returns:
        SigmoidFit with fitted parameters and R-squared.

    Raises:
        RuntimeError: If curve fitting fails to converge.
    """
    pts = [p for p in agg_points if p.hidden_width == hidden_width]
    pts.sort(key=lambda p: p.poison_fraction)

    x = np.array([p.poison_fraction for p in pts])
    y = np.array([p.test_acc_mean for p in pts])

    # Initial guesses: L ~ range of y, k ~ 10 (moderate steepness), x0 ~ 0.2, b ~ min(y)
    y_max = y.max()
    y_min = y.min()
    p0 = [y_max - y_min, 10.0, 0.2, y_min]
    bounds = ([0, 0.1, 0.0, 0.0], [1.5, 100.0, 1.0, 1.0])

    try:
        popt, _ = curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Sigmoid fit failed for hidden_width={hidden_width}: {exc}"
        ) from exc

    L, k, x0, b = popt

    # R-squared
    y_pred = _sigmoid(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if not (0.0 < chance_acc < 1.0):
        raise ValueError(f"chance_acc must be in (0, 1), got {chance_acc}")

    # Find threshold where accuracy drops to midpoint between clean and chance.
    clean_acc = y[0]  # poison_fraction=0
    target_acc = (clean_acc + chance_acc) / 2.0

    # Search numerically over a configurable range.
    observed_max_poison = float(np.max(x)) if len(x) else 0.0
    if threshold_search_max is not None:
        search_max = float(threshold_search_max)
    else:
        search_max = min(1.0, max(0.6, observed_max_poison + 0.1))

    if search_max <= 0:
        raise ValueError(f"threshold_search_max must be positive, got {search_max}")

    x_search = np.linspace(0.0, search_max, 2000)
    y_search = _sigmoid(x_search, *popt)
    idx_below = np.where(y_search <= target_acc)[0]
    if len(idx_below) > 0:
        threshold_midpoint = float(x_search[idx_below[0]])
    else:
        threshold_midpoint = float("inf")  # never drops that low in range

    return SigmoidFit(
        hidden_width=hidden_width,
        L=float(L),
        k=float(k),
        x0=float(x0),
        b=float(b),
        r_squared=float(r_squared),
        threshold_midpoint=threshold_midpoint,
    )


def compute_findings(
    agg_points: list[AggregatedPoint],
    fits: list[SigmoidFit],
) -> dict:
    """Derive key scientific findings from the analysis.

    Args:
        agg_points: Aggregated experiment results.
        fits: Sigmoid fits for each model size.

    Returns:
        Dictionary of findings.
    """
    findings = {}

    hidden_widths = sorted({p.hidden_width for p in agg_points})
    if not hidden_widths:
        hidden_widths = sorted({f.hidden_width for f in fits})

    poison_fractions = sorted({p.poison_fraction for p in agg_points})
    baseline_poison = poison_fractions[0] if poison_fractions else 0.0
    max_poison = poison_fractions[-1] if poison_fractions else 0.5

    # 1. Critical thresholds per model size
    thresholds = {f.hidden_width: f.threshold_midpoint for f in fits}
    findings["critical_thresholds"] = thresholds

    # 2. Steepness (sharpness of transition) per model size
    steepness = {f.hidden_width: f.k for f in fits}
    findings["steepness_k"] = steepness

    # 3. Check if larger models are more sensitive
    sorted_fits = sorted(fits, key=lambda f: f.hidden_width)
    if len(sorted_fits) >= 2:
        threshs = [f.threshold_midpoint for f in sorted_fits]
        # If threshold decreases with width, larger models are more sensitive
        findings["larger_models_more_sensitive"] = all(
            threshs[i] >= threshs[i + 1] for i in range(len(threshs) - 1)
        )
    else:
        findings["larger_models_more_sensitive"] = None

    # 4. Phase transition sharpness: is k > 5 for any model? (indicates sharp transition)
    findings["sharp_transition"] = any(f.k > 5.0 for f in fits)

    # 5. R-squared values (fit quality)
    findings["r_squared"] = {f.hidden_width: f.r_squared for f in fits}

    # 6. Generalization gap at high poison
    high_poison_gaps = {}
    for hw in hidden_widths:
        pts_hw = [p for p in agg_points if p.hidden_width == hw and p.poison_fraction == max_poison]
        if pts_hw:
            high_poison_gaps[hw] = pts_hw[0].gen_gap_mean
    findings["gen_gap_at_max_poison"] = {
        "poison_fraction": max_poison,
        "values": high_poison_gaps,
    }

    # Backward-compatible key for the current default sweep.
    high_poison_50pct = {}
    for hw in hidden_widths:
        pts_hw = [p for p in agg_points if p.hidden_width == hw and p.poison_fraction == 0.5]
        if pts_hw:
            high_poison_50pct[hw] = pts_hw[0].gen_gap_mean
    findings["gen_gap_at_50pct_poison"] = high_poison_50pct

    # 7. Clean accuracy (baseline)
    clean_accs = {}
    for hw in hidden_widths:
        pts_hw = [p for p in agg_points if p.hidden_width == hw and p.poison_fraction == baseline_poison]
        if pts_hw:
            clean_accs[hw] = pts_hw[0].test_acc_mean
    findings["clean_test_accuracy"] = clean_accs

    return findings


def build_results_payload(
    config: ExperimentConfig,
    results: list[RunResult],
    agg_points: list[AggregatedPoint],
    fits: list[SigmoidFit],
    findings: dict,
) -> dict:
    """Build a deterministic scientific results payload.

    Timing metadata is intentionally excluded so repeated runs with the same
    seeds produce the same scientific artifact.
    """
    serialized_runs = []
    for result in results:
        run_data = asdict(result)
        run_data.pop("elapsed_seconds", None)
        serialized_runs.append(run_data)

    return {
        "config": asdict(config),
        "runs": serialized_runs,
        "aggregated": [asdict(point) for point in agg_points],
        "sigmoid_fits": [asdict(fit) for fit in fits],
        "findings": findings,
        "metadata": {
            "total_runs": len(results),
            "n_poison_fractions": len(config.poison_fractions),
            "n_hidden_widths": len(config.hidden_widths),
            "n_seeds": len(config.seeds),
        },
    }


def build_performance_payload(
    results: list[RunResult],
    total_time_seconds: float,
) -> dict:
    """Build a performance payload with non-deterministic runtime metadata."""
    run_times = [result.elapsed_seconds for result in results]
    if run_times:
        mean_time = float(np.mean(run_times))
        min_time = float(np.min(run_times))
        max_time = float(np.max(run_times))
    else:
        mean_time = 0.0
        min_time = 0.0
        max_time = 0.0

    return {
        "total_time_seconds": float(total_time_seconds),
        "n_runs": len(results),
        "mean_run_time_seconds": mean_time,
        "min_run_time_seconds": min_time,
        "max_run_time_seconds": max_time,
    }
