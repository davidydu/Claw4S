# src/zipf_analysis.py
"""Zipf's law analysis: fitting, piecewise decomposition, breakpoint detection.

Implements Zipf-Mandelbrot fitting: f(r) = C / (r + q)^alpha
where r is rank, f(r) is frequency, and alpha, q, C are fitted parameters.

Uses OLS on log-transformed data for fitting, with grid search over q.
"""

import numpy as np
from collections import Counter


def compute_rank_frequency(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute rank-frequency distribution from token IDs.

    Returns (ranks, frequencies) sorted by frequency descending.
    Ranks start at 1.
    """
    if not token_ids:
        return np.array([], dtype=int), np.array([], dtype=int)

    counts = Counter(token_ids)
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    return ranks, freqs


def _ols_log_log(ranks: np.ndarray, freqs: np.ndarray, q: float) -> dict:
    """Fit log(f) = -alpha * log(r + q) + log(C) via OLS.

    Returns dict with alpha, C, r_squared, q.
    """
    log_r = np.log(ranks.astype(float) + q)
    log_f = np.log(freqs.astype(float))

    # OLS: log_f = slope * log_r + intercept
    # slope = -alpha, intercept = log(C)
    n = len(log_r)
    mean_x = np.mean(log_r)
    mean_y = np.mean(log_f)

    ss_xx = np.sum((log_r - mean_x) ** 2)
    ss_xy = np.sum((log_r - mean_x) * (log_f - mean_y))
    ss_yy = np.sum((log_f - mean_y) ** 2)

    if ss_xx < 1e-15:
        # Degenerate case: all ranks identical (shouldn't happen)
        return {"alpha": 0.0, "C": np.exp(mean_y), "r_squared": 0.0, "q": q}

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    # R^2
    ss_res = np.sum((log_f - (slope * log_r + intercept)) ** 2)
    r_squared = 1.0 - ss_res / ss_yy if ss_yy > 1e-15 else 0.0

    alpha = -slope  # f(r) = C * (r+q)^(-alpha), so log(f) = log(C) - alpha*log(r+q)
    C = np.exp(intercept)

    return {
        "alpha": float(alpha),
        "C": float(C),
        "r_squared": float(max(0.0, r_squared)),
        "q": float(q),
    }


def fit_zipf_mandelbrot(
    ranks: np.ndarray,
    freqs: np.ndarray,
    q_values: list[float] | None = None,
) -> dict:
    """Fit Zipf-Mandelbrot model: f(r) = C / (r + q)^alpha.

    Performs grid search over q values and returns the best fit
    (highest R^2).

    Returns dict with: alpha, q, r_squared, C
    """
    if len(ranks) == 0 or len(freqs) == 0:
        return {"alpha": 0.0, "q": 0.0, "r_squared": 0.0, "C": 0.0}

    if q_values is None:
        q_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Filter out zero frequencies (can't log-transform)
    mask = freqs > 0
    ranks_filtered = ranks[mask]
    freqs_filtered = freqs[mask]

    if len(ranks_filtered) < 2:
        return {"alpha": 0.0, "q": 0.0, "r_squared": 0.0, "C": 0.0}

    best = None
    for q in q_values:
        result = _ols_log_log(ranks_filtered, freqs_filtered, q)
        if best is None or result["r_squared"] > best["r_squared"]:
            best = result

    return best


def fit_piecewise_zipf(
    ranks: np.ndarray,
    freqs: np.ndarray,
) -> dict:
    """Fit Zipf exponent separately in head, body, and tail regions.

    - Head: top 10% of ranks (most frequent tokens)
    - Body: ranks 10%-90%
    - Tail: bottom 10% of ranks (least frequent tokens)

    Returns dict with head, body, tail sub-dicts, each containing alpha, r_squared.
    """
    n = len(ranks)
    if n < 10:
        # Not enough data for piecewise analysis
        fit = fit_zipf_mandelbrot(ranks, freqs, q_values=[0.0])
        return {
            "head": {"alpha": fit["alpha"], "r_squared": fit["r_squared"]},
            "body": {"alpha": fit["alpha"], "r_squared": fit["r_squared"]},
            "tail": {"alpha": fit["alpha"], "r_squared": fit["r_squared"]},
        }

    head_end = max(2, n // 10)
    tail_start = max(head_end + 1, n - n // 10)

    regions = {
        "head": (0, head_end),
        "body": (head_end, tail_start),
        "tail": (tail_start, n),
    }

    result = {}
    for region_name, (start, end) in regions.items():
        r = ranks[start:end]
        f = freqs[start:end]
        if len(r) < 2:
            result[region_name] = {"alpha": 0.0, "r_squared": 0.0}
        else:
            fit = fit_zipf_mandelbrot(r, f, q_values=[0.0])
            result[region_name] = {
                "alpha": fit["alpha"],
                "r_squared": fit["r_squared"],
            }

    return result


def detect_breakpoints(
    ranks: np.ndarray,
    freqs: np.ndarray,
    window_size: int = 50,
    threshold: float = 0.3,
) -> list[int]:
    """Detect breakpoints where local Zipf exponent changes significantly.

    Slides a window across the rank-frequency distribution and fits
    alpha locally. Reports ranks where the local alpha deviates from
    the global alpha by more than threshold.

    Returns list of rank values where breakpoints occur.
    """
    n = len(ranks)
    if n < window_size * 2:
        return []

    # Global fit
    global_fit = fit_zipf_mandelbrot(ranks, freqs, q_values=[0.0])
    global_alpha = global_fit["alpha"]

    breakpoints = []
    step = max(1, window_size // 2)

    prev_in_deviation = False
    for i in range(0, n - window_size, step):
        r_window = ranks[i : i + window_size]
        f_window = freqs[i : i + window_size]

        # Filter zero frequencies
        mask = f_window > 0
        if mask.sum() < 3:
            continue

        local_fit = _ols_log_log(r_window[mask], f_window[mask], 0.0)
        local_alpha = local_fit["alpha"]

        deviation = abs(local_alpha - global_alpha)
        in_deviation = deviation > threshold

        # Report the rank where we transition into a deviation zone
        if in_deviation and not prev_in_deviation:
            breakpoints.append(int(ranks[i]))

        prev_in_deviation = in_deviation

    return breakpoints


def analyze_corpus(token_ids: list[int], label: str) -> dict:
    """Run full Zipf analysis on a single token sequence.

    Returns dict with label, global fit, piecewise fit, breakpoints,
    and summary statistics.
    """
    if not token_ids:
        return {
            "label": label,
            "num_total_tokens": 0,
            "num_unique_tokens": 0,
            "global_fit": {"alpha": 0.0, "q": 0.0, "r_squared": 0.0, "C": 0.0},
            "piecewise_fit": {
                "head": {"alpha": 0.0, "r_squared": 0.0},
                "body": {"alpha": 0.0, "r_squared": 0.0},
                "tail": {"alpha": 0.0, "r_squared": 0.0},
            },
            "breakpoints": [],
        }

    ranks, freqs = compute_rank_frequency(token_ids)

    global_fit = fit_zipf_mandelbrot(ranks, freqs)
    piecewise_fit = fit_piecewise_zipf(ranks, freqs)
    breakpoints = detect_breakpoints(ranks, freqs)

    return {
        "label": label,
        "num_total_tokens": len(token_ids),
        "num_unique_tokens": int(len(set(token_ids))),
        "global_fit": global_fit,
        "piecewise_fit": piecewise_fit,
        "breakpoints": breakpoints,
    }
