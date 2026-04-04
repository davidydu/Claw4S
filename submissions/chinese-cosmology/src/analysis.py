# src/analysis.py
"""Statistical analysis for cross-system consistency evaluation.

Provides:
  compute_null_model        — shuffle-based null distribution
  bootstrap_ci              — percentile bootstrap confidence intervals
  compute_conditional_agreement — agreement rates for extreme vs middle charts
  apply_bonferroni          — Bonferroni multiple-comparisons correction
  compute_statistics        — aggregate cross-system statistics from records
  analyze_results           — orchestrate evaluator panel + statistics
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.evaluators import EvaluatorPanel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
DOMAINS = ["career", "wealth", "relationships", "health", "overall"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain_scores(records: list, prefix: str, domain: str) -> np.ndarray:
    """Extract a flat array of scores for system `prefix` and `domain`.

    Args:
        records: list of record dicts from analyze_results
        prefix:  one of 'bazi', 'ziwei', 'wuxing'
        domain:  one of DOMAINS

    Returns:
        1-D numpy array of floats
    """
    key = f"{prefix}_{domain}"
    return np.array([r[key] for r in records if key in r], dtype=float)


def _pearson_r(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson r; returns 0 for degenerate cases."""
    if len(xs) < 2:
        return 0.0
    std_x = float(np.std(xs))
    std_y = float(np.std(ys))
    if std_x < 1e-15 or std_y < 1e-15:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fisher_inference(
    r: float,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Return Fisher-z CI and two-sided p-value for Pearson r."""
    if n < 4:
        return (r, r, 1.0)

    eps = 1e-12
    r_clipped = max(min(r, 1.0 - eps), -1.0 + eps)
    z = math.atanh(r_clipped)
    se = 1.0 / math.sqrt(n - 3)

    z_crit = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else 1.959963984540054
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    ci_lo = math.tanh(z_lo)
    ci_hi = math.tanh(z_hi)

    z_stat = abs(z) / se
    p_value = 2.0 * (1.0 - _normal_cdf(z_stat))
    p_value = max(0.0, min(1.0, p_value))

    return (ci_lo, ci_hi, p_value)


# ---------------------------------------------------------------------------
# Task: Null model
# ---------------------------------------------------------------------------

def compute_null_model(
    scores_a: list,
    scores_b: list,
    n_permutations: int = 1000,
    seed: int = RANDOM_SEED,
) -> list:
    """Shuffle-based null distribution of Pearson |r|.

    Randomly permutes scores_b, computes |r| between scores_a and the
    permuted scores_b, and repeats n_permutations times.  This establishes
    the expected agreement under the null hypothesis (no relationship).

    Args:
        scores_a:       array-like of floats in [0, 1]
        scores_b:       array-like of floats in [0, 1], same length
        n_permutations: number of permutations
        seed:           RNG seed for reproducibility

    Returns:
        list of length n_permutations, each element |r| for one permutation
    """
    rng = np.random.default_rng(seed)
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    null_scores = []
    for _ in range(n_permutations):
        b_perm = rng.permutation(b)
        r = _pearson_r(a, b_perm)
        null_scores.append(abs(r))
    return null_scores


# ---------------------------------------------------------------------------
# Task: Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Args:
        values:      list of float observations
        confidence:  desired confidence level (e.g. 0.95 for 95%)
        n_bootstrap: number of bootstrap resamples
        seed:        RNG seed for reproducibility

    Returns:
        (lower, upper) CI bounds
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


# ---------------------------------------------------------------------------
# Task: Conditional agreement
# ---------------------------------------------------------------------------

def compute_conditional_agreement(records: list) -> Dict:
    """Compute agreement rates for extreme vs. middle charts.

    "Extreme" charts: top 20% or bottom 20% of BaZi career score.
    "Middle" charts: the remaining 60%.

    Computes BaZi–Zi Wei domain agreement (both > 0.5 or both ≤ 0.5) for each group.

    Args:
        records: list of record dicts with 'bazi_career', 'ziwei_career', etc.

    Returns:
        dict with 'extreme_agreement' and 'middle_agreement' rates in [0, 1]
    """
    if not records:
        return {"extreme_agreement": 0.0, "middle_agreement": 0.0}

    bazi_career = np.array([r.get("bazi_career", 0.5) for r in records])
    p20 = float(np.percentile(bazi_career, 20))
    p80 = float(np.percentile(bazi_career, 80))

    extreme_records = [r for r, s in zip(records, bazi_career)
                       if s <= p20 or s >= p80]
    middle_records = [r for r, s in zip(records, bazi_career)
                      if p20 < s < p80]

    def _agreement_rate(recs):
        if not recs:
            return 0.0
        agree = sum(
            1 for r in recs
            if (r.get("bazi_career", 0.5) > 0.5) == (r.get("ziwei_career", 0.5) > 0.5)
        )
        return agree / len(recs)

    return {
        "extreme_agreement": round(_agreement_rate(extreme_records), 6),
        "middle_agreement": round(_agreement_rate(middle_records), 6),
        "n_extreme": len(extreme_records),
        "n_middle": len(middle_records),
    }


# ---------------------------------------------------------------------------
# Task: Bonferroni correction
# ---------------------------------------------------------------------------

def apply_bonferroni(p_values: list, n_tests: int = 15) -> list:
    """Apply Bonferroni correction to a list of p-values.

    Multiplies each p-value by n_tests and clips to [0, 1].

    Args:
        p_values: list of raw p-values in [0, 1]
        n_tests:  total number of tests (default 15: 5 domains × 3 system pairs)

    Returns:
        list of corrected p-values, same length as input
    """
    return [min(float(p) * n_tests, 1.0) for p in p_values]


# ---------------------------------------------------------------------------
# Task: Compute statistics
# ---------------------------------------------------------------------------

def compute_statistics(records: list) -> Dict:
    """Compute cross-system statistics from a list of chart records.

    Each record is a flat dict with keys like 'bazi_career', 'ziwei_career',
    'wuxing_career', 'bazi_wealth', etc.

    Returns:
        dict with keys:
          - 'correlation': dict {domain: {'bazi_ziwei': float, 'bazi_wuxing': float,
                                           'ziwei_wuxing': float}}
          - 'domain_agreement': dict {domain: {'bazi_ziwei': float, ...}}
          - 'mutual_information': dict {domain: float}
          - 'temporal_patterns': list of yearly stats
          - 'conditional_agreement': dict from compute_conditional_agreement
          - 'n_records': int
    """
    if not records:
        return {
            "correlation": {},
            "correlation_inference": {},
            "domain_agreement": {},
            "mutual_information": {},
            "temporal_patterns": [],
            "conditional_agreement": {},
            "n_records": 0,
        }

    # Compute primary statistics on the full record set (no subsampling).
    correlation: Dict[str, Dict[str, float]] = {}
    correlation_inference: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    domain_agreement: Dict[str, Dict[str, float]] = {}
    mi: Dict[str, float] = {}
    inference_index: List[Tuple[str, str]] = []
    inference_p_values: List[float] = []

    for domain in DOMAINS:
        bazi = _extract_domain_scores(records, "bazi", domain)
        ziwei = _extract_domain_scores(records, "ziwei", domain)
        wuxing = _extract_domain_scores(records, "wuxing", domain)

        if len(bazi) == 0:
            continue

        pair_arrays = {
            "bazi_ziwei": (bazi, ziwei),
            "bazi_wuxing": (bazi, wuxing),
            "ziwei_wuxing": (ziwei, wuxing),
        }

        correlation[domain] = {}
        correlation_inference[domain] = {}
        for pair_name, xs_ys in pair_arrays.items():
            xs, ys = xs_ys
            n_obs = min(len(xs), len(ys))
            if n_obs < 2:
                correlation[domain][pair_name] = 0.0
                correlation_inference[domain][pair_name] = {
                    "r": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "p_value": 1.0,
                    "p_value_bonferroni": 1.0,
                    "n": n_obs,
                }
                continue

            xs_n = xs[:n_obs]
            ys_n = ys[:n_obs]
            r_val = _pearson_r(xs_n, ys_n)
            ci_lo, ci_hi, p_raw = _fisher_inference(r_val, n=n_obs, confidence=0.95)

            correlation[domain][pair_name] = round(r_val, 6)
            correlation_inference[domain][pair_name] = {
                "r": round(r_val, 6),
                "ci_lower": round(ci_lo, 6),
                "ci_upper": round(ci_hi, 6),
                "p_value": p_raw,
                "p_value_bonferroni": 1.0,
                "n": n_obs,
            }
            inference_index.append((domain, pair_name))
            inference_p_values.append(p_raw)

        # Domain agreement (both > 0.5 or both ≤ 0.5)
        def _agree_rate(a, b):
            return float(np.mean(((a > 0.5) & (b > 0.5)) | ((a <= 0.5) & (b <= 0.5))))

        domain_agreement[domain] = {
            "bazi_ziwei": round(_agree_rate(bazi, ziwei), 6),
            "bazi_wuxing": round(_agree_rate(bazi, wuxing), 6),
            "ziwei_wuxing": round(_agree_rate(ziwei, wuxing), 6),
        }

        # Mutual information (BaZi vs Zi Wei, 10-bin discretization)
        bazi_bins = np.floor(np.clip(bazi, 0, 1 - 1e-12) * 10).astype(int)
        ziwei_bins = np.floor(np.clip(ziwei, 0, 1 - 1e-12) * 10).astype(int)
        n = len(bazi_bins)
        joint_counts: Dict[Tuple[int, int], int] = {}
        marg_a: Dict[int, int] = {}
        marg_b: Dict[int, int] = {}
        for a_b, z_b in zip(bazi_bins, ziwei_bins):
            a_b, z_b = int(a_b), int(z_b)
            joint_counts[(a_b, z_b)] = joint_counts.get((a_b, z_b), 0) + 1
            marg_a[a_b] = marg_a.get(a_b, 0) + 1
            marg_b[z_b] = marg_b.get(z_b, 0) + 1
        mi_val = 0.0
        for (a_b, z_b), cnt in joint_counts.items():
            p_ab = cnt / n
            p_a = marg_a[a_b] / n
            p_b = marg_b[z_b] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi_val += p_ab * math.log(p_ab / (p_a * p_b))
        mi[domain] = round(max(0.0, mi_val), 6)

    if inference_p_values:
        corrected = apply_bonferroni(inference_p_values, n_tests=len(inference_p_values))
        for (domain, pair_name), p_corr in zip(inference_index, corrected):
            correlation_inference[domain][pair_name]["p_value_bonferroni"] = p_corr

    # Temporal patterns: group by year, compute mean agreement
    temporal: List[Dict] = []
    year_groups: Dict[int, List] = {}
    for r in records:
        dt_str = r.get("datetime", "")
        if dt_str:
            try:
                year = int(dt_str[:4])
                year_groups.setdefault(year, []).append(r)
            except (ValueError, IndexError):
                pass

    for year in sorted(year_groups.keys()):
        yr_recs = year_groups[year]
        bazi_c = _extract_domain_scores(yr_recs, "bazi", "career")
        ziwei_c = _extract_domain_scores(yr_recs, "ziwei", "career")
        if len(bazi_c) > 0:
            agree = float(np.mean(
                ((bazi_c > 0.5) & (ziwei_c > 0.5)) | ((bazi_c <= 0.5) & (ziwei_c <= 0.5))
            ))
            temporal.append({"year": year, "career_agreement": round(agree, 6)})

    conditional = compute_conditional_agreement(records)

    return {
        "correlation": correlation,
        "correlation_inference": correlation_inference,
        "domain_agreement": domain_agreement,
        "mutual_information": mi,
        "temporal_patterns": temporal,
        "conditional_agreement": conditional,
        "n_records": len(records),
    }


# ---------------------------------------------------------------------------
# Task: Analyze results (orchestrator)
# ---------------------------------------------------------------------------

def analyze_results(chart_results: list) -> Dict:
    """Orchestrate evaluator panel + statistical analysis.

    Args:
        chart_results: list of dicts from run_chart_analysis, each with
                       'datetime', 'bazi', 'ziwei', 'wuxing' keys

    Returns:
        dict with keys:
          - 'records':    flat list of per-chart dicts (with domain scores)
          - 'evaluator_results': list of ConsistencyResult dicts (per domain)
          - 'statistics': output of compute_statistics
    """
    # Flatten chart results into per-domain records
    flat_records = []
    for cr in chart_results:
        record: Dict = {"datetime": cr.get("datetime", "")}
        for system in ["bazi", "ziwei", "wuxing"]:
            sys_data = cr.get(system, {})
            scores = sys_data.get("domain_scores", {})
            for domain, score in scores.items():
                record[f"{system}_{domain}"] = score
        flat_records.append(record)

    # Run evaluator panel on career domain (representative)
    panel = EvaluatorPanel()
    evaluator_results_by_domain = {}
    for domain in DOMAINS:
        bazi_scores = [r.get(f"bazi_{domain}", 0.5) for r in flat_records]
        ziwei_scores = [r.get(f"ziwei_{domain}", 0.5) for r in flat_records]
        wuxing_scores = [r.get(f"wuxing_{domain}", 0.5) for r in flat_records]

        panel_out = panel.evaluate_all(bazi_scores, ziwei_scores, wuxing_scores)
        evaluator_results_by_domain[domain] = [
            {
                "evaluator_name": r.evaluator_name,
                "consistency_score": r.consistency_score,
                "evidence": r.evidence,
            }
            for r in panel_out
        ]

    stats = compute_statistics(flat_records)

    return {
        "records": flat_records,
        "evaluator_results": evaluator_results_by_domain,
        "statistics": stats,
    }
