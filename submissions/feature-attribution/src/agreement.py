"""Measure pairwise agreement between attribution methods via Spearman rank correlation."""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple


def pairwise_spearman(
    attributions: Dict[str, np.ndarray],
    method_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute Spearman rank correlation for each pair of attribution methods.

    Args:
        attributions: Dict mapping method name -> (d,) attribution array.
        method_pairs: List of (method_a, method_b) tuples.

    Returns:
        Dict mapping "method_a_vs_method_b" -> Spearman rho.
    """
    results = {}
    for m_a, m_b in method_pairs:
        attr_a = attributions[m_a]
        attr_b = attributions[m_b]
        rho, _ = spearmanr(attr_a, attr_b)
        # Handle NaN from constant arrays (all-zero attributions)
        if np.isnan(rho):
            rho = 0.0
        pair_key = f"{m_a}_vs_{m_b}"
        results[pair_key] = float(rho)
    return results


def aggregate_agreement(
    all_correlations: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate pairwise correlations across samples.

    Args:
        all_correlations: List of per-sample correlation dicts.

    Returns:
        Dict mapping pair_key -> {"mean": ..., "std": ...}.
    """
    if not all_correlations:
        raise ValueError("No correlation data to aggregate")

    keys = all_correlations[0].keys()
    result = {}
    for key in keys:
        values = [c[key] for c in all_correlations]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return result
