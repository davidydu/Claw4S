"""Metrics for the information-sharing experiment.

Computed from simulation history:
1. Sharing Rate — average disclosure level per agent type
2. Group Welfare — sum of all agents' payoffs
3. Individual Welfare Gap — max - min payoff across agents
4. Information Asymmetry (Gini) — Gini coefficient of cumulative info
5. Sharing Norm Convergence — std of sharing rates in last 10% of rounds
"""

from __future__ import annotations

import numpy as np


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1-D array.

    0 = perfect equality, 1 = maximal inequality.
    Returns 0.0 for arrays with all-zero or single-element inputs.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or np.all(v == 0):
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def compute_round_metrics(
    payoffs: np.ndarray,
    sharing_rates: np.ndarray,
    errors: np.ndarray,
) -> dict:
    """Compute per-round metrics.

    Parameters
    ----------
    payoffs : ndarray (n_agents,)
    sharing_rates : ndarray (n_agents,)
    errors : ndarray (n_agents,)

    Returns
    -------
    dict with keys: mean_sharing, group_welfare, welfare_gap,
                    info_asymmetry, mean_error
    """
    return {
        "mean_sharing": float(np.mean(sharing_rates)),
        "group_welfare": float(np.sum(payoffs)),
        "welfare_gap": float(np.max(payoffs) - np.min(payoffs)),
        "info_asymmetry": gini_coefficient(errors),  # higher error = less info
        "mean_error": float(np.mean(errors)),
    }


def compute_summary_metrics(
    round_metrics: list[dict],
    agent_types: list[str],
    per_agent_sharing: list[np.ndarray],
    per_agent_payoffs: list[np.ndarray],
    n_rounds: int,
) -> dict:
    """Compute summary metrics over the full simulation.

    Parameters
    ----------
    round_metrics : list of per-round metric dicts
    agent_types : list of agent type names
    per_agent_sharing : list of ndarray (n_agents,) per round
    per_agent_payoffs : list of ndarray (n_agents,) per round
    n_rounds : int

    Returns
    -------
    dict with summary statistics
    """
    n_agents = len(agent_types)
    tail_start = int(n_rounds * 0.9)

    # Overall averages
    avg_sharing = float(np.mean([m["mean_sharing"] for m in round_metrics]))
    avg_welfare = float(np.mean([m["group_welfare"] for m in round_metrics]))
    avg_gap = float(np.mean([m["welfare_gap"] for m in round_metrics]))
    avg_asymmetry = float(np.mean([m["info_asymmetry"] for m in round_metrics]))

    # Tail (last 10%) averages — measures equilibrium behavior
    tail_metrics = round_metrics[tail_start:]
    tail_sharing = float(np.mean([m["mean_sharing"] for m in tail_metrics]))
    tail_welfare = float(np.mean([m["group_welfare"] for m in tail_metrics]))

    # Per-agent-type sharing rates in tail
    sharing_arr = np.array(per_agent_sharing)  # (n_rounds, n_agents)
    tail_sharing_arr = sharing_arr[tail_start:]
    type_sharing = {}
    for i in range(n_agents):
        t = agent_types[i]
        if t not in type_sharing:
            type_sharing[t] = []
        type_sharing[t].extend(tail_sharing_arr[:, i].tolist())
    per_type_sharing = {t: float(np.mean(v)) for t, v in type_sharing.items()}

    # Sharing norm convergence: std of mean sharing in last 10%
    tail_means = [m["mean_sharing"] for m in tail_metrics]
    norm_convergence = float(np.std(tail_means))

    # Per-agent cumulative payoffs
    payoff_arr = np.array(per_agent_payoffs)  # (n_rounds, n_agents)
    cum_payoffs = payoff_arr.sum(axis=0)
    per_type_payoffs = {}
    for i in range(n_agents):
        t = agent_types[i]
        if t not in per_type_payoffs:
            per_type_payoffs[t] = []
        per_type_payoffs[t].append(float(cum_payoffs[i]))
    per_type_payoffs = {t: float(np.mean(v)) for t, v in per_type_payoffs.items()}

    return {
        "avg_sharing_rate": avg_sharing,
        "avg_group_welfare": avg_welfare,
        "avg_welfare_gap": avg_gap,
        "avg_info_asymmetry": avg_asymmetry,
        "tail_sharing_rate": tail_sharing,
        "tail_group_welfare": tail_welfare,
        "norm_convergence_std": norm_convergence,
        "per_type_sharing": per_type_sharing,
        "per_type_cumulative_payoff": per_type_payoffs,
    }
