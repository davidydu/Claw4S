"""Metrics for measuring emergent social norms.

Four core metrics quantify norm emergence dynamics:

1. Norm Convergence Time: the round at which a single action captures >= 80%
   of all interactions in a trailing window. If convergence never occurs,
   returns total_rounds (censored).

2. Norm Efficiency: ratio of realized average payoff to the optimal
   coordination payoff. Values in [0, 1]; 1.0 means the population
   converged to the welfare-maximizing equilibrium.

3. Norm Fragility: the fraction of innovators in the population at which
   the dominant norm is displaced. Measured by progressively injecting
   innovators and checking if the norm survives.

4. Norm Diversity: number of distinct behavioral clusters, defined as
   actions commanding >= 10% of recent interactions. Values in {1, 2, 3}.
"""

import numpy as np


CONVERGENCE_THRESHOLD = 0.80
DIVERSITY_THRESHOLD = 0.10
WINDOW_SIZE = 500


def norm_convergence_time(
    action_history: np.ndarray,
    total_rounds: int,
) -> int:
    """Round at which one action captures >= 80% of a trailing window.

    Parameters
    ----------
    action_history : ndarray of shape (total_rounds,)
        The action chosen in each pairwise interaction (by convention,
        the action of the first agent in the pair; both play the same
        action when coordinating).
    total_rounds : int
        Total number of rounds in the simulation.

    Returns
    -------
    int
        Round of convergence, or total_rounds if never converged.
    """
    n = len(action_history)
    if n < WINDOW_SIZE:
        return total_rounds

    # Use a sliding window with counts
    window = action_history[:WINDOW_SIZE]
    counts = np.bincount(window, minlength=3).astype(np.float64)

    if counts.max() / WINDOW_SIZE >= CONVERGENCE_THRESHOLD:
        return WINDOW_SIZE

    for i in range(WINDOW_SIZE, n):
        # Slide: add new, remove old
        counts[action_history[i]] += 1
        counts[action_history[i - WINDOW_SIZE]] -= 1

        if counts.max() / WINDOW_SIZE >= CONVERGENCE_THRESHOLD:
            return i + 1

    return total_rounds


def norm_efficiency(
    payoff_history: np.ndarray,
    optimal_payoff: float,
    tail_fraction: float = 0.2,
) -> float:
    """Ratio of realized tail-average payoff to optimal coordination payoff.

    Parameters
    ----------
    payoff_history : ndarray of shape (total_rounds,)
        Average per-agent payoff at each round.
    optimal_payoff : float
        Maximum per-agent payoff under perfect coordination.
    tail_fraction : float
        Fraction of final rounds to average over (default 0.2 = last 20%).

    Returns
    -------
    float
        Efficiency in [0, 1]. Returns 0.0 if optimal_payoff is 0.
    """
    if optimal_payoff <= 0 or len(payoff_history) == 0:
        return 0.0

    tail_start = max(0, int(len(payoff_history) * (1 - tail_fraction)))
    tail_avg = float(np.mean(payoff_history[tail_start:]))
    return min(tail_avg / optimal_payoff, 1.0)


def norm_diversity(action_history: np.ndarray, tail_fraction: float = 0.2) -> int:
    """Number of actions with >= 10% share in the tail of the simulation.

    Parameters
    ----------
    action_history : ndarray of shape (total_rounds,)
        Action chosen in each interaction.
    tail_fraction : float
        Fraction of final rounds to analyze.

    Returns
    -------
    int
        Number of distinct behavioral clusters (1, 2, or 3).
    """
    n = len(action_history)
    if n == 0:
        return 0

    tail_start = max(0, int(n * (1 - tail_fraction)))
    tail = action_history[tail_start:]
    counts = np.bincount(tail, minlength=3)
    fractions = counts / max(len(tail), 1)
    return int(np.sum(fractions >= DIVERSITY_THRESHOLD))


def norm_fragility(
    action_history: np.ndarray,
    innovator_fractions: list[float],
    action_histories_by_fraction: list[np.ndarray],
) -> float:
    """Fraction of innovators at which the baseline norm is displaced.

    This is computed post-hoc: given the baseline simulation's dominant action,
    we check at which innovator fraction a different action becomes dominant.

    Parameters
    ----------
    action_history : ndarray
        Baseline (no extra innovators) action history.
    innovator_fractions : list[float]
        Fractions of innovators tested (sorted ascending).
    action_histories_by_fraction : list[ndarray]
        Action history for each innovator fraction.

    Returns
    -------
    float
        Innovator fraction at which norm breaks, or 1.0 if never broken.
    """
    # Identify baseline dominant action from tail
    n = len(action_history)
    tail_start = max(0, int(n * 0.8))
    tail = action_history[tail_start:]
    if len(tail) == 0:
        return 1.0
    baseline_counts = np.bincount(tail, minlength=3)
    dominant_action = int(np.argmax(baseline_counts))

    for frac, hist in zip(innovator_fractions, action_histories_by_fraction):
        n2 = len(hist)
        tail_start2 = max(0, int(n2 * 0.8))
        tail2 = hist[tail_start2:]
        if len(tail2) == 0:
            continue
        counts = np.bincount(tail2, minlength=3)
        new_dominant = int(np.argmax(counts))
        if new_dominant != dominant_action:
            return frac

    return 1.0
