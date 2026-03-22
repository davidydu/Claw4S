"""
Coordination game with parameterized prior disagreement.

N agents must simultaneously choose one of K actions.
Payoff = 1.0 if ALL agents choose the same action, 0.0 otherwise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class CoordinationGame:
    """Pure coordination game with disagreement-parameterised priors.

    Parameters
    ----------
    n_agents : int
        Number of agents (N).
    n_actions : int
        Number of possible coordination points (K).
    disagreement : float
        0.0 = all agents share the same "correct action" belief.
        1.0 = priors are maximally spread (each agent favours a
              different action as much as possible).
    seed : int
        Random seed for prior generation.
    """

    n_agents: int = 4
    n_actions: int = 5
    disagreement: float = 0.0
    seed: int = 42

    # Generated after init
    priors: np.ndarray = field(default=None, repr=False)  # (n_agents, n_actions)

    def __post_init__(self) -> None:
        self.priors = self._generate_priors()

    # ------------------------------------------------------------------
    # Prior generation
    # ------------------------------------------------------------------
    def _generate_priors(self) -> np.ndarray:
        """Generate agent prior beliefs over actions.

        At disagreement=0 every agent has the same peaked distribution.
        At disagreement=1 each agent's peak is rotated to a different action.
        Intermediate values interpolate linearly between these extremes.
        """
        rng = np.random.default_rng(self.seed)
        K = self.n_actions
        N = self.n_agents

        # Base peaked distribution: one action is strongly preferred
        # Use a concentration parameter so the peak is clear
        base_peak = np.zeros(K)
        peak_action = rng.integers(0, K)
        base_peak[peak_action] = 5.0  # strong preference
        for j in range(K):
            if j != peak_action:
                base_peak[j] = 0.2  # small residual

        base_peak = base_peak / base_peak.sum()  # normalise

        # Uniform distribution (maximum entropy)
        uniform = np.ones(K) / K

        # Consensus priors: all agents share the base_peak
        consensus_priors = np.tile(base_peak, (N, 1))  # (N, K)

        # Dispersed priors: each agent's peak is shifted
        dispersed_priors = np.empty((N, K))
        for i in range(N):
            shift = (i * K) // N  # spread agents across actions
            shifted = np.roll(base_peak, shift)
            dispersed_priors[i] = shifted

        # Interpolate: prior_i = (1 - d) * consensus + d * dispersed
        priors = (1.0 - self.disagreement) * consensus_priors + self.disagreement * dispersed_priors

        # Re-normalise each row (should already be normalised, but be safe)
        row_sums = priors.sum(axis=1, keepdims=True)
        priors = priors / row_sums

        return priors

    # ------------------------------------------------------------------
    # Payoff
    # ------------------------------------------------------------------
    def payoff(self, actions: List[int]) -> np.ndarray:
        """Compute payoffs for a round.

        Parameters
        ----------
        actions : list[int]
            Action chosen by each agent (length N, values in 0..K-1).

        Returns
        -------
        np.ndarray of shape (N,) with payoff per agent.
        """
        assert len(actions) == self.n_agents
        if len(set(actions)) == 1:
            return np.ones(self.n_agents)
        else:
            return np.zeros(self.n_agents)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def preferred_actions(self) -> List[int]:
        """Return each agent's initially most-preferred action."""
        return [int(np.argmax(self.priors[i])) for i in range(self.n_agents)]

    def prior_entropy(self) -> np.ndarray:
        """Shannon entropy of each agent's prior (nats)."""
        # Clip to avoid log(0)
        p = np.clip(self.priors, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    def agreement_score(self) -> float:
        """Fraction of agent-pairs that share the same preferred action.

        1.0 = perfect agreement, 0.0 = all prefer different actions.
        """
        prefs = self.preferred_actions()
        n = len(prefs)
        if n < 2:
            return 1.0
        agree = sum(
            1 for i in range(n) for j in range(i + 1, n) if prefs[i] == prefs[j]
        )
        total = n * (n - 1) // 2
        return agree / total
