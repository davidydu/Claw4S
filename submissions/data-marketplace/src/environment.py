"""DataEnvironment — hidden world with N states and a true distribution.

The environment represents an unknown world that buyers want to learn about.
Data sellers provide noisy samples from this world; data quality determines
how faithfully those samples reflect the true distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DataEnvironment:
    """Hidden environment with ``n_states`` discrete states.

    Parameters
    ----------
    n_states : int
        Number of discrete world states.
    rng : np.random.Generator
        Random number generator (for reproducibility).
    true_dist : NDArray | None
        If *None* a random Dirichlet-drawn distribution is used.
    """

    def __init__(
        self,
        n_states: int = 5,
        rng: np.random.Generator | None = None,
        true_dist: NDArray | None = None,
    ) -> None:
        self.n_states = n_states
        self.rng = rng if rng is not None else np.random.default_rng()
        if true_dist is not None:
            assert len(true_dist) == n_states
            self.true_dist = np.asarray(true_dist, dtype=np.float64)
            self.true_dist /= self.true_dist.sum()
        else:
            self.true_dist = self.rng.dirichlet(np.ones(n_states))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_true(self, n: int = 1) -> NDArray:
        """Draw *n* i.i.d. samples from the true distribution.

        Returns integer state indices, shape ``(n,)``.
        """
        return self.rng.choice(self.n_states, size=n, p=self.true_dist)

    def sample_noisy(self, quality: float, n: int = 1) -> NDArray:
        """Draw *n* samples whose fidelity depends on *quality* in [0, 1].

        ``quality=1.0`` → samples from the true distribution.
        ``quality=0.0`` → samples from uniform distribution.
        Intermediate values blend the two.
        """
        quality = float(np.clip(quality, 0.0, 1.0))
        blended = quality * self.true_dist + (1.0 - quality) * np.ones(self.n_states) / self.n_states
        return self.rng.choice(self.n_states, size=n, p=blended)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def kl_divergence(self, q: NDArray) -> float:
        """KL(true || q) — how far *q* is from the true distribution.

        Uses a small epsilon to avoid log(0).
        """
        eps = 1e-12
        p = self.true_dist
        q = np.asarray(q, dtype=np.float64)
        q = q / q.sum()
        return float(np.sum(p * np.log((p + eps) / (q + eps))))

    def optimal_decision_value(self) -> float:
        """Value of a perfectly informed buyer (picks the mode)."""
        return float(np.max(self.true_dist))

    def decision_value(self, belief: NDArray) -> float:
        """Expected value when a buyer picks the state with highest belief.

        The buyer chooses argmax(belief), but the payoff is the true
        probability of that state.
        """
        chosen = int(np.argmax(belief))
        return float(self.true_dist[chosen])
