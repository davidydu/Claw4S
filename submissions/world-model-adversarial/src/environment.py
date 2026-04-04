"""Hidden environment with state distributions for adversarial signaling games.

The environment has a discrete set of possible states.  At each round,
it exposes the true state to the adversary and (optionally) drifts to a
new state according to a configurable regime.

State-drift regimes
-------------------
* **stable** -- the true state never changes.
* **slow_drift** -- the true state changes every ``drift_interval`` rounds
  (default 5 000).
* **volatile** -- the true state changes every ``drift_interval`` rounds
  (default 500).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


DriftRegime = Literal["stable", "slow_drift", "volatile"]

# Default drift intervals per regime.
_DEFAULT_DRIFT_INTERVALS: dict[DriftRegime, int | None] = {
    "stable": None,
    "slow_drift": 5_000,
    "volatile": 500,
}


@dataclass
class HiddenEnvironment:
    """A hidden environment whose true state can drift over time.

    Parameters
    ----------
    n_states : int
        Number of discrete states.
    drift_regime : DriftRegime
        How often the true state changes.
    drift_interval : int | None
        Override the default drift interval for the chosen regime.
    rng : np.random.Generator | None
        Random number generator.  If *None*, one is created from
        ``seed``.
    seed : int
        Seed used when ``rng`` is *None*.
    """

    n_states: int = 5
    drift_regime: DriftRegime = "stable"
    drift_interval: int | None = None
    rng: np.random.Generator | None = field(default=None, repr=False)
    seed: int = 0

    # ---- internal state (set in __post_init__) ----
    true_state: int = field(init=False)
    _round: int = field(init=False, default=0)
    _effective_drift_interval: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        self._effective_drift_interval = (
            self.drift_interval
            if self.drift_interval is not None
            else _DEFAULT_DRIFT_INTERVALS[self.drift_regime]
        )
        # Draw initial true state uniformly.
        self.true_state = int(self.rng.integers(0, self.n_states))
        self._round = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def true_state_distribution(self) -> NDArray[np.float64]:
        """Return a one-hot distribution over states for the current true state."""
        dist = np.zeros(self.n_states, dtype=np.float64)
        dist[self.true_state] = 1.0
        return dist

    def step(self) -> int:
        """Advance one round and potentially drift.  Returns the true state."""
        self._round += 1
        if (
            self._effective_drift_interval is not None
            and self._round % self._effective_drift_interval == 0
        ):
            self.true_state = int(self.rng.integers(0, self.n_states))
        return self.true_state

    def generate_noisy_signal(
        self, signal: int, noise_level: float
    ) -> int:
        """Optionally corrupt *signal* with uniform noise.

        With probability ``noise_level`` the signal is replaced by a
        uniformly random state.  With probability ``1 - noise_level``
        the signal is returned unchanged.
        """
        if noise_level <= 0.0:
            return signal
        if self.rng.random() < noise_level:
            return int(self.rng.integers(0, self.n_states))
        return signal

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment to a fresh initial state."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.true_state = int(self.rng.integers(0, self.n_states))
        self._round = 0
