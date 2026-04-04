"""Learner and adversary agents for the adversarial signaling game.

Learners
--------
* **NaiveLearner** -- full Bayesian updater that trusts every signal.
* **SkepticalLearner** -- discounts each signal by a fixed trust factor.
* **AdaptiveLearner** -- tracks signal accuracy and adjusts trust online.

Adversaries
-----------
* **RandomAdversary** -- sends uniformly random signals.
* **StrategicAdversary** -- greedily maximises single-round belief
  distortion.
* **PatientAdversary** -- sends truthful signals for *K* rounds to build
  credibility, then switches to maximally deceptive signals.

All learners apply a belief floor after each update to prevent beliefs
from collapsing to zero (which would make them irrecoverable).  The
floor is set to ``belief_floor / n_states`` per state, ensuring that
the learner always retains some probability mass on every hypothesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class Learner(ABC):
    """Base class for all learners."""

    def __init__(
        self,
        n_states: int,
        *,
        belief_floor: float = 0.01,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = n_states
        self.belief_floor = belief_floor
        self.rng = rng or np.random.default_rng()
        # Uniform prior.
        self.beliefs: NDArray[np.float64] = np.ones(n_states) / n_states

    def _apply_floor(self) -> None:
        """Ensure every state has at least ``belief_floor / n_states``."""
        if self.belief_floor > 0:
            floor_per_state = self.belief_floor / self.n_states
            self.beliefs = np.maximum(self.beliefs, floor_per_state)
            self.beliefs /= self.beliefs.sum()

    @abstractmethod
    def update(self, signal: int) -> None:
        """Update beliefs given a received signal."""

    def choose_action(self) -> int:
        """Choose the state with highest belief (MAP estimate)."""
        return int(np.argmax(self.beliefs))

    def reset(self) -> None:
        """Reset to uniform prior."""
        self.beliefs = np.ones(self.n_states) / self.n_states


class Adversary(ABC):
    """Base class for all adversaries."""

    def __init__(self, n_states: int, *, rng: np.random.Generator | None = None):
        self.n_states = n_states
        self.rng = rng or np.random.default_rng()
        self._round: int = 0

    @abstractmethod
    def choose_signal(self, true_state: int, learner_beliefs: NDArray[np.float64]) -> int:
        """Choose a signal to send given the true state and learner's beliefs."""

    def reset(self) -> None:
        self._round = 0


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

class NaiveLearner(Learner):
    """Bayesian updater that fully trusts every signal.

    On receiving signal *s*, multiplies belief[s] by ``signal_strength``
    and renormalises.  A higher ``signal_strength`` means the learner
    puts more weight on each new signal relative to its prior.
    """

    def __init__(
        self,
        n_states: int,
        *,
        signal_strength: float = 3.0,
        belief_floor: float = 0.01,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_states, belief_floor=belief_floor, rng=rng)
        self.signal_strength = signal_strength

    def update(self, signal: int) -> None:
        likelihood = np.ones(self.n_states)
        likelihood[signal] = self.signal_strength
        self.beliefs *= likelihood
        total = self.beliefs.sum()
        if total > 0:
            self.beliefs /= total
        else:
            self.beliefs = np.ones(self.n_states) / self.n_states
        self._apply_floor()


class SkepticalLearner(Learner):
    """Bayesian updater that discounts signals by a fixed trust factor.

    ``trust`` in [0, 1] controls how much of the signal likelihood
    is blended with the uniform distribution.  trust=1 is equivalent
    to NaiveLearner; trust=0 ignores all signals.
    """

    def __init__(
        self,
        n_states: int,
        *,
        trust: float = 0.4,
        signal_strength: float = 3.0,
        belief_floor: float = 0.01,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_states, belief_floor=belief_floor, rng=rng)
        self.trust = trust
        self.signal_strength = signal_strength

    def update(self, signal: int) -> None:
        full_likelihood = np.ones(self.n_states)
        full_likelihood[signal] = self.signal_strength
        uniform = np.ones(self.n_states)
        likelihood = self.trust * full_likelihood + (1.0 - self.trust) * uniform
        self.beliefs *= likelihood
        total = self.beliefs.sum()
        if total > 0:
            self.beliefs /= total
        else:
            self.beliefs = np.ones(self.n_states) / self.n_states
        self._apply_floor()


class AdaptiveLearner(Learner):
    """Learner that tracks signal accuracy and adjusts trust dynamically.

    Maintains an exponential moving average of signal accuracy
    (fraction of signals that matched the learner's subsequent best
    guess).  Uses this as the trust factor for a skeptical update.
    """

    def __init__(
        self,
        n_states: int,
        *,
        initial_trust: float = 0.7,
        ema_alpha: float = 0.02,
        signal_strength: float = 3.0,
        belief_floor: float = 0.01,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_states, belief_floor=belief_floor, rng=rng)
        self.trust = initial_trust
        self.initial_trust = initial_trust
        self.ema_alpha = ema_alpha
        self.signal_strength = signal_strength
        self._last_signal: int | None = None
        self._last_action: int | None = None

    def update(self, signal: int) -> None:
        # If we have history, evaluate whether the previous signal was
        # accurate (i.e. consistent with our current best action).
        if self._last_signal is not None and self._last_action is not None:
            accurate = 1.0 if self._last_signal == self._last_action else 0.0
            self.trust = (1 - self.ema_alpha) * self.trust + self.ema_alpha * accurate
            # Clamp trust to [0.05, 0.95] to prevent total collapse.
            self.trust = float(np.clip(self.trust, 0.05, 0.95))

        # Skeptical update using current trust.
        full_likelihood = np.ones(self.n_states)
        full_likelihood[signal] = self.signal_strength
        uniform = np.ones(self.n_states)
        likelihood = self.trust * full_likelihood + (1.0 - self.trust) * uniform
        self.beliefs *= likelihood
        total = self.beliefs.sum()
        if total > 0:
            self.beliefs /= total
        else:
            self.beliefs = np.ones(self.n_states) / self.n_states
        self._apply_floor()

        self._last_signal = signal

    def choose_action(self) -> int:
        action = super().choose_action()
        self._last_action = action
        return action

    def reset(self) -> None:
        super().reset()
        self.trust = self.initial_trust
        self._last_signal = None
        self._last_action = None


# ---------------------------------------------------------------------------
# Adversaries
# ---------------------------------------------------------------------------

class RandomAdversary(Adversary):
    """Sends uniformly random signals (baseline)."""

    def choose_signal(self, true_state: int, learner_beliefs: NDArray[np.float64]) -> int:
        return int(self.rng.integers(0, self.n_states))


class StrategicAdversary(Adversary):
    """Greedily sends the signal that maximises immediate belief distortion.

    Picks the state *s* that has the highest current belief weight **and**
    is NOT the true state.  This reinforces the learner's strongest
    incorrect belief, pulling it further from truth.  If all non-true
    states have equal belief, picks one at random.
    """

    def choose_signal(self, true_state: int, learner_beliefs: NDArray[np.float64]) -> int:
        # Mask the true state.
        masked = learner_beliefs.copy()
        masked[true_state] = -1.0
        # Pick the non-true state with highest current belief.
        best = int(np.argmax(masked))
        return best


class PatientAdversary(Adversary):
    """Builds credibility, then exploits.

    Sends truthful signals for the first ``credibility_rounds``, then
    switches to maximally deceptive signals (same strategy as
    StrategicAdversary).
    """

    def __init__(
        self,
        n_states: int,
        *,
        credibility_rounds: int = 200,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_states, rng=rng)
        self.credibility_rounds = credibility_rounds

    def choose_signal(self, true_state: int, learner_beliefs: NDArray[np.float64]) -> int:
        self._round += 1
        if self._round <= self.credibility_rounds:
            return true_state
        # Deceptive phase: same as StrategicAdversary.
        masked = learner_beliefs.copy()
        masked[true_state] = -1.0
        return int(np.argmax(masked))

    def reset(self) -> None:
        super().reset()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

LEARNER_TYPES: dict[str, type[Learner]] = {
    "NL": NaiveLearner,
    "SL": SkepticalLearner,
    "AL": AdaptiveLearner,
}

ADVERSARY_TYPES: dict[str, type[Adversary]] = {
    "RA": RandomAdversary,
    "SA": StrategicAdversary,
    "PA": PatientAdversary,
}


def make_learner(
    code: str, n_states: int, *, rng: np.random.Generator | None = None
) -> Learner:
    """Create a learner from its short code."""
    return LEARNER_TYPES[code](n_states, rng=rng)


def make_adversary(
    code: str, n_states: int, *, rng: np.random.Generator | None = None
) -> Adversary:
    """Create an adversary from its short code."""
    return ADVERSARY_TYPES[code](n_states, rng=rng)
