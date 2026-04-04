"""Worker agent types for the delegation game.

Each worker chooses an effort level (1-5) each round. Output quality is
effort + Gaussian noise. Different worker types use different strategies
to pick their effort level.
"""

from __future__ import annotations

import numpy as np
from typing import Protocol


class Worker(Protocol):
    """Protocol for worker agents."""
    name: str
    worker_type: str

    def choose_effort(self, round_num: int, history: list[dict]) -> int:
        """Return effort in [1, 5] given round number and history."""
        ...

    def reset(self) -> None:
        """Reset internal state for a new simulation."""
        ...


class HonestWorker:
    """Always exerts maximum effort (5)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.worker_type = "honest"

    def choose_effort(self, round_num: int, history: list[dict]) -> int:
        return 5

    def reset(self) -> None:
        pass


class ShirkerWorker:
    """Always exerts minimum effort (1)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.worker_type = "shirker"

    def choose_effort(self, round_num: int, history: list[dict]) -> int:
        return 1

    def reset(self) -> None:
        pass


class StrategicWorker:
    """Optimizes effort vs. pay. Picks effort that maximized
    expected(wage - effort_cost) based on the incentive scheme's
    implied marginal return. Uses a simple heuristic: if last round's
    pay per unit effort was above the effort cost, increase effort;
    otherwise decrease it.
    """

    def __init__(self, name: str, rng: np.random.Generator | None = None) -> None:
        self.name = name
        self.worker_type = "strategic"
        self._effort = 3  # start at midpoint
        self._rng = rng or np.random.default_rng()

    def choose_effort(self, round_num: int, history: list[dict]) -> int:
        if not history:
            return self._effort
        # Look at own last entry
        my_last = [h for h in history if h["worker"] == self.name]
        if not my_last:
            return self._effort
        last = my_last[-1]
        pay = last["wage"]
        effort = last["effort"]
        effort_cost = effort * 1.0  # cost = effort * 1.0
        pay_per_effort = pay / max(effort, 1)
        if pay_per_effort > 1.2:
            self._effort = min(5, self._effort + 1)
        elif pay_per_effort < 0.8:
            self._effort = max(1, self._effort - 1)
        return self._effort

    def reset(self) -> None:
        self._effort = 3


class AdaptiveWorker:
    """Learns optimal effort over time using exponential moving average
    of pay-per-effort. Starts exploring broadly, then converges.
    """

    def __init__(self, name: str, rng: np.random.Generator | None = None) -> None:
        self.name = name
        self.worker_type = "adaptive"
        self._rng = rng or np.random.default_rng()
        self._ema_returns: dict[int, float] = {}  # effort -> EMA of net return
        self._alpha = 0.1  # EMA smoothing
        self._effort = 3

    def choose_effort(self, round_num: int, history: list[dict]) -> int:
        if not history:
            # Explore uniformly in early rounds
            return int(self._rng.integers(1, 6))
        my_last = [h for h in history if h["worker"] == self.name]
        if not my_last:
            return int(self._rng.integers(1, 6))
        last = my_last[-1]
        e = last["effort"]
        net = last["wage"] - e * 1.0
        if e not in self._ema_returns:
            self._ema_returns[e] = net
        else:
            self._ema_returns[e] = (
                self._alpha * net + (1 - self._alpha) * self._ema_returns[e]
            )

        # Epsilon-greedy: 10% explore, 90% exploit
        if self._rng.random() < 0.1 or len(self._ema_returns) < 3:
            return int(self._rng.integers(1, 6))
        best_effort = max(self._ema_returns, key=self._ema_returns.get)
        return best_effort

    def reset(self) -> None:
        self._ema_returns = {}
        self._effort = 3


def create_worker(worker_type: str, name: str,
                  rng: np.random.Generator | None = None) -> Worker:
    """Factory function to create workers by type string."""
    constructors = {
        "honest": lambda: HonestWorker(name),
        "shirker": lambda: ShirkerWorker(name),
        "strategic": lambda: StrategicWorker(name, rng),
        "adaptive": lambda: AdaptiveWorker(name, rng),
    }
    if worker_type not in constructors:
        raise ValueError(f"Unknown worker type: {worker_type!r}. "
                         f"Choose from {list(constructors)}")
    return constructors[worker_type]()
