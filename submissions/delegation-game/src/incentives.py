"""Incentive schemes for the principal-agent delegation game.

Each scheme takes the observed output qualities for all workers in a round
and returns the wages paid to each worker.
"""

from __future__ import annotations

import numpy as np


# Base wage that ensures participation (all schemes pay at least this)
BASE_WAGE = 1.0


class IncentiveScheme:
    """Base class for incentive schemes."""

    name: str

    def compute_wages(self, qualities: list[float],
                      worker_names: list[str],
                      round_num: int,
                      reputation_scores: dict[str, float] | None = None
                      ) -> list[float]:
        """Return wage for each worker given observed qualities."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state."""
        pass


class FixedPay(IncentiveScheme):
    """Constant wage regardless of output quality.

    Workers receive a flat wage each round.
    """

    name = "fixed_pay"

    def __init__(self, wage: float = 3.0) -> None:
        self.wage = wage

    def compute_wages(self, qualities, worker_names, round_num,
                      reputation_scores=None):
        return [self.wage] * len(qualities)


class PieceRate(IncentiveScheme):
    """Wage proportional to observed output quality.

    wage_i = base + rate * quality_i
    """

    name = "piece_rate"

    def __init__(self, rate: float = 0.5, base: float = BASE_WAGE) -> None:
        self.rate = rate
        self.base = base

    def compute_wages(self, qualities, worker_names, round_num,
                      reputation_scores=None):
        return [self.base + self.rate * max(q, 0.0) for q in qualities]


class Tournament(IncentiveScheme):
    """Top performer gets a bonus; others get base wage.

    Ties broken by giving all tied workers the bonus (split).
    """

    name = "tournament"

    def __init__(self, bonus: float = 4.0, base: float = BASE_WAGE) -> None:
        self.bonus = bonus
        self.base = base

    def compute_wages(self, qualities, worker_names, round_num,
                      reputation_scores=None):
        if not qualities:
            return []
        max_q = max(qualities)
        winners = [i for i, q in enumerate(qualities) if q == max_q]
        split_bonus = self.bonus / len(winners)
        wages = []
        for i in range(len(qualities)):
            if i in winners:
                wages.append(self.base + split_bonus)
            else:
                wages.append(self.base)
        return wages


class ReputationBased(IncentiveScheme):
    """Wage based on exponential moving average of past quality.

    Reputation starts at 0.5. Each round, reputation is updated:
        rep_new = alpha * quality_normalized + (1 - alpha) * rep_old
    Wage = base + rep_bonus * reputation
    """

    name = "reputation"

    def __init__(self, alpha: float = 0.1, rep_bonus: float = 4.0,
                 base: float = BASE_WAGE) -> None:
        self.alpha = alpha
        self.rep_bonus = rep_bonus
        self.base = base
        self._reputations: dict[str, float] = {}

    def compute_wages(self, qualities, worker_names, round_num,
                      reputation_scores=None):
        wages = []
        for name, q in zip(worker_names, qualities):
            if name not in self._reputations:
                self._reputations[name] = 0.5
            # Normalize quality to [0, 1] range (effort 1-5 + noise, clamp)
            q_norm = np.clip((q - 1.0) / 4.0, 0.0, 1.0)
            self._reputations[name] = (
                self.alpha * q_norm
                + (1 - self.alpha) * self._reputations[name]
            )
            wages.append(self.base + self.rep_bonus * self._reputations[name])
        return wages

    def get_reputations(self) -> dict[str, float]:
        return dict(self._reputations)

    def reset(self) -> None:
        self._reputations = {}


SCHEME_REGISTRY: dict[str, type[IncentiveScheme]] = {
    "fixed_pay": FixedPay,
    "piece_rate": PieceRate,
    "tournament": Tournament,
    "reputation": ReputationBased,
}


def create_scheme(name: str, **kwargs) -> IncentiveScheme:
    """Factory function to create incentive schemes by name."""
    if name not in SCHEME_REGISTRY:
        raise ValueError(f"Unknown scheme: {name!r}. "
                         f"Choose from {list(SCHEME_REGISTRY)}")
    return SCHEME_REGISTRY[name](**kwargs)
