"""Data buyers — Naive, Reputation-tracking, Analytical.

Each buyer maintains a Bayesian belief (Dirichlet posterior) about the
hidden environment and makes purchasing decisions each round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.sellers import Offer


# ── Base class ────────────────────────────────────────────────────
class BaseBuyer:
    """Abstract buyer base class."""

    buyer_type: str = "base"

    def __init__(self, buyer_id: int, n_states: int, rng: np.random.Generator,
                 budget_per_round: float = 0.5) -> None:
        self.buyer_id = buyer_id
        self.n_states = n_states
        self.rng = rng
        self.budget = budget_per_round

        # Bayesian belief: Dirichlet posterior (start with uniform prior α=1)
        self.alpha = np.ones(n_states, dtype=np.float64)  # Dirichlet params
        self.total_spent = 0.0
        self.total_value = 0.0
        self.n_purchases = 0
        self.round_history: list[dict[str, Any]] = []

    @property
    def belief(self) -> NDArray:
        """Current MAP estimate of the world distribution."""
        return self.alpha / self.alpha.sum()

    def choose_offer(self, offers: list[Offer], round_idx: int) -> int | None:
        """Return the index into *offers* to purchase, or None to abstain."""
        raise NotImplementedError

    def update_belief(self, data_samples: NDArray) -> None:
        """Bayesian update: add observed counts to Dirichlet parameters."""
        counts = np.bincount(data_samples, minlength=self.n_states)
        self.alpha += counts.astype(np.float64)

    def record_purchase(self, price: float, decision_value: float) -> None:
        self.total_spent += price
        self.total_value += decision_value
        self.n_purchases += 1

    @property
    def welfare(self) -> float:
        """Average surplus (value - price) per purchase."""
        if self.n_purchases == 0:
            return 0.0
        return (self.total_value - self.total_spent) / self.n_purchases


# ── Naive Buyer ───────────────────────────────────────────────────
class NaiveBuyer(BaseBuyer):
    """Trusts quality claims.  Picks the cheapest offer that claims high quality.

    "High quality" = claimed_quality >= 0.7.  If nothing qualifies,
    picks the cheapest offer within budget.
    """

    buyer_type = "naive"

    def choose_offer(self, offers: list[Offer], round_idx: int) -> int | None:
        if not offers:
            return None

        # Prefer high-claimed-quality offers, pick cheapest among them
        high_q = [(i, o) for i, o in enumerate(offers) if o.claimed_quality >= 0.7 and o.price <= self.budget]
        if high_q:
            return min(high_q, key=lambda x: x[1].price)[0]

        # Fallback: cheapest affordable
        affordable = [(i, o) for i, o in enumerate(offers) if o.price <= self.budget]
        if affordable:
            return min(affordable, key=lambda x: x[1].price)[0]

        return None


# ── Reputation Buyer ──────────────────────────────────────────────
class ReputationBuyer(BaseBuyer):
    """Tracks per-seller accuracy and prefers sellers with good track records.

    After each purchase, the buyer compares the data quality (measured by
    how much the data improved their belief) to the claimed quality.
    A reputation score accumulates, and the buyer selects the seller with
    the best *value = reputation / price* ratio.
    """

    buyer_type = "reputation"

    def __init__(self, buyer_id: int, n_states: int, rng: np.random.Generator,
                 budget_per_round: float = 0.5) -> None:
        super().__init__(buyer_id, n_states, rng, budget_per_round)
        self.reputation: dict[int, float] = {}  # seller_id → score
        self.purchase_count: dict[int, int] = {}  # seller_id → count
        self._explore_prob = 0.15  # exploration rate

    def _rep_score(self, seller_id: int) -> float:
        n = self.purchase_count.get(seller_id, 0)
        if n == 0:
            return 0.5  # prior: neutral
        return self.reputation.get(seller_id, 0.0) / n

    def choose_offer(self, offers: list[Offer], round_idx: int) -> int | None:
        if not offers:
            return None

        affordable = [(i, o) for i, o in enumerate(offers) if o.price <= self.budget]
        if not affordable:
            return None

        # Explore: occasionally try a random seller
        if self.rng.random() < self._explore_prob:
            return affordable[self.rng.integers(len(affordable))][0]

        # Exploit: best value = reputation_score / price
        best_idx, best_val = None, -np.inf
        for i, o in affordable:
            rep = self._rep_score(o.seller_id)
            value = rep / max(o.price, 1e-6)
            if value > best_val:
                best_val = value
                best_idx = i
        return best_idx

    def update_reputation(self, seller_id: int, claimed_quality: float,
                          experienced_quality: float) -> None:
        """Update reputation after observing data quality.

        ``experienced_quality`` is in [0, 1] — the fraction of samples
        that matched the buyer's posterior mode (a proxy for quality).
        """
        if seller_id not in self.reputation:
            self.reputation[seller_id] = 0.0
            self.purchase_count[seller_id] = 0

        # Score: +1 if experienced quality matches claimed, penalise over-claim
        gap = claimed_quality - experienced_quality
        score = 1.0 - min(max(gap, 0.0), 1.0)  # 1 if honest, 0 if max over-claim
        self.reputation[seller_id] += score
        self.purchase_count[seller_id] += 1


# ── Analytical Buyer ──────────────────────────────────────────────
class AnalyticalBuyer(BaseBuyer):
    """Independently estimates data quality by comparing purchased data
    to own observations.

    Each round, the buyer also draws a small free "observation" sample
    (n=5) from the true environment and uses it to cross-validate
    purchased data.  The buyer computes value = estimated_quality / price
    and buys from the best-value seller.
    """

    buyer_type = "analytical"

    def __init__(self, buyer_id: int, n_states: int, rng: np.random.Generator,
                 budget_per_round: float = 0.5) -> None:
        super().__init__(buyer_id, n_states, rng, budget_per_round)
        self._quality_estimates: dict[int, list[float]] = {}  # seller_id → list of quality ests
        self._explore_prob = 0.15  # exploration rate

    def _estimated_quality(self, seller_id: int) -> float:
        ests = self._quality_estimates.get(seller_id, [])
        if not ests:
            return 0.5  # prior
        # Exponentially weighted average (recent matters more)
        weights = np.exp(np.linspace(-2, 0, len(ests)))
        return float(np.average(ests, weights=weights))

    def choose_offer(self, offers: list[Offer], round_idx: int) -> int | None:
        if not offers:
            return None

        affordable = [(i, o) for i, o in enumerate(offers) if o.price <= self.budget]
        if not affordable:
            return None

        # Explore: occasionally try a random seller to gather data
        if self.rng.random() < self._explore_prob:
            return affordable[self.rng.integers(len(affordable))][0]

        # Exploit: pick seller with best estimated_quality / price ratio
        best_idx, best_val = None, -np.inf
        for i, o in affordable:
            eq = self._estimated_quality(o.seller_id)
            value = eq / max(o.price, 1e-6)
            if value > best_val:
                best_val = value
                best_idx = i
        return best_idx

    def update_quality_estimate(self, seller_id: int, data_samples: NDArray,
                                observation_samples: NDArray) -> None:
        """Cross-validate data against own observations.

        Quality estimate = 1 - total_variation_distance(data_dist, obs_dist).
        """
        n = self.n_states
        data_dist = np.bincount(data_samples, minlength=n).astype(float)
        obs_dist = np.bincount(observation_samples, minlength=n).astype(float)

        # Normalise (add small smoothing)
        data_dist = (data_dist + 0.5) / (data_dist.sum() + 0.5 * n)
        obs_dist = (obs_dist + 0.5) / (obs_dist.sum() + 0.5 * n)

        tvd = 0.5 * np.sum(np.abs(data_dist - obs_dist))
        quality_est = float(1.0 - tvd)

        if seller_id not in self._quality_estimates:
            self._quality_estimates[seller_id] = []
        self._quality_estimates[seller_id].append(quality_est)


# ── Factory ───────────────────────────────────────────────────────
BUYER_TYPES = {
    "naive": NaiveBuyer,
    "reputation": ReputationBuyer,
    "analytical": AnalyticalBuyer,
}


def create_buyer(buyer_type: str, buyer_id: int, n_states: int,
                 rng: np.random.Generator, **kwargs: Any) -> BaseBuyer:
    cls = BUYER_TYPES[buyer_type]
    return cls(buyer_id=buyer_id, n_states=n_states, rng=rng, **kwargs)
