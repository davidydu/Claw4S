"""DataMarketplace — order matching, transactions, and round execution.

Orchestrates the interaction between sellers, buyers, and the hidden
environment over multiple rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environment import DataEnvironment
from src.sellers import BaseSeller, Offer
from src.buyers import BaseBuyer, ReputationBuyer, AnalyticalBuyer


# ── Transaction record ────────────────────────────────────────────
@dataclass
class Transaction:
    """One completed purchase in a single round."""

    round_idx: int
    buyer_id: int
    seller_id: int
    price: float
    claimed_quality: float
    actual_quality: float
    decision_value: float
    buyer_type: str
    seller_type: str


# ── Information regime ────────────────────────────────────────────
REGIMES = {"transparent", "opaque", "partial"}


# ── Marketplace ───────────────────────────────────────────────────
class DataMarketplace:
    """Simulates a multi-round data marketplace.

    Parameters
    ----------
    env : DataEnvironment
        The hidden environment.
    sellers : list[BaseSeller]
        Participating sellers.
    buyers : list[BaseBuyer]
        Participating buyers.
    info_regime : str
        One of ``"transparent"`` (quality visible before purchase),
        ``"opaque"`` (only claims visible), or ``"partial"`` (quality
        revealed after purchase).
    data_samples_per_purchase : int
        Number of data samples delivered per transaction.
    observation_samples : int
        Number of free observation samples Analytical buyers get each round.
    """

    def __init__(
        self,
        env: DataEnvironment,
        sellers: list[BaseSeller],
        buyers: list[BaseBuyer],
        info_regime: str = "opaque",
        data_samples_per_purchase: int = 20,
        observation_samples: int = 5,
    ) -> None:
        assert info_regime in REGIMES, f"Unknown regime {info_regime}"
        self.env = env
        self.sellers = sellers
        self.buyers = buyers
        self.info_regime = info_regime
        self.data_samples_per_purchase = data_samples_per_purchase
        self.observation_samples = observation_samples
        self.transactions: list[Transaction] = []

    # ------------------------------------------------------------------
    # Core round execution
    # ------------------------------------------------------------------

    def run_round(self, round_idx: int) -> list[Transaction]:
        """Execute one marketplace round.  Returns transactions."""

        # 1. Sellers post offers
        offers = [s.make_offer(round_idx) for s in self.sellers]

        # 2. If transparent, buyers see actual quality
        visible_offers = []
        for o in offers:
            if self.info_regime == "transparent":
                # Override claimed_quality with actual
                visible_offers.append(Offer(
                    seller_id=o.seller_id,
                    seller_type=o.seller_type,
                    price=o.price,
                    claimed_quality=o.actual_quality,
                    actual_quality=o.actual_quality,
                ))
            else:
                visible_offers.append(o)

        # 3. Each buyer picks an offer (or abstains)
        round_txns: list[Transaction] = []
        seller_sold: dict[int, bool] = {s.seller_id: False for s in self.sellers}

        for buyer in self.buyers:
            choice = buyer.choose_offer(visible_offers, round_idx)
            if choice is None:
                continue

            offer = offers[choice]  # use original (not visible) for actual_quality
            seller = self.sellers[choice]

            # 4. Generate data and deliver to buyer
            data = self.env.sample_noisy(offer.actual_quality, n=self.data_samples_per_purchase)
            buyer.update_belief(data)

            # 5. Compute buyer decision value (how good their belief now is)
            dv = self.env.decision_value(buyer.belief)

            # 6. Record transaction
            buyer.record_purchase(offer.price, dv)
            seller.record_sale(offer.price)
            seller_sold[seller.seller_id] = True

            txn = Transaction(
                round_idx=round_idx,
                buyer_id=buyer.buyer_id,
                seller_id=seller.seller_id,
                price=offer.price,
                claimed_quality=offer.claimed_quality,
                actual_quality=offer.actual_quality,
                decision_value=dv,
                buyer_type=buyer.buyer_type,
                seller_type=seller.seller_type,
            )
            round_txns.append(txn)
            self.transactions.append(txn)

            # 7. Post-purchase feedback (regime-dependent)
            if self.info_regime in ("partial", "transparent"):
                # Buyer learns actual quality after purchase
                if isinstance(buyer, ReputationBuyer):
                    mode = int(np.argmax(buyer.belief))
                    experienced_q = float(np.mean(data == mode))
                    buyer.update_reputation(seller.seller_id, offer.claimed_quality, experienced_q)
                if isinstance(buyer, AnalyticalBuyer):
                    obs = self.env.sample_true(n=self.observation_samples)
                    buyer.update_quality_estimate(seller.seller_id, data, obs)
            # In opaque regime: no quality feedback — buyers only learn
            # from the Bayesian update on the data itself

        # 8. Record no-sale for sellers that didn't sell
        for seller in self.sellers:
            if not seller_sold[seller.seller_id]:
                seller.record_no_sale()

        return round_txns

    def run(self, n_rounds: int) -> list[Transaction]:
        """Run the full simulation for *n_rounds*."""
        for r in range(n_rounds):
            self.run_round(r)
        return self.transactions

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def price_quality_correlation(self) -> float:
        """Pearson correlation between transaction price and actual quality."""
        if len(self.transactions) < 2:
            return 0.0
        prices = np.array([t.price for t in self.transactions])
        quals = np.array([t.actual_quality for t in self.transactions])
        if np.std(prices) < 1e-12 or np.std(quals) < 1e-12:
            return 0.0
        return float(np.corrcoef(prices, quals)[0, 1])

    def buyer_welfare(self) -> dict[str, float]:
        """Average decision value per buyer type."""
        from collections import defaultdict
        by_type: dict[str, list[float]] = defaultdict(list)
        for t in self.transactions:
            by_type[t.buyer_type].append(t.decision_value)
        return {k: float(np.mean(v)) for k, v in by_type.items()}

    def market_efficiency(self) -> float:
        """Allocative efficiency: fraction of transactions where the buyer
        chose the highest-quality available seller.

        Captures whether the market directs trade toward good sellers,
        independent of Bayesian convergence.
        """
        if not self.transactions:
            return 0.0
        best_quality = max(s.quality for s in self.sellers)
        correct = sum(1 for t in self.transactions
                      if abs(t.actual_quality - best_quality) < 1e-9)
        return float(correct / len(self.transactions))

    def price_efficiency(self) -> float:
        """Average quality-per-unit-price across transactions.

        Higher means buyers get more quality per dollar spent.
        Capped at 2.0 for normalisation.
        """
        if not self.transactions:
            return 0.0
        ratios = [t.actual_quality / max(t.price, 1e-6) for t in self.transactions]
        return float(min(np.mean(ratios), 2.0))

    def surplus_rate(self) -> float:
        """Average buyer surplus (decision_value - price) per transaction."""
        if not self.transactions:
            return 0.0
        return float(np.mean([t.decision_value - t.price for t in self.transactions]))

    def lemons_index(self) -> float:
        """Fraction of market share held by low-quality sellers (quality < 0.4)."""
        if not self.transactions:
            return 0.0
        low_q = sum(1 for t in self.transactions if t.actual_quality < 0.4)
        return float(low_q / len(self.transactions))

    def reputation_accuracy(self) -> float:
        """How well reputation scores predict actual quality (across RepBuyers)."""
        corr_data: list[tuple[float, float]] = []
        for buyer in self.buyers:
            if isinstance(buyer, ReputationBuyer):
                for sid in buyer.reputation:
                    rep = buyer._rep_score(sid)
                    # Find the actual quality of this seller
                    for s in self.sellers:
                        if s.seller_id == sid:
                            corr_data.append((rep, s.quality))
                            break
        if len(corr_data) < 2:
            return 0.0
        reps, quals = zip(*corr_data)
        reps, quals = np.array(reps), np.array(quals)
        if np.std(reps) < 1e-12 or np.std(quals) < 1e-12:
            return 0.0
        return float(np.corrcoef(reps, quals)[0, 1])
