"""Data sellers — Honest, Strategic, Predatory.

Each seller has an intrinsic ``quality`` in [0, 1] and a ``production_cost``
that scales with quality.  Every round a seller posts an *offer*:
``(price, claimed_quality)`` and, if purchased, delivers data of its
actual quality via ``DataEnvironment.sample_noisy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Offer dataclass ───────────────────────────────────────────────
@dataclass
class Offer:
    """A posted offer in the marketplace."""

    seller_id: int
    seller_type: str
    price: float
    claimed_quality: float
    actual_quality: float  # hidden from buyers in opaque regimes


# ── Base class ────────────────────────────────────────────────────
class BaseSeller:
    """Abstract seller base class."""

    seller_type: str = "base"

    def __init__(self, seller_id: int, quality: float, rng: np.random.Generator) -> None:
        self.seller_id = seller_id
        self.quality = float(quality)  # actual quality ∈ [0, 1]
        self.production_cost = 0.2 * self.quality  # cost ∝ quality
        self.rng = rng
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.total_sales = 0
        self.round_history: list[dict[str, Any]] = []

    @property
    def profit(self) -> float:
        return self.total_revenue - self.total_cost

    def make_offer(self, round_idx: int) -> Offer:
        raise NotImplementedError

    def record_sale(self, price: float) -> None:
        self.total_revenue += price
        self.total_cost += self.production_cost
        self.total_sales += 1

    def record_no_sale(self) -> None:
        pass  # no cost if no sale


# ── Honest Seller ─────────────────────────────────────────────────
class HonestSeller(BaseSeller):
    """Prices proportional to quality, never misrepresents.

    ``price = quality * base_price_multiplier``
    ``claimed_quality = actual_quality``
    """

    seller_type = "honest"

    def __init__(self, seller_id: int, quality: float, rng: np.random.Generator,
                 base_price: float = 0.5) -> None:
        super().__init__(seller_id, quality, rng)
        self.base_price = base_price

    def make_offer(self, round_idx: int) -> Offer:
        price = self.quality * self.base_price
        return Offer(
            seller_id=self.seller_id,
            seller_type=self.seller_type,
            price=price,
            claimed_quality=self.quality,
            actual_quality=self.quality,
        )


# ── Strategic Seller ──────────────────────────────────────────────
class StrategicSeller(BaseSeller):
    """Adapts price and claims to maximise profit.

    Starts by over-claiming quality.  If sales drop, gradually reduces
    claimed quality (but not actual quality).  Price tracks claimed quality.
    """

    seller_type = "strategic"

    def __init__(self, seller_id: int, quality: float, rng: np.random.Generator,
                 base_price: float = 0.5) -> None:
        super().__init__(seller_id, quality, rng)
        self.base_price = base_price
        # Start by claiming much higher quality than actual
        self.claimed_q = min(1.0, self.quality + 0.4)
        self._recent_sales: list[bool] = []

    def make_offer(self, round_idx: int) -> Offer:
        # Adapt claims based on recent success (every 50 rounds)
        if len(self._recent_sales) >= 50:
            hit_rate = sum(self._recent_sales[-50:]) / 50
            if hit_rate < 0.2:
                # Not selling — lower claims slightly
                self.claimed_q = max(self.quality, self.claimed_q - 0.05)
            elif hit_rate > 0.6:
                # Selling well — try inflating more
                self.claimed_q = min(1.0, self.claimed_q + 0.02)

        price = self.claimed_q * self.base_price
        return Offer(
            seller_id=self.seller_id,
            seller_type=self.seller_type,
            price=price,
            claimed_quality=self.claimed_q,
            actual_quality=self.quality,
        )

    def record_sale(self, price: float) -> None:
        super().record_sale(price)
        self._recent_sales.append(True)

    def record_no_sale(self) -> None:
        super().record_no_sale()
        self._recent_sales.append(False)


# ── Predatory Seller ──────────────────────────────────────────────
class PredatorySeller(BaseSeller):
    """Deliberately sells low-quality data at high prices.

    Always claims top quality regardless of actual quality.
    Targets markets where buyers cannot verify quality upfront.
    """

    seller_type = "predatory"

    def __init__(self, seller_id: int, quality: float, rng: np.random.Generator,
                 base_price: float = 0.5) -> None:
        super().__init__(seller_id, quality, rng)
        self.base_price = base_price

    def make_offer(self, round_idx: int) -> Offer:
        # Always claims max quality, prices at top tier
        return Offer(
            seller_id=self.seller_id,
            seller_type=self.seller_type,
            price=0.95 * self.base_price,  # just below the max-quality honest price
            claimed_quality=1.0,
            actual_quality=self.quality,
        )


# ── Factory ───────────────────────────────────────────────────────
SELLER_TYPES = {
    "honest": HonestSeller,
    "strategic": StrategicSeller,
    "predatory": PredatorySeller,
}


def create_seller(seller_type: str, seller_id: int, quality: float,
                  rng: np.random.Generator, **kwargs: Any) -> BaseSeller:
    cls = SELLER_TYPES[seller_type]
    return cls(seller_id=seller_id, quality=quality, rng=rng, **kwargs)
