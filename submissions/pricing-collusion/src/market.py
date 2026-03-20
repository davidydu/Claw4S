# src/market.py
"""Market simulation engine with logit demand model."""

import numpy as np
from scipy.optimize import minimize_scalar


class LogitMarket:
    """Parametric Bertrand competition with logit demand.

    Prices are normalized to marginal cost (unitless markup ratios).
    """

    def __init__(self, n_sellers, alpha, costs, price_min, price_max,
                 price_grid_size):
        self.n_sellers = n_sellers
        self.alpha = alpha
        self.costs = np.array(costs, dtype=np.float64)
        self.price_min = price_min
        self.price_max = price_max
        self.price_grid_size = price_grid_size
        self.price_grid = np.linspace(price_min, price_max, price_grid_size)

    @classmethod
    def from_preset(cls, preset_name):
        """Create a market from a named preset."""
        config = MARKET_PRESETS[preset_name]
        return cls(**config)

    def compute_demand(self, prices):
        """Logit demand: P(buy from i) proportional to exp(-alpha * p_i)."""
        utilities = -self.alpha * prices
        # Numerically stable softmax
        utilities = utilities - utilities.max()
        exp_u = np.exp(utilities)
        return exp_u / exp_u.sum()

    def compute_profits(self, prices):
        """Profit for each seller: (p_i - c_i) * demand_i."""
        demand = self.compute_demand(prices)
        return (prices - self.costs) * demand

    def nash_price(self):
        """Compute symmetric Nash equilibrium price for logit Bertrand.

        For symmetric firms with logit demand, the Nash FOC gives:
        p* = c + alpha * N / (N - 1) for N >= 2
        """
        n = self.n_sellers
        c = self.costs[0]  # symmetric case
        if n < 2:
            return self.price_max  # monopoly
        p_nash = c + self.alpha * n / (n - 1)
        return np.clip(p_nash, self.price_min, self.price_max)

    def monopoly_price(self):
        """Compute joint-profit-maximizing (collusive) price.

        Maximize total profit assuming all sellers set the same price.
        """
        def neg_total_profit(p):
            prices = np.full(self.n_sellers, p)
            return -self.compute_profits(prices).sum()

        result = minimize_scalar(neg_total_profit,
                                 bounds=(self.price_min, self.price_max),
                                 method="bounded")
        return result.x


MARKET_PRESETS = {
    "e-commerce": {
        "n_sellers": 2,
        "alpha": 0.25,
        "costs": [1.0, 1.0],
        "price_min": 1.0,
        "price_max": 2.0,
        "price_grid_size": 15,
    },
    "ride-share": {
        "n_sellers": 2,
        "alpha": 0.5,
        "costs": [1.0, 1.3],
        "price_min": 1.0,
        "price_max": 3.0,
        "price_grid_size": 15,
    },
    "commodity": {
        "n_sellers": 2,
        "alpha": 1.0,
        "costs": [1.0, 1.0],
        "price_min": 1.0,
        "price_max": 1.5,
        "price_grid_size": 15,
    },
}
