# tests/test_market.py
import numpy as np
from src.market import LogitMarket, MARKET_PRESETS


def test_logit_demand_sums_to_one():
    """Demand probabilities must sum to 1."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    prices = np.array([1.3, 1.5])
    demand = market.compute_demand(prices)
    assert abs(demand.sum() - 1.0) < 1e-10


def test_logit_demand_lower_price_higher_demand():
    """Lower price should get higher demand share."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    prices = np.array([1.2, 1.8])
    demand = market.compute_demand(prices)
    assert demand[0] > demand[1]


def test_logit_demand_equal_prices_equal_demand():
    """Equal prices should yield equal demand."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    prices = np.array([1.5, 1.5])
    demand = market.compute_demand(prices)
    assert abs(demand[0] - demand[1]) < 1e-10


def test_profit_at_cost_is_zero():
    """Selling at marginal cost yields zero profit."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    prices = np.array([1.0, 1.5])
    profits = market.compute_profits(prices)
    assert abs(profits[0]) < 1e-10


def test_nash_price_above_cost():
    """Nash equilibrium price must be above marginal cost."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    nash = market.nash_price()
    assert nash > 1.0


def test_monopoly_above_nash():
    """Monopoly (collusive) price must be above Nash."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    nash = market.nash_price()
    monopoly = market.monopoly_price()
    assert monopoly > nash


def test_price_grid():
    """Price grid should have correct size and bounds."""
    market = LogitMarket(n_sellers=2, alpha=0.25, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    assert len(market.price_grid) == 15
    assert market.price_grid[0] == 1.0
    assert market.price_grid[-1] == 2.0


def test_market_presets_exist():
    """All three domain presets should be defined."""
    assert "e-commerce" in MARKET_PRESETS
    assert "ride-share" in MARKET_PRESETS
    assert "commodity" in MARKET_PRESETS


def test_market_from_preset():
    """Should create a market from a preset name."""
    market = LogitMarket.from_preset("e-commerce")
    assert market.n_sellers == 2
    assert market.alpha == 0.25
