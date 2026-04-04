# tests/test_market.py
import numpy as np
from src.market import LogitMarket, MARKET_PRESETS
from src.shocks import CostShock, DemandShock


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
    market = LogitMarket(n_sellers=2, alpha=3.0, costs=[1.0, 1.0],
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
    assert market.alpha == 3.0


def test_cost_shock_modifies_cost():
    """Cost shock should increase one seller's cost."""
    market = LogitMarket(n_sellers=2, alpha=3.0, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    shock = CostShock(seller_id=0, cost_increase=0.3, trigger_round=300_000,
                      total_rounds=500_000)
    original_cost = market.costs[0]
    shock.apply(market)
    assert market.costs[0] == original_cost + 0.3
    assert market.costs[1] == 1.0  # other seller unaffected


def test_cost_shock_trigger_timing():
    """Cost shock should only trigger at the right round."""
    shock = CostShock(seller_id=0, cost_increase=0.3, trigger_round=300_000,
                      total_rounds=500_000)
    assert shock.should_trigger(299_999) is False
    assert shock.should_trigger(300_000) is True
    assert shock.should_trigger(300_001) is False  # only triggers once


def test_demand_shock_modifies_alpha():
    """Demand shock should shift the market's price sensitivity."""
    market = LogitMarket(n_sellers=2, alpha=3.0, costs=[1.0, 1.0],
                         price_min=1.0, price_max=2.0, price_grid_size=15)
    shock = DemandShock(alpha_shift=0.6, trigger_round=400_000,
                        total_rounds=500_000)
    shock.apply(market)
    assert abs(market.alpha - 3.6) < 1e-10
