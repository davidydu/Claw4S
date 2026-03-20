# src/experiment.py
"""Experiment runner for the pricing collusion simulation."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.market import LogitMarket, MARKET_PRESETS
from src.agents import create_agent
from src.shocks import CostShock, DemandShock


@dataclass
class ExperimentConfig:
    """Configuration for a single simulation run."""
    matchup: str            # e.g. "QQ", "QS", "Q-TFT"
    memory: int             # agent memory length M
    preset: str             # market preset name
    shocks: bool            # whether to inject shocks
    seed: int               # random seed
    total_rounds: int = 500_000


@dataclass
class SimulationResult:
    """Output of a single simulation run."""
    config: ExperimentConfig
    price_history: np.ndarray       # (T, N) price grid indices
    profit_history: np.ndarray      # (T, N) profits
    final_avg_price: float          # mean price in last 20%
    nash_price: float
    monopoly_price: float
    convergence_round: Optional[int] = None
    saved_states: Optional[list] = None
    agents: Optional[list] = None
    # Shock analysis
    pre_shock_price: Optional[float] = None
    post_shock_price: Optional[float] = None
    recovery_rounds: Optional[int] = None


# Matchup code -> (agent_type_0, agent_type_1)
MATCHUPS = {
    "QQ": ("q_learning", "q_learning"),
    "SS": ("sarsa", "sarsa"),
    "PG-PG": ("policy_gradient", "policy_gradient"),
    "QS": ("q_learning", "sarsa"),
    "Q-TFT": ("q_learning", "tit_for_tat"),
    "Q-Competitive": ("q_learning", "competitive"),
}


def _detect_convergence(price_history, window=1000, tolerance=0.01):
    """Find the first round where price grid indices stabilize.

    Operates on integer grid indices (0 to K-1). Tolerance is relative
    to the mean index — e.g., tolerance=0.01 requires std < 1% of mean index.
    """
    if len(price_history) < window * 2:
        return None
    for t in range(window, len(price_history) - window):
        seg = price_history[t:t + window]
        if seg.std(axis=0).max() < tolerance * seg.mean():
            return t
    return None


def run_simulation(config):
    """Run a single pricing simulation."""
    market = LogitMarket.from_preset(config.preset)
    agent_types = MATCHUPS[config.matchup]
    rng = np.random.default_rng(config.seed)

    agents = []
    for i, atype in enumerate(agent_types):
        agent_seed = int(rng.integers(0, 2**31))
        agents.append(create_agent(
            atype, agent_id=i, market=market, memory=config.memory,
            seed=agent_seed, total_rounds=config.total_rounds,
        ))

    # Set up shocks
    shocks = []
    if config.shocks:
        t_cost = int(config.total_rounds * 0.6)
        t_demand = int(config.total_rounds * 0.8)
        shocks.append(CostShock(seller_id=0, cost_increase=0.3,
                                trigger_round=t_cost,
                                total_rounds=config.total_rounds))
        shocks.append(DemandShock(alpha_shift=market.alpha * 0.2,  # 20% of current alpha
                                  trigger_round=t_demand,
                                  total_rounds=config.total_rounds))

    T = config.total_rounds
    N = market.n_sellers
    price_history = np.zeros((T, N), dtype=int)
    profit_history = np.zeros((T, N), dtype=np.float64)
    saved_states = None
    save_round = int(T * 0.9)

    # Shock analysis tracking
    pre_shock_prices = []
    post_shock_prices = []

    for t in range(T):
        # Check shocks
        for shock in shocks:
            if shock.should_trigger(t):
                shock.apply(market)

        # Each agent chooses action
        history_so_far = price_history[:t]
        actions = [agent.choose_action(history_so_far) for agent in agents]

        # Record
        price_history[t] = actions
        prices = market.price_grid[np.array(actions)]
        profits = market.compute_profits(prices)
        profit_history[t] = profits

        # Update agents
        next_history = price_history[:t + 1]
        for i, agent in enumerate(agents):
            agent.update(history_so_far, actions[i], profits[i], next_history)

        # Save states at T*0.9 for counterfactual auditor
        if t == save_round:
            saved_states = [agent.save_state() for agent in agents]

        # Track shock analysis
        if config.shocks:
            t_cost = int(T * 0.6)
            if t_cost - 1000 <= t < t_cost:
                pre_shock_prices.append(prices.mean())
            elif t_cost + 1000 <= t < t_cost + 2000:
                post_shock_prices.append(prices.mean())

    # Compute summary stats
    tail_start = int(T * 0.8)
    tail_prices = market.price_grid[price_history[tail_start:]]
    final_avg_price = float(tail_prices.mean())
    convergence_round = _detect_convergence(price_history)

    result = SimulationResult(
        config=config,
        price_history=price_history,
        profit_history=profit_history,
        final_avg_price=final_avg_price,
        nash_price=float(market.nash_price()),
        monopoly_price=float(market.monopoly_price()),
        convergence_round=convergence_round,
        saved_states=saved_states,
        agents=agents,
    )

    if config.shocks and pre_shock_prices and post_shock_prices:
        result.pre_shock_price = float(np.mean(pre_shock_prices))
        result.post_shock_price = float(np.mean(post_shock_prices))
        # Recovery: rounds after shock until price stabilizes again
        result.recovery_rounds = _detect_convergence(
            price_history[int(T * 0.6):], window=500
        )

    return result
