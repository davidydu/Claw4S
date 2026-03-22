"""Core simulation engine for emergent norm dynamics.

Runs a population of agents through repeated pairwise interactions in a
coordination game. Each round, a random pair is selected; both choose an
action and receive payoffs. The simulation records action choices and
payoffs for downstream metric computation.

References:
    Shoham, Y. & Tennenholtz, M. (1997). "On the Emergence of Social
    Conventions: Modeling, Analysis, and Simulations."
"""

from __future__ import annotations

import numpy as np

from src.agents import Agent, AgentType, create_population
from src.game import GameConfig, NUM_ACTIONS
from src.metrics import (
    norm_convergence_time,
    norm_diversity,
    norm_efficiency,
    norm_fragility,
)


def _get_population_counts(agents: list[Agent]) -> np.ndarray:
    """Aggregate action counts across the population."""
    counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for agent in agents:
        counts[agent.last_action] += 1
    return counts


def run_simulation(
    game: GameConfig,
    composition: dict[AgentType, int],
    total_rounds: int,
    seed: int,
) -> dict:
    """Run a single simulation and return raw trajectory data.

    Parameters
    ----------
    game : GameConfig
        The coordination game to play.
    composition : dict[AgentType, int]
        Number of each agent type in the population.
    total_rounds : int
        Number of pairwise interactions to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        action_history : ndarray of shape (total_rounds,)
        payoff_history : ndarray of shape (total_rounds,)
        agents : list[Agent] (final state)
    """
    rng = np.random.default_rng(seed)
    agents = create_population(composition, rng)
    n_agents = len(agents)

    action_history = np.empty(total_rounds, dtype=np.int32)
    payoff_history = np.empty(total_rounds, dtype=np.float64)

    for t in range(total_rounds):
        # Select a random pair (without replacement)
        idx = rng.choice(n_agents, size=2, replace=False)
        agent_i, agent_j = agents[idx[0]], agents[idx[1]]

        pop_counts = _get_population_counts(agents)

        action_i = agent_i.choose_action(pop_counts, rng)
        action_j = agent_j.choose_action(pop_counts, rng)

        payoff_i, payoff_j = game.payoff(action_i, action_j)

        agent_i.update(action_i, payoff_i)
        agent_j.update(action_j, payoff_j)

        # Record the pair's actions (use action_i as representative)
        action_history[t] = action_i
        payoff_history[t] = (payoff_i + payoff_j) / 2.0

    return {
        "action_history": action_history,
        "payoff_history": payoff_history,
        "agents": agents,
    }


def compute_sim_metrics(
    game: GameConfig,
    composition: dict[AgentType, int],
    total_rounds: int,
    seed: int,
) -> dict:
    """Run simulation and compute all four metrics.

    Returns a dict with simulation parameters and metric values.
    """
    result = run_simulation(game, composition, total_rounds, seed)
    action_history = result["action_history"]
    payoff_history = result["payoff_history"]

    optimal = game.optimal_welfare()

    # Fragility: run additional sims with increasing innovator fractions
    total_agents = sum(composition.values())
    innovator_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    frag_histories: list[np.ndarray] = []

    for frac in innovator_fractions:
        n_innovators = max(1, int(total_agents * frac))
        # Replace some agents with innovators
        frag_comp = dict(composition)
        # Remove agents proportionally from non-innovator types
        non_innov_types = [t for t in frag_comp if t != AgentType.INNOVATOR]
        remaining = total_agents - n_innovators
        if non_innov_types and remaining > 0:
            frag_comp_new: dict[AgentType, int] = {}
            non_innov_total = sum(frag_comp.get(t, 0) for t in non_innov_types)
            if non_innov_total > 0:
                for t in non_innov_types:
                    share = frag_comp.get(t, 0) / non_innov_total
                    frag_comp_new[t] = max(0, int(remaining * share))
            # Ensure we hit the target population size
            assigned = sum(frag_comp_new.values())
            if assigned < remaining and non_innov_types:
                frag_comp_new[non_innov_types[0]] += remaining - assigned
            frag_comp_new[AgentType.INNOVATOR] = n_innovators
        else:
            frag_comp_new = {AgentType.INNOVATOR: total_agents}

        frag_result = run_simulation(game, frag_comp_new, total_rounds, seed + 1000)
        frag_histories.append(frag_result["action_history"])

    convergence = norm_convergence_time(action_history, total_rounds)
    efficiency = norm_efficiency(payoff_history, optimal)
    diversity = norm_diversity(action_history)
    fragility = norm_fragility(
        action_history, innovator_fractions, frag_histories
    )

    return {
        "game": game.name,
        "composition": {t.value: n for t, n in composition.items()},
        "population_size": total_agents,
        "total_rounds": total_rounds,
        "seed": seed,
        "convergence_time": convergence,
        "efficiency": round(efficiency, 4),
        "diversity": diversity,
        "fragility": round(fragility, 2),
    }
