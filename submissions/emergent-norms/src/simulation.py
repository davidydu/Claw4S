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


def _allocate_non_innovators(
    composition: dict[AgentType, int],
    remaining: int,
) -> dict[AgentType, int]:
    """Allocate remaining slots across non-innovators proportionally."""
    non_innov_types = [t for t, n in composition.items() if t != AgentType.INNOVATOR and n > 0]
    if remaining <= 0 or not non_innov_types:
        return {}

    total_non_innov = sum(composition[t] for t in non_innov_types)
    if total_non_innov == 0:
        return {}

    allocated: dict[AgentType, int] = {}
    fractional_parts: list[tuple[float, AgentType]] = []
    assigned = 0

    for t in non_innov_types:
        exact = remaining * (composition[t] / total_non_innov)
        count = int(np.floor(exact))
        allocated[t] = count
        assigned += count
        fractional_parts.append((exact - count, t))

    to_assign = remaining - assigned
    if to_assign > 0:
        for _, t in sorted(fractional_parts, reverse=True)[:to_assign]:
            allocated[t] += 1

    return allocated


def _build_fragility_composition(
    composition: dict[AgentType, int],
    total_agents: int,
    requested_fraction: float,
) -> tuple[dict[AgentType, int], float]:
    """Build a perturbation composition for fragility without reducing innovators.

    requested_fraction is interpreted as target innovator share. If the baseline
    already has a higher share, we keep the baseline innovator count.
    """
    baseline_innovators = composition.get(AgentType.INNOVATOR, 0)
    requested_innovators = max(1, int(total_agents * requested_fraction))
    target_innovators = min(total_agents, max(baseline_innovators, requested_innovators))

    remaining = total_agents - target_innovators
    frag_comp = _allocate_non_innovators(composition, remaining)
    frag_comp[AgentType.INNOVATOR] = target_innovators
    return frag_comp, target_innovators / total_agents


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

    # Fragility: run additional sims with increasing innovator fractions.
    # For innovator-heavy baselines, do not reduce innovator share.
    total_agents = sum(composition.values())
    requested_innovator_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    effective_innovator_fractions: list[float] = []
    frag_histories: list[np.ndarray] = []
    seen_innovator_counts: set[int] = set()

    for frac in requested_innovator_fractions:
        frag_comp, effective_fraction = _build_fragility_composition(
            composition, total_agents, frac
        )
        innovator_count = frag_comp[AgentType.INNOVATOR]
        if innovator_count in seen_innovator_counts:
            continue
        seen_innovator_counts.add(innovator_count)

        frag_result = run_simulation(game, frag_comp, total_rounds, seed + 1000)
        frag_histories.append(frag_result["action_history"])
        effective_innovator_fractions.append(effective_fraction)

    convergence = norm_convergence_time(action_history, total_rounds)
    efficiency = norm_efficiency(payoff_history, optimal)
    diversity = norm_diversity(action_history)
    fragility = norm_fragility(
        action_history, effective_innovator_fractions, frag_histories
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
