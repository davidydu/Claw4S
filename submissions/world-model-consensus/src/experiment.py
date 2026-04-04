"""
Experiment configuration, simulation runner, and composition definitions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.game import CoordinationGame
from src.agents import make_agent, BaseAgent


# ======================================================================
# Compositions
# ======================================================================

# Each composition maps agent index -> type name.
# The actual number of agents is determined by n_agents; compositions
# are defined as *patterns* that repeat / truncate to fit.

COMPOSITIONS: Dict[str, List[str]] = {
    "all_adaptive":      ["adaptive"] * 10,   # truncated to N
    "all_stubborn":      ["stubborn"] * 10,
    "mixed":             ["adaptive", "adaptive", "stubborn", "stubborn"] * 3,
    "leader_followers":  ["leader"] + ["follower"] * 9,
}


def get_composition(name: str, n_agents: int) -> List[str]:
    """Return a list of agent-type strings of length *n_agents*."""
    pattern = COMPOSITIONS[name]
    return [pattern[i % len(pattern)] for i in range(n_agents)]


# ======================================================================
# Simulation config
# ======================================================================

@dataclass
class SimulationConfig:
    n_agents: int = 4
    n_actions: int = 5
    disagreement: float = 0.0
    composition: str = "all_adaptive"
    n_rounds: int = 10_000
    seed: int = 42


# ======================================================================
# Simulation result
# ======================================================================

@dataclass
class SimulationResult:
    config: SimulationConfig
    action_history: np.ndarray    # (n_rounds, n_agents) int
    payoff_history: np.ndarray    # (n_rounds, n_agents) float
    coordinated: np.ndarray       # (n_rounds,) bool — did everyone match?
    preferred_actions: List[int]  # initial preferred action per agent


# ======================================================================
# Simulation runner
# ======================================================================

def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    """Run one complete simulation and return the result."""

    game = CoordinationGame(
        n_agents=cfg.n_agents,
        n_actions=cfg.n_actions,
        disagreement=cfg.disagreement,
        seed=cfg.seed,
    )

    comp = get_composition(cfg.composition, cfg.n_agents)
    agents: List[BaseAgent] = []
    for i, atype in enumerate(comp):
        agents.append(make_agent(atype, agent_id=i,
                                 prior=game.priors[i], seed=cfg.seed))

    preferred = game.preferred_actions()

    action_history = np.empty((cfg.n_rounds, cfg.n_agents), dtype=np.int32)
    payoff_history = np.empty((cfg.n_rounds, cfg.n_agents), dtype=np.float64)
    coordinated = np.empty(cfg.n_rounds, dtype=bool)

    for t in range(cfg.n_rounds):
        actions = [a.choose_action() for a in agents]
        payoffs = game.payoff(actions)

        action_history[t] = actions
        payoff_history[t] = payoffs
        coordinated[t] = payoffs[0] > 0.5  # all-or-nothing game

        for a in agents:
            a.update(actions)

    return SimulationResult(
        config=cfg,
        action_history=action_history,
        payoff_history=payoff_history,
        coordinated=coordinated,
        preferred_actions=preferred,
    )


# ======================================================================
# Full experiment matrix
# ======================================================================

DISAGREEMENT_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]
GROUP_SIZES = [3, 4, 6]
SEEDS = [42, 123, 7]

def build_experiment_matrix() -> List[SimulationConfig]:
    """Build the full 4 x 7 x 3 x 3 = 252 simulation configs."""
    configs = []
    for comp_name in COMPOSITIONS:
        for d in DISAGREEMENT_LEVELS:
            for n in GROUP_SIZES:
                for s in SEEDS:
                    configs.append(SimulationConfig(
                        n_agents=n,
                        n_actions=5,
                        disagreement=d,
                        composition=comp_name,
                        n_rounds=10_000,
                        seed=s,
                    ))
    return configs
