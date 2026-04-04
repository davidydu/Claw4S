"""Agent types for the reward-hacking propagation simulation.

Agent types
-----------
- explorer: high exploration rate, likely to discover hacks independently
- imitator: copies successful neighbors' strategies readily
- conservative: slow to change behavior
- monitor: tracks divergence between proxy and true reward; does not adopt hacks

Each agent has:
  - strategy: "honest" or "hack"
  - proxy_reward: observed reward (hack gives higher proxy)
  - true_reward: ground-truth reward (hack gives lower true)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --- Reward parameters (module-level constants) ---
HONEST_PROXY_MEAN = 1.0
HONEST_PROXY_STD = 0.3
HONEST_TRUE_MEAN = 1.0
HONEST_TRUE_STD = 0.3

HACK_PROXY_MEAN = 1.5      # hack looks moderately better on proxy metric
HACK_PROXY_STD = 0.4
HACK_TRUE_MEAN = 0.3       # but is terrible on true metric
HACK_TRUE_STD = 0.15

# Adoption probability scaling: P(adopt) = sigmoid(beta * reward_gap)
# beta controls how sensitive agents are to reward differences
# Lower values = slower spread, allowing topology and monitors to matter
BETA_EXPLORER = 0.5
BETA_IMITATOR = 1.2
BETA_CONSERVATIVE = 0.15
BETA_MONITOR = 0.0  # monitors never adopt


@dataclass
class Agent:
    """A single agent in the simulation."""
    agent_id: int
    agent_type: str  # "explorer", "imitator", "conservative", "monitor"
    strategy: str = "honest"
    proxy_reward: float = 0.0
    true_reward: float = 0.0
    rounds_hacking: int = 0
    # For monitors: track observed divergence
    divergence_history: list[float] = field(default_factory=list)

    @property
    def beta(self) -> float:
        if self.agent_type == "explorer":
            return BETA_EXPLORER
        elif self.agent_type == "imitator":
            return BETA_IMITATOR
        elif self.agent_type == "conservative":
            return BETA_CONSERVATIVE
        else:  # monitor
            return BETA_MONITOR

    @property
    def is_hacking(self) -> bool:
        return self.strategy == "hack"


def create_agent_population(
    n: int,
    monitor_fraction: float,
    rng: np.random.Generator,
) -> list[Agent]:
    """Create a population of N agents with specified monitor fraction.

    Non-monitor agents are split equally among explorer, imitator, conservative.

    Parameters
    ----------
    n : int
        Total number of agents.
    monitor_fraction : float
        Fraction of agents that are monitors (0.0 to 1.0).
    rng : numpy.random.Generator
        RNG for shuffling.

    Returns
    -------
    agents : list of Agent
    """
    n_monitors = int(round(n * monitor_fraction))
    n_others = n - n_monitors

    types: list[str] = []
    # Split remaining agents into 3 types as evenly as possible
    base, rem = divmod(n_others, 3)
    for i, t in enumerate(["explorer", "imitator", "conservative"]):
        count = base + (1 if i < rem else 0)
        types.extend([t] * count)
    types.extend(["monitor"] * n_monitors)

    # Shuffle so types are randomly distributed
    perm = rng.permutation(len(types))
    shuffled = [types[i] for i in perm]

    return [Agent(agent_id=i, agent_type=shuffled[i]) for i in range(n)]


def sample_reward(agent: Agent, rng: np.random.Generator) -> tuple[float, float]:
    """Sample (proxy_reward, true_reward) for an agent based on its strategy."""
    if agent.strategy == "hack":
        proxy = rng.normal(HACK_PROXY_MEAN, HACK_PROXY_STD)
        true = rng.normal(HACK_TRUE_MEAN, HACK_TRUE_STD)
    else:
        proxy = rng.normal(HONEST_PROXY_MEAN, HONEST_PROXY_STD)
        true = rng.normal(HONEST_TRUE_MEAN, HONEST_TRUE_STD)
    return float(proxy), float(true)
