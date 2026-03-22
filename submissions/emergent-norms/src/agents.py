"""Agent types for the emergent norms simulation.

Four agent types model different behavioral strategies:
- Conformist: copies the most common action in the population
- Innovator: randomly explores new actions with probability epsilon
- Traditionalist: sticks to the first action that gave a good payoff
- Adaptive: uses EMA belief updates with softmax action selection

References:
    Axelrod, R. (1986). "An Evolutionary Approach to Norms."
    Young, H.P. (1993). "The Evolution of Conventions."
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from src.game import NUM_ACTIONS


class AgentType(str, Enum):
    CONFORMIST = "conformist"
    INNOVATOR = "innovator"
    TRADITIONALIST = "traditionalist"
    ADAPTIVE = "adaptive"


class Agent:
    """Base agent in the norm emergence simulation."""

    __slots__ = (
        "agent_type", "agent_id", "action_counts", "total_payoff",
        "num_interactions", "last_action",
        # Adaptive-specific
        "beliefs", "ema_alpha", "temperature",
        # Traditionalist-specific
        "anchor_action", "anchor_threshold",
        # Innovator-specific
        "epsilon",
    )

    def __init__(
        self,
        agent_type: AgentType,
        agent_id: int,
        rng: np.random.Generator,
        *,
        ema_alpha: float = 0.1,
        temperature: float = 1.0,
        anchor_threshold: float = 2.0,
        epsilon: float = 0.15,
    ) -> None:
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self.total_payoff = 0.0
        self.num_interactions = 0
        self.last_action = int(rng.integers(0, NUM_ACTIONS))

        # Adaptive agent: EMA beliefs over action values
        self.beliefs = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        self.ema_alpha = ema_alpha
        self.temperature = temperature

        # Traditionalist: lock onto first good action
        self.anchor_action: int | None = None
        self.anchor_threshold = anchor_threshold

        # Innovator: exploration rate
        self.epsilon = epsilon

    def choose_action(self, population_counts: np.ndarray, rng: np.random.Generator) -> int:
        """Select an action based on agent type and current beliefs."""
        if self.agent_type == AgentType.CONFORMIST:
            action = self._conformist_action(population_counts, rng)
        elif self.agent_type == AgentType.INNOVATOR:
            action = self._innovator_action(rng)
        elif self.agent_type == AgentType.TRADITIONALIST:
            action = self._traditionalist_action(rng)
        elif self.agent_type == AgentType.ADAPTIVE:
            action = self._adaptive_action(rng)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        self.last_action = action
        self.action_counts[action] += 1
        return action

    def update(self, action: int, payoff: float) -> None:
        """Update internal state after receiving a payoff."""
        self.total_payoff += payoff
        self.num_interactions += 1

        if self.agent_type == AgentType.ADAPTIVE:
            # EMA update: increase belief in the chosen action proportionally
            target = np.zeros(NUM_ACTIONS)
            target[action] = payoff
            self.beliefs = (1 - self.ema_alpha) * self.beliefs + self.ema_alpha * target

        elif self.agent_type == AgentType.TRADITIONALIST:
            if self.anchor_action is None and payoff >= self.anchor_threshold:
                self.anchor_action = action

    def _conformist_action(self, population_counts: np.ndarray, rng: np.random.Generator) -> int:
        """Play the most common action in the population."""
        total = population_counts.sum()
        if total == 0:
            return int(rng.integers(0, NUM_ACTIONS))
        # Break ties randomly
        max_count = population_counts.max()
        candidates = np.where(population_counts == max_count)[0]
        return int(rng.choice(candidates))

    def _innovator_action(self, rng: np.random.Generator) -> int:
        """Usually play last action, but explore with probability epsilon."""
        if rng.random() < self.epsilon:
            return int(rng.integers(0, NUM_ACTIONS))
        return self.last_action

    def _traditionalist_action(self, rng: np.random.Generator) -> int:
        """Stick to anchor action once found, otherwise explore."""
        if self.anchor_action is not None:
            return self.anchor_action
        return int(rng.integers(0, NUM_ACTIONS))

    def _adaptive_action(self, rng: np.random.Generator) -> int:
        """Softmax selection over EMA beliefs."""
        # Numerical stability: subtract max
        logits = self.beliefs / max(self.temperature, 1e-8)
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return int(rng.choice(NUM_ACTIONS, p=probs))

    @property
    def avg_payoff(self) -> float:
        if self.num_interactions == 0:
            return 0.0
        return self.total_payoff / self.num_interactions


def create_population(
    composition: dict[AgentType, int],
    rng: np.random.Generator,
) -> list[Agent]:
    """Create a population of agents with the given type composition."""
    agents: list[Agent] = []
    agent_id = 0
    for agent_type, count in composition.items():
        for _ in range(count):
            agents.append(Agent(agent_type, agent_id, rng))
            agent_id += 1
    return agents
