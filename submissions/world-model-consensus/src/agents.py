"""
Agent types for the coordination game.

Each agent maintains a belief distribution over K actions and selects
an action each round.  After each round agents observe all chosen actions
and (optionally) update beliefs.

Agent types
-----------
- StubbornAgent  : never updates beliefs; always plays prior-best action.
- AdaptiveAgent  : updates beliefs via EMA toward observed group behaviour.
- LeaderAgent    : like Stubborn but conceptually a focal-point creator.
- FollowerAgent  : high learning rate; quickly adopts the most popular action.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List


class BaseAgent(ABC):
    """Abstract base class for coordination-game agents."""

    def __init__(self, agent_id: int, prior: np.ndarray, seed: int = 0):
        """
        Parameters
        ----------
        agent_id : int
            Unique identifier for this agent.
        prior : np.ndarray
            Initial belief distribution over K actions (sums to 1).
        seed : int
            Random seed for any stochastic action selection.
        """
        self.agent_id = agent_id
        self.prior = prior.copy()
        self.beliefs = prior.copy()
        self.n_actions = len(prior)
        self.rng = np.random.default_rng(seed + agent_id * 1000)
        self.history: List[int] = []

    # ------------------------------------------------------------------
    @abstractmethod
    def choose_action(self) -> int:
        """Select an action for this round."""

    @abstractmethod
    def update(self, actions: List[int]) -> None:
        """Update beliefs after observing all agents' actions."""

    # ------------------------------------------------------------------
    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

    def preferred_action(self) -> int:
        return int(np.argmax(self.beliefs))

    def __repr__(self) -> str:
        return f"{self.agent_type}(id={self.agent_id}, pref={self.preferred_action()})"


# ======================================================================
# Concrete agent types
# ======================================================================


class StubbornAgent(BaseAgent):
    """Never updates beliefs — always plays its prior-best action."""

    def choose_action(self) -> int:
        action = int(np.argmax(self.prior))
        self.history.append(action)
        return action

    def update(self, actions: List[int]) -> None:
        pass  # beliefs never change


class AdaptiveAgent(BaseAgent):
    """Updates beliefs via exponential moving average (EMA).

    After each round, the agent observes the empirical action
    distribution and blends it into beliefs:

        beliefs <- (1 - lr) * beliefs + lr * empirical

    Uses epsilon-greedy exploration to break symmetry deadlocks.
    """

    def __init__(self, agent_id: int, prior: np.ndarray, seed: int = 0,
                 learning_rate: float = 0.1, epsilon: float = 0.05):
        super().__init__(agent_id, prior, seed)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def choose_action(self) -> int:
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            action = int(np.argmax(self.beliefs))
        self.history.append(action)
        return action

    def update(self, actions: List[int]) -> None:
        empirical = np.zeros(self.n_actions)
        for a in actions:
            empirical[a] += 1.0
        empirical /= len(actions)

        lr = self.learning_rate
        self.beliefs = (1.0 - lr) * self.beliefs + lr * empirical
        # Re-normalise (numerically safe)
        self.beliefs = self.beliefs / self.beliefs.sum()


class LeaderAgent(BaseAgent):
    """Focal-point creator — always plays its prior-best action.

    Behaviourally identical to StubbornAgent, but semantically represents
    an agent that *intends* to create a focal point for others to follow.
    """

    def choose_action(self) -> int:
        action = int(np.argmax(self.prior))
        self.history.append(action)
        return action

    def update(self, actions: List[int]) -> None:
        pass  # leaders don't update


class FollowerAgent(BaseAgent):
    """High learning rate agent that quickly adopts the most popular action.

    Uses a much higher EMA learning rate than AdaptiveAgent (default 0.5),
    so it converges to the group's modal action within a few rounds.
    Uses epsilon-greedy exploration to break symmetry deadlocks.
    """

    def __init__(self, agent_id: int, prior: np.ndarray, seed: int = 0,
                 learning_rate: float = 0.5, epsilon: float = 0.02):
        super().__init__(agent_id, prior, seed)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def choose_action(self) -> int:
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            action = int(np.argmax(self.beliefs))
        self.history.append(action)
        return action

    def update(self, actions: List[int]) -> None:
        empirical = np.zeros(self.n_actions)
        for a in actions:
            empirical[a] += 1.0
        empirical /= len(actions)

        lr = self.learning_rate
        self.beliefs = (1.0 - lr) * self.beliefs + lr * empirical
        self.beliefs = self.beliefs / self.beliefs.sum()


# ======================================================================
# Factory
# ======================================================================

AGENT_TYPES = {
    "stubborn": StubbornAgent,
    "adaptive": AdaptiveAgent,
    "leader": LeaderAgent,
    "follower": FollowerAgent,
}


def make_agent(agent_type: str, agent_id: int, prior: np.ndarray,
               seed: int = 0, **kwargs) -> BaseAgent:
    """Construct an agent by type name."""
    cls = AGENT_TYPES[agent_type]
    return cls(agent_id=agent_id, prior=prior, seed=seed, **kwargs)
