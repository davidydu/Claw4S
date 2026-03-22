"""Agent types for the information-sharing game.

Each agent type implements a disclosure strategy:
- Open: always shares 100%
- Secretive: never shares (0%)
- Reciprocal: matches average sharing of others (with momentum)
- Strategic: shares when expected benefit exceeds expected cost (EWA learning)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """Base class for information-sharing agents."""

    def __init__(self, agent_id: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.rng = rng

    @abstractmethod
    def choose_disclosure(
        self,
        round_num: int,
        history_own_payoffs: list[float],
        history_others_sharing: list[float],
    ) -> float:
        """Return disclosure level in [0, 1]."""

    @abstractmethod
    def type_name(self) -> str:
        """Return human-readable agent type."""


class OpenAgent(Agent):
    """Always shares 100% of private information."""

    def choose_disclosure(
        self,
        round_num: int,
        history_own_payoffs: list[float],
        history_others_sharing: list[float],
    ) -> float:
        return 1.0

    def type_name(self) -> str:
        return "Open"


class SecretiveAgent(Agent):
    """Never shares any private information."""

    def choose_disclosure(
        self,
        round_num: int,
        history_own_payoffs: list[float],
        history_others_sharing: list[float],
    ) -> float:
        return 0.0

    def type_name(self) -> str:
        return "Secretive"


class ReciprocalAgent(Agent):
    """Shares proportionally to what others shared last round.

    Uses exponential moving average of others' sharing rates with
    momentum parameter alpha = 0.1. Starts at 0.5 disclosure.
    """

    def __init__(self, agent_id: int, rng: np.random.Generator, alpha: float = 0.1):
        super().__init__(agent_id, rng)
        self.alpha = alpha
        self.ema_others_sharing = 0.5

    def choose_disclosure(
        self,
        round_num: int,
        history_own_payoffs: list[float],
        history_others_sharing: list[float],
    ) -> float:
        if len(history_others_sharing) > 0:
            self.ema_others_sharing = (
                (1 - self.alpha) * self.ema_others_sharing
                + self.alpha * history_others_sharing[-1]
            )
        return float(np.clip(self.ema_others_sharing, 0.0, 1.0))

    def type_name(self) -> str:
        return "Reciprocal"


class StrategicAgent(Agent):
    """Shares only when the expected benefit exceeds the expected cost.

    Uses Experience-Weighted Attraction (EWA) learning over 11 discrete
    disclosure levels {0.0, 0.1, ..., 1.0}. Selects actions via softmax
    with temperature that anneals from tau_init to tau_min.
    """

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        n_actions: int = 11,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_decay: float = 0.999,
        learning_rate: float = 0.05,
    ):
        super().__init__(agent_id, rng)
        self.n_actions = n_actions
        self.actions = np.linspace(0.0, 1.0, n_actions)
        self.attractions = np.zeros(n_actions)  # EWA attractions
        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.learning_rate = learning_rate
        self.last_action_idx: int | None = None

    def choose_disclosure(
        self,
        round_num: int,
        history_own_payoffs: list[float],
        history_others_sharing: list[float],
    ) -> float:
        # Update attractions based on last payoff
        if self.last_action_idx is not None and len(history_own_payoffs) > 0:
            payoff = history_own_payoffs[-1]
            self.attractions[self.last_action_idx] += (
                self.learning_rate * (payoff - self.attractions[self.last_action_idx])
            )

        # Softmax selection
        scaled = self.attractions / max(self.tau, self.tau_min)
        scaled -= np.max(scaled)  # numerical stability
        probs = np.exp(scaled)
        probs /= probs.sum()

        idx = self.rng.choice(self.n_actions, p=probs)
        self.last_action_idx = idx

        # Anneal temperature
        self.tau = max(self.tau * self.tau_decay, self.tau_min)

        return float(self.actions[idx])

    def type_name(self) -> str:
        return "Strategic"


# ---- Factory ----

AGENT_TYPES = {
    "open": OpenAgent,
    "secretive": SecretiveAgent,
    "reciprocal": ReciprocalAgent,
    "strategic": StrategicAgent,
}


def create_agents(
    composition: list[str], rng: np.random.Generator
) -> list[Agent]:
    """Create agents from a list of type names.

    Parameters
    ----------
    composition : list[str]
        e.g. ["open", "open", "secretive", "strategic"]
    rng : Generator
        Shared RNG for reproducibility.

    Returns
    -------
    list[Agent]
    """
    agents = []
    for i, atype in enumerate(composition):
        cls = AGENT_TYPES[atype.lower()]
        agents.append(cls(agent_id=i, rng=rng))
    return agents
