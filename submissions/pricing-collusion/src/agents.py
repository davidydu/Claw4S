"""Pricing agent implementations for Bertrand competition."""

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all pricing agents."""

    def __init__(self, agent_id, market):
        self.agent_id = agent_id
        self.market = market
        self.n_actions = market.price_grid_size
        self._learning = True

    @abstractmethod
    def choose_action(self, price_history):
        """Choose a price grid index given the history of price indices.

        Args:
            price_history: ndarray of shape (rounds_so_far, n_sellers),
                           each entry is a price grid index.
        Returns:
            action: int, index into market.price_grid
        """

    def update(self, state, action, reward, next_state):
        """Update internal state after observing outcome. Default: no-op."""

    def save_state(self):
        """Return a copy of internal state for counterfactual replay."""
        return None

    def load_state(self, state):
        """Load a previously saved internal state."""

    def set_learning(self, enabled):
        """Enable or disable learning (for counterfactual replay)."""
        self._learning = enabled


class CompetitiveAgent(BaseAgent):
    """Always prices at the Nash equilibrium."""

    def __init__(self, agent_id, market):
        super().__init__(agent_id, market)
        nash = market.nash_price()
        self._nash_action = int(np.argmin(np.abs(market.price_grid - nash)))

    def choose_action(self, price_history):
        return self._nash_action


class TitForTatAgent(BaseAgent):
    """Matches the opponent's last price. Plays mid-range on first round."""

    def __init__(self, agent_id, market):
        super().__init__(agent_id, market)
        self._default_action = market.price_grid_size // 2

    def choose_action(self, price_history):
        if len(price_history) == 0:
            return self._default_action
        # Get the other agent's last action
        opponent_id = 1 - self.agent_id  # works for N=2
        return int(price_history[-1, opponent_id])


class TileCoding:
    """Tile coding for function approximation with large state spaces.

    Uses multiple tilings with hashing to a fixed-size weight table.
    """

    def __init__(self, n_dims, n_tilings=8, tiles_per_dim=16, table_size=65536,
                 rng=None):
        self.n_dims = n_dims
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.table_size = table_size
        rng = rng or np.random.default_rng(0)
        # Random offsets for each tiling
        self.offsets = rng.uniform(0, 1.0 / tiles_per_dim,
                                   size=(n_tilings, n_dims))

    def get_tiles(self, state):
        """Return tile indices for a normalized state vector in [0, 1]^n_dims."""
        state = np.asarray(state, dtype=np.float64)
        tiles = []
        for t in range(self.n_tilings):
            shifted = state + self.offsets[t]
            coords = np.floor(shifted * self.tiles_per_dim).astype(int)
            # Hash: combine tiling index and coordinates
            h = hash((t,) + tuple(coords)) % self.table_size
            tiles.append(h)
        return tiles


class QLearningAgent(BaseAgent):
    """Tabular Q-learning with epsilon-greedy exploration.

    Uses tile coding for memory M >= 3.
    """

    def __init__(self, agent_id, market, memory=1, discount=0.95,
                 eta_start=0.15, eta_end=0.01, eps_start=1.0, eps_end=0.01,
                 total_rounds=500_000, seed=0):
        super().__init__(agent_id, market)
        self.memory = memory
        self.discount = discount
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.total_rounds = total_rounds
        self.rng = np.random.default_rng(seed)
        self._learning = True
        self._round = 0

        n_sellers = market.n_sellers
        self.use_tiles = memory >= 3
        if self.use_tiles:
            n_dims = n_sellers * memory
            self.tile_coder = TileCoding(n_dims=n_dims, rng=self.rng)
            self.weights = np.zeros((self.tile_coder.table_size, self.n_actions))
        else:
            # Tabular: state = tuple of last M price indices for all sellers
            self.q_table = {}

    def _get_state_key(self, price_history):
        """Convert price history to state representation."""
        if len(price_history) < self.memory:
            # Pad with zeros if not enough history
            padded = np.zeros((self.memory, self.market.n_sellers), dtype=int)
            if len(price_history) > 0:
                padded[-len(price_history):] = price_history[-self.memory:]
            price_history = padded
        else:
            price_history = price_history[-self.memory:]

        if self.use_tiles:
            # Normalize to [0, 1]
            normalized = price_history.flatten().astype(float) / (self.n_actions - 1)
            return self.tile_coder.get_tiles(normalized)
        else:
            return tuple(price_history.flatten())

    def _get_q_values(self, state_key):
        if self.use_tiles:
            return self.weights[state_key].sum(axis=0)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)
            return self.q_table[state_key]

    def _epsilon(self):
        decay_end = int(self.total_rounds * 0.4)
        if self._round >= decay_end:
            return self.eps_end
        t = self._round / decay_end
        return self.eps_start + (self.eps_end - self.eps_start) * t

    def _eta(self):
        decay_end = int(self.total_rounds * 0.4)
        if self._round >= decay_end:
            return self.eta_end
        t = self._round / decay_end
        return self.eta_start + (self.eta_end - self.eta_start) * t

    def choose_action(self, price_history):
        state_key = self._get_state_key(price_history)
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))
        q_values = self._get_q_values(state_key)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        if not self._learning:
            return
        self._round += 1
        state_key = self._get_state_key(state)
        next_key = self._get_state_key(next_state)
        q_current = self._get_q_values(state_key)
        q_next = self._get_q_values(next_key)
        target = reward + self.discount * np.max(q_next)
        eta = self._eta()

        if self.use_tiles:
            error = target - q_current[action]
            for tile in state_key:
                self.weights[tile, action] += eta * error / len(state_key)
        else:
            q_current[action] += eta * (target - q_current[action])

    def save_state(self):
        if self.use_tiles:
            return {"weights": self.weights.copy(), "round": self._round}
        else:
            return {"q_table": {k: v.copy() for k, v in self.q_table.items()},
                    "round": self._round}

    def load_state(self, saved):
        if self.use_tiles:
            self.weights = saved["weights"].copy()
        else:
            self.q_table = {k: v.copy() for k, v in saved["q_table"].items()}
        self._round = saved["round"]


class SARSAAgent(BaseAgent):
    """On-policy SARSA with epsilon-greedy. Same structure as Q-learning
    but updates toward the action actually taken next (on-policy)."""

    def __init__(self, agent_id, market, memory=1, discount=0.95,
                 eta_start=0.15, eta_end=0.01, eps_start=1.0, eps_end=0.01,
                 total_rounds=500_000, seed=0):
        super().__init__(agent_id, market)
        self.memory = memory
        self.discount = discount
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.total_rounds = total_rounds
        self.rng = np.random.default_rng(seed)
        self._learning = True
        self._round = 0
        self._next_action = None

        n_sellers = market.n_sellers
        self.use_tiles = memory >= 3
        if self.use_tiles:
            n_dims = n_sellers * memory
            self.tile_coder = TileCoding(n_dims=n_dims, rng=self.rng)
            self.weights = np.zeros((self.tile_coder.table_size, self.n_actions))
        else:
            self.q_table = {}

    def _get_state_key(self, price_history):
        if len(price_history) < self.memory:
            padded = np.zeros((self.memory, self.market.n_sellers), dtype=int)
            if len(price_history) > 0:
                padded[-len(price_history):] = price_history[-self.memory:]
            price_history = padded
        else:
            price_history = price_history[-self.memory:]
        if self.use_tiles:
            normalized = price_history.flatten().astype(float) / (self.n_actions - 1)
            return self.tile_coder.get_tiles(normalized)
        else:
            return tuple(price_history.flatten())

    def _get_q_values(self, state_key):
        if self.use_tiles:
            return self.weights[state_key].sum(axis=0)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)
            return self.q_table[state_key]

    def _epsilon(self):
        decay_end = int(self.total_rounds * 0.4)
        if self._round >= decay_end:
            return self.eps_end
        t = self._round / decay_end
        return self.eps_start + (self.eps_end - self.eps_start) * t

    def _eta(self):
        decay_end = int(self.total_rounds * 0.4)
        if self._round >= decay_end:
            return self.eta_end
        t = self._round / decay_end
        return self.eta_start + (self.eta_end - self.eta_start) * t

    def choose_action(self, price_history):
        if self._next_action is not None:
            return self._next_action
        state_key = self._get_state_key(price_history)
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))
        q_values = self._get_q_values(state_key)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        if not self._learning:
            return
        self._round += 1
        state_key = self._get_state_key(state)
        next_key = self._get_state_key(next_state)
        q_current = self._get_q_values(state_key)
        # SARSA: choose next action now (on-policy)
        self._next_action = self.choose_action(next_state)
        q_next = self._get_q_values(next_key)
        target = reward + self.discount * q_next[self._next_action]
        eta = self._eta()

        if self.use_tiles:
            error = target - q_current[action]
            for tile in state_key:
                self.weights[tile, action] += eta * error / len(state_key)
        else:
            q_current[action] += eta * (target - q_current[action])

    def save_state(self):
        if self.use_tiles:
            return {"weights": self.weights.copy(), "round": self._round}
        else:
            return {"q_table": {k: v.copy() for k, v in self.q_table.items()},
                    "round": self._round}

    def load_state(self, saved):
        if self.use_tiles:
            self.weights = saved["weights"].copy()
        else:
            self.q_table = {k: v.copy() for k, v in saved["q_table"].items()}
        self._round = saved["round"]
        self._next_action = None  # reset cached action on state load


class PolicyGradientAgent(BaseAgent):
    """REINFORCE with softmax policy over discrete price actions.

    Uses per-round profit as reward with running average baseline.
    """

    def __init__(self, agent_id, market, memory=1,
                 eta_start=0.15, eta_end=0.01, eps_start=1.0, eps_end=0.01,
                 total_rounds=500_000, seed=0):
        super().__init__(agent_id, market)
        self.memory = memory
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.total_rounds = total_rounds
        self.rng = np.random.default_rng(seed)
        self._learning = True
        self._round = 0
        self._avg_reward = 0.0

        n_sellers = market.n_sellers
        self.use_tiles = memory >= 3
        if self.use_tiles:
            n_dims = n_sellers * memory
            self.tile_coder = TileCoding(n_dims=n_dims, rng=self.rng)
            self.weights = np.zeros((self.tile_coder.table_size, self.n_actions))
        else:
            self.theta = {}  # state -> preference vector

    def _get_state_key(self, price_history):
        if len(price_history) < self.memory:
            padded = np.zeros((self.memory, self.market.n_sellers), dtype=int)
            if len(price_history) > 0:
                padded[-len(price_history):] = price_history[-self.memory:]
            price_history = padded
        else:
            price_history = price_history[-self.memory:]
        if self.use_tiles:
            normalized = price_history.flatten().astype(float) / (self.n_actions - 1)
            return self.tile_coder.get_tiles(normalized)
        else:
            return tuple(price_history.flatten())

    def _get_preferences(self, state_key):
        if self.use_tiles:
            return self.weights[state_key].sum(axis=0)
        else:
            if state_key not in self.theta:
                self.theta[state_key] = np.zeros(self.n_actions)
            return self.theta[state_key]

    def _softmax(self, preferences):
        p = preferences - preferences.max()
        exp_p = np.exp(p)
        return exp_p / exp_p.sum()

    def _eta(self):
        decay_end = int(self.total_rounds * 0.4)
        if self._round >= decay_end:
            return self.eta_end
        t = self._round / decay_end
        return self.eta_start + (self.eta_end - self.eta_start) * t

    def choose_action(self, price_history):
        state_key = self._get_state_key(price_history)
        prefs = self._get_preferences(state_key)
        probs = self._softmax(prefs)
        return int(self.rng.choice(self.n_actions, p=probs))

    def update(self, state, action, reward, next_state):
        if not self._learning:
            return
        self._round += 1
        # Update running average baseline
        self._avg_reward += 0.01 * (reward - self._avg_reward)
        advantage = reward - self._avg_reward

        state_key = self._get_state_key(state)
        prefs = self._get_preferences(state_key)
        probs = self._softmax(prefs)
        eta = self._eta()

        # Policy gradient: increase preference for chosen action proportional to advantage
        grad = -probs.copy()
        grad[action] += 1.0  # (1 - pi(a|s)) for chosen, -pi(a|s) for others

        if self.use_tiles:
            for tile in state_key:
                self.weights[tile] += eta * advantage * grad / len(state_key)
        else:
            prefs += eta * advantage * grad

    def save_state(self):
        if self.use_tiles:
            return {"weights": self.weights.copy(), "round": self._round,
                    "avg_reward": self._avg_reward}
        else:
            return {"theta": {k: v.copy() for k, v in self.theta.items()},
                    "round": self._round, "avg_reward": self._avg_reward}

    def load_state(self, saved):
        if self.use_tiles:
            self.weights = saved["weights"].copy()
        else:
            self.theta = {k: v.copy() for k, v in saved["theta"].items()}
        self._round = saved["round"]
        self._avg_reward = saved["avg_reward"]


# Agent name -> type mapping
AGENT_TYPES = {
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "policy_gradient": PolicyGradientAgent,
    "tit_for_tat": TitForTatAgent,
    "competitive": CompetitiveAgent,
}


def create_agent(agent_type, agent_id, market, memory=1, seed=0,
                 total_rounds=500_000):
    """Factory function to create agents by name."""
    cls = AGENT_TYPES[agent_type]
    if cls in (CompetitiveAgent, TitForTatAgent):
        return cls(agent_id=agent_id, market=market)
    return cls(agent_id=agent_id, market=market, memory=memory, seed=seed,
               total_rounds=total_rounds)
