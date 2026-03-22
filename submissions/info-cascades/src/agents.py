"""Agent types for information cascade simulation.

Each agent receives a private signal and observes predecessors' actions,
then chooses action A (0) or B (1).

Agent types:
- Bayesian: optimal Bayes' rule updater
- Heuristic: follows majority if it exceeds a threshold
- Contrarian: goes against the crowd with some probability
- NoisyBayesian: Bayesian with computational noise (logit perturbation)
"""

import math
import random
from typing import Protocol


# Actions: 0 = A, 1 = B
ACTION_A = 0
ACTION_B = 1


class Agent(Protocol):
    """Protocol for cascade agents."""

    def choose(
        self,
        signal: int,
        predecessors: list[int],
        signal_quality: float,
    ) -> int:
        """Choose action A (0) or B (1).

        Args:
            signal: Private signal (0 or 1), correct with prob signal_quality.
            predecessors: List of actions chosen by preceding agents.
            signal_quality: Probability that signal matches the true state.

        Returns:
            0 (action A) or 1 (action B).
        """
        ...


def _log_likelihood_ratio(predecessors: list[int], signal_quality: float) -> float:
    """Compute log-likelihood ratio from observed actions.

    Assumes each predecessor was Bayesian and their actions reveal whether
    their posterior favored A or B. Under this assumption each action that
    equals A (0) contributes log(q/(1-q)) toward state A, and vice versa.

    Returns log(P(state=A | actions) / P(state=B | actions)) assuming
    uniform prior, i.e., sum of individual LLRs.
    """
    q = signal_quality
    llr_per_action = math.log(q / (1.0 - q))
    count_a = predecessors.count(ACTION_A)
    count_b = predecessors.count(ACTION_B)
    return (count_a - count_b) * llr_per_action


class BayesianAgent:
    """Optimal Bayesian agent. Combines public LLR with private signal LLR."""

    def choose(
        self,
        signal: int,
        predecessors: list[int],
        signal_quality: float,
    ) -> int:
        public_llr = _log_likelihood_ratio(predecessors, signal_quality)
        private_llr = math.log(signal_quality / (1.0 - signal_quality))
        if signal == ACTION_B:
            private_llr = -private_llr

        total_llr = public_llr + private_llr  # positive favors A
        if total_llr > 0:
            return ACTION_A
        elif total_llr < 0:
            return ACTION_B
        else:
            # Tie: follow private signal (standard convention)
            return signal


class HeuristicAgent:
    """Follows majority if majority fraction > threshold, else follows signal."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def choose(
        self,
        signal: int,
        predecessors: list[int],
        signal_quality: float,
    ) -> int:
        if not predecessors:
            return signal
        n = len(predecessors)
        count_a = predecessors.count(ACTION_A)
        frac_a = count_a / n
        if frac_a > self.threshold:
            return ACTION_A
        elif (1.0 - frac_a) > self.threshold:
            return ACTION_B
        else:
            return signal


class ContrarianAgent:
    """Goes against the majority with probability p_contrarian."""

    def __init__(self, p_contrarian: float = 0.3, rng: random.Random | None = None):
        self.p_contrarian = p_contrarian
        self.rng = rng or random.Random()

    def choose(
        self,
        signal: int,
        predecessors: list[int],
        signal_quality: float,
    ) -> int:
        if not predecessors:
            return signal
        count_a = predecessors.count(ACTION_A)
        count_b = len(predecessors) - count_a
        if count_a > count_b:
            majority = ACTION_A
        elif count_b > count_a:
            majority = ACTION_B
        else:
            majority = signal

        if self.rng.random() < self.p_contrarian:
            return 1 - majority  # go against
        else:
            return majority


class NoisyBayesianAgent:
    """Bayesian agent with logit noise (bounded rationality)."""

    def __init__(self, noise_std: float = 1.0, rng: random.Random | None = None):
        self.noise_std = noise_std
        self.rng = rng or random.Random()

    def choose(
        self,
        signal: int,
        predecessors: list[int],
        signal_quality: float,
    ) -> int:
        public_llr = _log_likelihood_ratio(predecessors, signal_quality)
        private_llr = math.log(signal_quality / (1.0 - signal_quality))
        if signal == ACTION_B:
            private_llr = -private_llr

        noise = self.rng.gauss(0, self.noise_std)
        total_llr = public_llr + private_llr + noise
        if total_llr > 0:
            return ACTION_A
        elif total_llr < 0:
            return ACTION_B
        else:
            return signal


def make_agent(agent_type: str, rng: random.Random | None = None) -> Agent:
    """Factory function for creating agents by type name.

    Args:
        agent_type: One of "bayesian", "heuristic", "contrarian", "noisy_bayesian".
        rng: Random number generator for stochastic agents.

    Returns:
        An agent instance.
    """
    if agent_type == "bayesian":
        return BayesianAgent()
    elif agent_type == "heuristic":
        return HeuristicAgent(threshold=0.6)
    elif agent_type == "contrarian":
        return ContrarianAgent(p_contrarian=0.3, rng=rng)
    elif agent_type == "noisy_bayesian":
        return NoisyBayesianAgent(noise_std=1.0, rng=rng)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
