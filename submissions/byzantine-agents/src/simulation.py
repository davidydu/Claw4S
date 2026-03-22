"""Simulation engine for Byzantine fault tolerance experiments.

Each simulation consists of *rounds* independent voting rounds.
In each round:
  1. A ground-truth option is drawn uniformly from [0, K).
  2. Each agent receives noisy observations of the true option.
     - MajorityVoter and CautiousVoter: 1 sample each.
     - BayesianVoter: ``bayesian_samples`` samples each (default 3).
     - Byzantine agents: 1 sample each (they may or may not use it).
     Each sample equals the true option with probability ``signal_quality``
     and is uniformly random otherwise.
  3. Agents vote (or abstain).
  4. The committee decision is the plurality vote (ties broken randomly).
  5. We record whether the decision matches ground truth.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.agents import (
    K_OPTIONS,
    make_honest_agent,
    make_byzantine_agent,
    Agent,
)


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a single simulation run."""
    committee_size: int          # N
    honest_type: str             # key into HONEST_TYPES
    byzantine_type: str          # key into BYZANTINE_TYPES
    byzantine_fraction: float    # f in [0, 1]
    rounds: int = 1_000         # voting rounds per sim
    signal_quality: float = 0.6 # P(observation = true option)
    bayesian_samples: int = 3   # number of samples for BayesianVoter
    seed: int = 42


@dataclass
class SimResult:
    """Results of a single simulation."""
    config: SimConfig
    accuracy: float              # fraction of rounds decided correctly
    accuracy_std: float          # binomial standard error
    num_correct: int
    num_rounds: int
    num_honest: int
    num_byzantine: int


def _generate_observations(
    true_option: int,
    n_samples: int,
    signal_quality: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate observation count vector of shape (K,).

    Each of *n_samples* independent observations equals *true_option*
    with probability *signal_quality*, else uniform random in [0, K).
    Returns the count of how many times each option was observed.
    """
    observations = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if rng.random() < signal_quality:
            observations[i] = true_option
        else:
            observations[i] = rng.integers(0, K_OPTIONS)
    counts = np.bincount(observations, minlength=K_OPTIONS).astype(float)
    return counts


def run_simulation(cfg: SimConfig) -> SimResult:
    """Execute a full simulation and return aggregated results."""
    rng = np.random.default_rng(cfg.seed)

    n_byzantine = int(round(cfg.committee_size * cfg.byzantine_fraction))
    n_honest = cfg.committee_size - n_byzantine

    honest_agents: list[Agent] = [make_honest_agent(cfg.honest_type) for _ in range(n_honest)]
    byzantine_agents: list[Agent] = [make_byzantine_agent(cfg.byzantine_type) for _ in range(n_byzantine)]
    all_agents = honest_agents + byzantine_agents

    # Determine number of samples per agent
    n_samples_per_agent = []
    for i, agent in enumerate(all_agents):
        if i < n_honest and cfg.honest_type == "bayesian":
            n_samples_per_agent.append(cfg.bayesian_samples)
        else:
            n_samples_per_agent.append(1)

    correct = 0

    for _ in range(cfg.rounds):
        true_option = int(rng.integers(0, K_OPTIONS))

        # Collect votes
        votes: list[int] = []
        for i, agent in enumerate(all_agents):
            obs = _generate_observations(
                true_option, n_samples_per_agent[i], cfg.signal_quality, rng
            )
            v = agent.vote(obs, rng)
            if v >= 0:  # -1 means abstain
                votes.append(v)

        if not votes:
            # Everyone abstained — treat as incorrect
            continue

        # Plurality vote with random tie-breaking
        counts = np.bincount(votes, minlength=K_OPTIONS)
        max_count = counts.max()
        winners = np.where(counts == max_count)[0]
        decision = int(rng.choice(winners))

        if decision == true_option:
            correct += 1

    accuracy = correct / cfg.rounds
    # Binomial standard error
    accuracy_std = float(np.sqrt(accuracy * (1 - accuracy) / cfg.rounds))

    return SimResult(
        config=cfg,
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        num_correct=correct,
        num_rounds=cfg.rounds,
        num_honest=n_honest,
        num_byzantine=n_byzantine,
    )
