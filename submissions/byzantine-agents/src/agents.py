"""Agent implementations for Byzantine fault tolerance simulation.

Three honest voter types and three Byzantine strategies.

Signal model: each agent receives a *noisy categorical observation*.
With probability ``signal_quality`` the observation equals the true option;
otherwise it is uniformly random among K options. The signal is delivered
as a length-K vector of *counts* (how many times each option appeared in
the agent's private sample).  MajorityVoter gets 1 sample,
BayesianVoter gets ``n_samples`` samples (default 3), and CautiousVoter
gets 1 sample but abstains when the posterior is too uncertain.
"""

import numpy as np
from typing import Protocol


K_OPTIONS = 5  # number of decision options


class Agent(Protocol):
    """Protocol for all agents."""

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        """Return a vote given a signal count vector of shape (K,)."""
        ...


# ---------------------------------------------------------------------------
# Honest agents
# ---------------------------------------------------------------------------

class MajorityVoter:
    """Votes for the option with the highest count in its single observation.

    With only one sample this is equivalent to voting for the observed option.
    Ties (which don't happen with a single sample) are broken randomly.
    """

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        max_val = signals.max()
        winners = np.where(signals == max_val)[0]
        return int(rng.choice(winners))


class BayesianVoter:
    """Receives multiple observations and computes a posterior.

    Uses a Dirichlet-Categorical conjugate model with a uniform
    Dirichlet(1,...,1) prior.  The vote is the MAP estimate of the
    posterior (argmax of alpha + counts).  With more samples this
    voter is better at identifying the true option than MajorityVoter.
    """

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        # Posterior Dirichlet alpha = prior (1) + counts
        posterior_alpha = 1.0 + signals
        # MAP for Dirichlet-Categorical is argmax of alpha
        max_val = posterior_alpha.max()
        winners = np.where(posterior_alpha == max_val)[0]
        return int(rng.choice(winners))


class CautiousVoter:
    """Votes only when confident; abstains otherwise.

    Computes the posterior mean from a single observation.  Abstains
    (returns -1) if the posterior probability of the best option is
    below ``threshold``.
    """

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        posterior_alpha = 1.0 + signals
        posterior_mean = posterior_alpha / posterior_alpha.sum()
        best = int(np.argmax(posterior_mean))
        if posterior_mean[best] < self.threshold:
            return -1  # abstain
        return best


# ---------------------------------------------------------------------------
# Byzantine agents
# ---------------------------------------------------------------------------

class RandomByzantine:
    """Votes uniformly at random, ignoring all signals."""

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        return int(rng.integers(0, K_OPTIONS))


class StrategicByzantine:
    """Coordinates to vote for a fixed wrong option to concentrate
    adversarial votes.

    Always votes for option 0 regardless of the true answer.  When the
    true answer happens to be 0 this accidentally helps, but for
    K=5 that only happens 20% of the time.
    """

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        return 0  # always vote option 0


class MimickingByzantine:
    """Appears honest most of the time but flips to adversarial with
    probability ``flip_prob`` each round.

    When not flipping, votes like a MajorityVoter (argmax of signal).
    When flipping, votes for a coordinated wrong answer (option 0).
    """

    def __init__(self, flip_prob: float = 0.3):
        self.flip_prob = flip_prob

    def vote(self, signals: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.flip_prob:
            return 0  # coordinated wrong vote
        max_val = signals.max()
        winners = np.where(signals == max_val)[0]
        return int(rng.choice(winners))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

HONEST_TYPES = {
    "majority": MajorityVoter,
    "bayesian": BayesianVoter,
    "cautious": CautiousVoter,
}

BYZANTINE_TYPES = {
    "random": RandomByzantine,
    "strategic": StrategicByzantine,
    "mimicking": MimickingByzantine,
}


def make_honest_agent(name: str) -> Agent:
    """Instantiate an honest agent by name."""
    return HONEST_TYPES[name]()


def make_byzantine_agent(name: str) -> Agent:
    """Instantiate a Byzantine agent by name."""
    return BYZANTINE_TYPES[name]()
