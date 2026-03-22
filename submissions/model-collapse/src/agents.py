"""Agent types that learn and produce data across generations.

Three agent types model different strategies for handling synthetic data:
  - Naive: learns from all training data equally
  - Selective: filters out low-confidence samples before learning
  - Anchored: mixes training data with a fraction of ground-truth samples

Ground-truth fraction is applied at the data-pipeline level in the
simulation loop, so Naive and Selective agents also benefit from it.
The Anchored agent additionally uses gt_fraction in its learn() method
to mix in ground truth on top of whatever the pipeline provides.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from .distributions import (
    fit_kde,
    sample_from_kde,
    sample_ground_truth,
)

# Number of samples each agent produces per generation
SAMPLES_PER_GENERATION = 2000

# Number of ground-truth reference samples for metrics (fixed across runs)
REFERENCE_SAMPLES = 5000


class BaseAgent:
    """Base class for generational agents."""

    def __init__(self, dist_name: str, gt_fraction: float, rng: np.random.Generator):
        self.dist_name = dist_name
        self.gt_fraction = gt_fraction  # fraction of ground-truth to mix in
        self.rng = rng
        self.kde: gaussian_kde | None = None

    def learn(self, training_data: np.ndarray) -> None:
        """Fit internal model to training data."""
        raise NotImplementedError

    def produce(self, n: int) -> np.ndarray:
        """Generate *n* synthetic samples from the learned model."""
        if self.kde is None:
            raise RuntimeError("Agent has not learned yet")
        return sample_from_kde(self.kde, n, self.rng)


class NaiveAgent(BaseAgent):
    """Learns from all available training data without filtering."""

    agent_type = "naive"

    def learn(self, training_data: np.ndarray) -> None:
        self.kde = fit_kde(training_data)


class SelectiveAgent(BaseAgent):
    """Filters out low-confidence samples before learning.

    Drops samples whose KDE density (under a preliminary fit) falls
    below the 10th percentile, then re-fits on the retained samples.
    """

    agent_type = "selective"
    percentile_cutoff = 10  # drop bottom 10% by density

    def learn(self, training_data: np.ndarray) -> None:
        # Preliminary fit on all data
        preliminary = fit_kde(training_data)
        densities = preliminary(training_data)
        threshold = np.percentile(densities, self.percentile_cutoff)
        filtered = training_data[densities >= threshold]
        # Re-fit on filtered data (at least 50 samples to avoid degenerate KDE)
        if len(filtered) < 50:
            filtered = training_data
        self.kde = fit_kde(filtered)


class AnchoredAgent(BaseAgent):
    """Mixes training data with fresh ground-truth samples in learn().

    On top of whatever the data pipeline provides, the anchored agent
    adds gt_fraction * len(training_data) ground-truth samples before
    fitting.  This means it gets ground truth both from the pipeline
    (shared with all agents) and from its own internal anchoring.
    """

    agent_type = "anchored"

    def learn(self, training_data: np.ndarray) -> None:
        n_gt = max(1, int(self.gt_fraction * len(training_data)))
        gt_samples = sample_ground_truth(self.dist_name, n_gt, self.rng)
        combined = np.concatenate([training_data, gt_samples])
        self.kde = fit_kde(combined)


AGENT_CLASSES = {
    "naive": NaiveAgent,
    "selective": SelectiveAgent,
    "anchored": AnchoredAgent,
}
