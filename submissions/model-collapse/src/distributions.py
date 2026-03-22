"""Ground-truth distributions and KDE-based learned distributions.

Each ground-truth distribution is a mixture of 3 Gaussians in 1D.
Agents learn distributions via kernel density estimation (KDE) on
training data, then generate synthetic samples from the learned KDE.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.stats import wasserstein_distance as _wasserstein

# ---------------------------------------------------------------------------
# Ground-truth mixture-of-Gaussians
# ---------------------------------------------------------------------------

# Three named distributions with distinct shapes
DISTRIBUTIONS = {
    "bimodal": {
        "means": [-3.0, 0.0, 3.0],
        "stds": [0.8, 0.3, 0.8],
        "weights": [0.45, 0.10, 0.45],
    },
    "skewed": {
        "means": [-1.0, 2.0, 6.0],
        "stds": [0.5, 1.0, 0.7],
        "weights": [0.50, 0.30, 0.20],
    },
    "uniform_like": {
        "means": [-2.0, 0.0, 2.0],
        "stds": [1.2, 1.2, 1.2],
        "weights": [0.33, 0.34, 0.33],
    },
}


def sample_ground_truth(dist_name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw *n* samples from the named mixture-of-Gaussians."""
    cfg = DISTRIBUTIONS[dist_name]
    means = np.array(cfg["means"])
    stds = np.array(cfg["stds"])
    weights = np.array(cfg["weights"])
    weights = weights / weights.sum()  # normalise

    components = rng.choice(len(means), size=n, p=weights)
    samples = rng.normal(loc=means[components], scale=stds[components])
    return samples


def ground_truth_pdf(dist_name: str, x: np.ndarray) -> np.ndarray:
    """Evaluate the true PDF at points *x*."""
    cfg = DISTRIBUTIONS[dist_name]
    means = np.array(cfg["means"])
    stds = np.array(cfg["stds"])
    weights = np.array(cfg["weights"])
    weights = weights / weights.sum()

    pdf = np.zeros_like(x, dtype=float)
    for mu, sigma, w in zip(means, stds, weights):
        pdf += w * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return pdf


# ---------------------------------------------------------------------------
# KDE wrapper
# ---------------------------------------------------------------------------

def fit_kde(samples: np.ndarray, bw_method: str = "silverman") -> gaussian_kde:
    """Fit a KDE to 1-D samples."""
    return gaussian_kde(samples, bw_method=bw_method)


def sample_from_kde(kde: gaussian_kde, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw *n* samples from a fitted KDE (seeded)."""
    # scipy KDE resample doesn't accept a Generator, so we seed via legacy
    legacy_seed = int(rng.integers(0, 2**31))
    return kde.resample(n, seed=legacy_seed).flatten()


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def kl_divergence_numerical(
    dist_name: str,
    kde: gaussian_kde,
    x_min: float = -15.0,
    x_max: float = 15.0,
) -> float:
    """KL(true || learned) via numerical integration.

    Returns KL divergence in nats.  Clamps the learned PDF below at 1e-12
    to avoid log(0).
    """
    def integrand(x):
        p = float(ground_truth_pdf(dist_name, np.array([x]))[0])
        q = max(float(kde(np.array([x]))[0]), 1e-12)
        if p < 1e-15:
            return 0.0
        return p * np.log(p / q)

    val, _ = quad(integrand, x_min, x_max, limit=200)
    return max(val, 0.0)  # numerical noise can make it slightly negative


def wasserstein(true_samples: np.ndarray, learned_samples: np.ndarray) -> float:
    """Earth-mover distance between two sets of 1-D samples."""
    return float(_wasserstein(true_samples, learned_samples))
