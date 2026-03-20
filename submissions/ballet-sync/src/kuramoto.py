"""Kuramoto synchronization model with RK4 integration and configurable topologies."""

import numpy as np
from src.agents import create_dancers


class KuramotoModel:
    """Spatially-embedded Kuramoto oscillator model."""

    def __init__(self, n, K, sigma, omega0=1.0, topology="all-to-all",
                 dt=0.01, stage_size=10.0, seed=0, topology_kwargs=None):
        self.n = n
        self.K = K
        self.sigma = sigma
        self.omega0 = omega0
        self.dt = dt
        self.seed = seed

        dancers = create_dancers(n, omega0, sigma, stage_size, seed)
        self.phases = np.array([d.phase for d in dancers])
        self.frequencies = np.array([d.frequency for d in dancers])
        self.positions = np.array([[d.x, d.y] for d in dancers])

        kwargs = topology_kwargs or {}
        self.adjacency = build_topology(topology, n, self.positions, **kwargs)

    @classmethod
    def from_preset(cls, preset_name, K, seed=0):
        """Create model from a domain preset."""
        config = DOMAIN_PRESETS[preset_name]
        return cls(K=K, seed=seed, **config)

    def compute_order_parameter(self, phases):
        """Kuramoto order parameter r = (1/N)|Σ exp(iθ)|."""
        return float(np.abs(np.mean(np.exp(1j * phases))))

    def _coupling(self, phases):
        """Compute coupling term for each oscillator."""
        coupling = np.zeros(self.n)
        for i in range(self.n):
            neighbors = self.adjacency[i]
            if len(neighbors) > 0:
                diffs = np.sin(phases[neighbors] - phases[i])
                coupling[i] = self.K * diffs.mean()
        return coupling

    def _deriv(self, phases):
        """dθ/dt = ω + coupling."""
        return self.frequencies + self._coupling(phases)

    def step(self):
        """One RK4 integration step."""
        dt = self.dt
        k1 = dt * self._deriv(self.phases)
        k2 = dt * self._deriv(self.phases + k1 / 2)
        k3 = dt * self._deriv(self.phases + k2 / 2)
        k4 = dt * self._deriv(self.phases + k3)
        self.phases += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # Wrap to [0, 2π)
        self.phases = self.phases % (2 * np.pi)


def build_topology(name, n, positions=None, **kwargs):
    """Build adjacency list for a given topology.

    Returns: dict mapping node_id -> list of neighbor_ids
    """
    builder = TOPOLOGIES[name]
    return builder(n, positions, **kwargs)


def _all_to_all(n, positions, **kwargs):
    adj = {}
    for i in range(n):
        adj[i] = [j for j in range(n) if j != i]
    return adj


def _nearest_k(n, positions, k=4, **kwargs):
    if positions is None:
        positions = np.column_stack([np.arange(n), np.zeros(n)])
    dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    adj = {}
    for i in range(n):
        sorted_idx = np.argsort(dists[i])
        adj[i] = [int(j) for j in sorted_idx[1:k + 1]]  # skip self
    return adj


def _hierarchical(n, positions, **kwargs):
    """Principal (0) -> soloists (1,2) -> corps (3..n-1)."""
    adj = {i: [] for i in range(n)}
    # Principal connects to soloists
    n_soloists = min(2, n - 1)
    for s in range(1, n_soloists + 1):
        adj[0].append(s)
        adj[s].append(0)

    # Distribute corps among soloists
    corps = list(range(n_soloists + 1, n))
    for idx, c in enumerate(corps):
        soloist = 1 + (idx % n_soloists)
        adj[soloist].append(c)
        adj[c].append(soloist)

    return adj


def _ring(n, positions, **kwargs):
    adj = {}
    for i in range(n):
        adj[i] = [(i - 1) % n, (i + 1) % n]
    return adj


TOPOLOGIES = {
    "all-to-all": _all_to_all,
    "nearest-k": _nearest_k,
    "hierarchical": _hierarchical,
    "ring": _ring,
}


DOMAIN_PRESETS = {
    "ballet-corps": {
        "n": 12, "sigma": 0.5, "topology": "hierarchical",
    },
    "fireflies": {
        "n": 50, "sigma": 1.0, "topology": "nearest-k",
        "topology_kwargs": {"k": 6},
    },
    "drum-circle": {
        "n": 8, "sigma": 0.3, "topology": "all-to-all",
    },
    "power-grid": {
        "n": 10, "sigma": 0.2, "topology": "ring",
    },
}


def analytical_kc(sigma):
    """Analytical critical coupling for all-to-all Kuramoto with Gaussian frequencies."""
    return 2 * sigma * np.sqrt(2 * np.pi) / np.pi
