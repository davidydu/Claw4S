# tests/test_kuramoto.py
import numpy as np
from src.kuramoto import (
    KuramotoModel, build_topology, TOPOLOGIES, DOMAIN_PRESETS,
    analytical_kc,
)


def test_order_parameter_synchronized():
    """All same phase -> r = 1."""
    phases = np.zeros(12)
    model = KuramotoModel(n=12, K=1.0, sigma=0.5, topology="all-to-all")
    r = model.compute_order_parameter(phases)
    assert abs(r - 1.0) < 1e-10


def test_order_parameter_desynchronized():
    """Evenly spaced phases -> r ≈ 0."""
    phases = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    model = KuramotoModel(n=12, K=1.0, sigma=0.5, topology="all-to-all")
    r = model.compute_order_parameter(phases)
    assert r < 0.1


def test_rk4_step_no_coupling():
    """With K=0, phases advance by omega*dt."""
    model = KuramotoModel(n=3, K=0.0, sigma=0.0, omega0=1.0, topology="all-to-all", seed=42)
    phases_before = model.phases.copy()
    model.step()
    expected = phases_before + model.dt * model.frequencies
    np.testing.assert_allclose(model.phases, expected, atol=1e-10)


def test_rk4_step_with_coupling():
    """With K>0, phases should change differently than K=0."""
    model = KuramotoModel(n=6, K=2.0, sigma=0.5, topology="all-to-all", seed=42)
    phases_before = model.phases.copy()
    model.step()
    free_advance = phases_before + model.dt * model.frequencies
    # Coupled phases should differ from free advance
    assert not np.allclose(model.phases, free_advance)


def test_simulation_increases_sync():
    """Strong coupling should increase order parameter over time."""
    model = KuramotoModel(n=12, K=3.0, sigma=0.3, topology="all-to-all", seed=42)
    r_initial = model.compute_order_parameter(model.phases)
    for _ in range(5000):
        model.step()
    r_final = model.compute_order_parameter(model.phases)
    assert r_final > r_initial


def test_topologies_exist():
    """All 4 topologies should be defined."""
    for name in ["all-to-all", "nearest-k", "hierarchical", "ring"]:
        assert name in TOPOLOGIES


def test_build_topology_all_to_all():
    """All-to-all: every node connected to every other."""
    adj = build_topology("all-to-all", n=4, positions=None)
    # Each node should have 3 neighbors (all except self)
    for i in range(4):
        assert len(adj[i]) == 3
        assert i not in adj[i]


def test_build_topology_ring():
    """Ring: each node connected to 2 neighbors."""
    adj = build_topology("ring", n=6, positions=None)
    for i in range(6):
        assert len(adj[i]) == 2


def test_build_topology_hierarchical():
    """Hierarchical: principal -> soloists -> corps."""
    adj = build_topology("hierarchical", n=12, positions=None)
    # Principal (node 0) should be connected to soloists (nodes 1, 2)
    assert 1 in adj[0] and 2 in adj[0]


def test_domain_presets():
    """All 4 domain presets should exist."""
    assert "ballet-corps" in DOMAIN_PRESETS
    assert "fireflies" in DOMAIN_PRESETS
    assert "drum-circle" in DOMAIN_PRESETS
    assert "power-grid" in DOMAIN_PRESETS


def test_from_preset():
    """Should create a model from a preset name."""
    model = KuramotoModel.from_preset("ballet-corps", K=1.0, seed=0)
    assert model.n == 12
    assert model.sigma == 0.5


def test_analytical_kc():
    """K_c ≈ 1.596 * sigma for all-to-all Gaussian."""
    kc = analytical_kc(sigma=0.5)
    assert abs(kc - 0.798) < 0.01


def test_reproducibility():
    """Same seed should produce identical trajectories."""
    m1 = KuramotoModel(n=6, K=1.0, sigma=0.5, topology="all-to-all", seed=42)
    m2 = KuramotoModel(n=6, K=1.0, sigma=0.5, topology="all-to-all", seed=42)
    for _ in range(100):
        m1.step()
        m2.step()
    np.testing.assert_array_equal(m1.phases, m2.phases)
