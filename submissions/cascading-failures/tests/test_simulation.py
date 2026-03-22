"""Tests for the simulation engine."""

from src.network import ring, star, chain
from src.simulation import SimConfig, run_simulation, ERROR_THRESHOLD


def _make_config(adj, **kwargs) -> SimConfig:
    defaults = dict(
        n_agents=len(adj),
        topology_name="test",
        agent_type="fragile",
        adj=adj,
        shock_node=0,
        shock_magnitude=10.0,
        shock_round=10,
        shock_duration=20,
        total_rounds=200,
        seed=42,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def test_no_shock_no_cascade():
    """Without shock, cascade size should be 0."""
    adj = ring(10)
    cfg = _make_config(adj, shock_round=9999, total_rounds=100)
    result = run_simulation(cfg)
    assert result.cascade_size == 0.0


def test_shock_causes_infection():
    """A shock should infect at least the shocked node."""
    adj = ring(10)
    cfg = _make_config(adj, shock_magnitude=50.0)
    result = run_simulation(cfg)
    assert result.cascade_size > 0.0


def test_cascade_size_bounded():
    """Cascade size should be in [0, 1]."""
    adj = star(10)
    cfg = _make_config(adj, shock_node=0, shock_magnitude=50.0)
    result = run_simulation(cfg)
    assert 0.0 <= result.cascade_size <= 1.0


def test_robust_resists_more():
    """Robust agents should have smaller or equal cascade than fragile."""
    adj = ring(10)
    cfg_frag = _make_config(adj, agent_type="fragile", shock_magnitude=10.0)
    cfg_rob = _make_config(adj, agent_type="robust", shock_magnitude=10.0)
    r_frag = run_simulation(cfg_frag)
    r_rob = run_simulation(cfg_rob)
    # Robust should resist at least as well
    assert r_rob.cascade_size <= r_frag.cascade_size + 0.05


def test_deterministic():
    """Same config should produce same results."""
    adj = ring(10)
    cfg = _make_config(adj, seed=42)
    r1 = run_simulation(cfg)
    r2 = run_simulation(cfg)
    assert r1.cascade_size == r2.cascade_size
    assert r1.cascade_speed == r2.cascade_speed


def test_systemic_risk_nonnegative():
    """Systemic risk should never be negative."""
    adj = chain(10)
    cfg = _make_config(adj, shock_magnitude=5.0)
    result = run_simulation(cfg)
    assert result.systemic_risk >= 0.0


def test_star_hub_attack():
    """Attacking the hub of a star should affect more nodes than a leaf."""
    adj = star(10)
    cfg_hub = _make_config(adj, shock_node=0, shock_magnitude=20.0)
    cfg_leaf = _make_config(adj, shock_node=5, shock_magnitude=20.0)
    r_hub = run_simulation(cfg_hub)
    r_leaf = run_simulation(cfg_leaf)
    assert r_hub.cascade_size >= r_leaf.cascade_size
