# tests/test_agents.py
import numpy as np
from src.agents import DancerAgent, create_dancers


def test_dancer_agent_fields():
    """DancerAgent should have phase, frequency, and position."""
    agent = DancerAgent(phase=0.5, frequency=1.0, x=3.0, y=4.0)
    assert agent.phase == 0.5
    assert agent.frequency == 1.0
    assert agent.x == 3.0
    assert agent.y == 4.0


def test_create_dancers():
    """create_dancers should return N agents with correct distributions."""
    dancers = create_dancers(n=12, omega0=1.0, sigma=0.5, stage_size=10.0, seed=42)
    assert len(dancers) == 12
    phases = [d.phase for d in dancers]
    assert all(0 <= p < 2 * np.pi for p in phases)
    freqs = [d.frequency for d in dancers]
    assert abs(np.mean(freqs) - 1.0) < 0.5  # roughly centered


def test_create_dancers_reproducible():
    """Same seed -> same dancers."""
    d1 = create_dancers(n=6, omega0=1.0, sigma=0.5, stage_size=10.0, seed=42)
    d2 = create_dancers(n=6, omega0=1.0, sigma=0.5, stage_size=10.0, seed=42)
    for a, b in zip(d1, d2):
        assert a.phase == b.phase
        assert a.frequency == b.frequency
