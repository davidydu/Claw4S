# tests/test_experiment.py
import numpy as np
from src.experiment import ExperimentConfig, run_simulation, K_RANGE


def test_config_defaults():
    config = ExperimentConfig(K=1.0, topology="all-to-all", n=12, sigma=0.5, seed=0)
    assert config.total_steps == 10_000
    assert config.dt == 0.01


def test_run_simulation_small():
    config = ExperimentConfig(K=1.0, topology="all-to-all", n=6, sigma=0.5,
                              seed=42, total_steps=500)
    result = run_simulation(config)
    assert result.phase_history.shape == (500, 6)
    assert 0.0 <= result.final_r <= 1.0


def test_run_simulation_reproducible():
    config = ExperimentConfig(K=1.0, topology="all-to-all", n=6, sigma=0.5,
                              seed=42, total_steps=200)
    r1 = run_simulation(config)
    r2 = run_simulation(config)
    np.testing.assert_array_equal(r1.phase_history, r2.phase_history)


def test_run_simulation_zero_coupling():
    """K=0 should produce low sync."""
    config = ExperimentConfig(K=0.0, topology="all-to-all", n=12, sigma=0.5,
                              seed=42, total_steps=1000)
    result = run_simulation(config)
    assert result.final_r < 0.4


def test_run_simulation_strong_coupling():
    """Large K should produce high sync."""
    config = ExperimentConfig(K=3.0, topology="all-to-all", n=12, sigma=0.3,
                              seed=42, total_steps=5000)
    result = run_simulation(config)
    assert result.final_r > 0.6


def test_k_range():
    """K sweep should have 20 values from 0 to 2.85."""
    assert len(K_RANGE) == 20
    assert K_RANGE[0] == 0.0
    assert abs(K_RANGE[-1] - 2.85) < 0.01
