# tests/test_experiment.py
import numpy as np
from src.experiment import ExperimentConfig, run_simulation, MATCHUPS


def test_experiment_config_defaults():
    """ExperimentConfig should have sensible defaults."""
    config = ExperimentConfig(matchup="QQ", memory=1, preset="e-commerce",
                              shocks=False, seed=0)
    assert config.total_rounds == 500_000


def test_run_simulation_small():
    """Run a small simulation and check output structure."""
    config = ExperimentConfig(matchup="QQ", memory=1, preset="e-commerce",
                              shocks=False, seed=42, total_rounds=1000)
    result = run_simulation(config)
    assert result.price_history.shape == (1000, 2)
    assert result.profit_history.shape == (1000, 2)
    assert 0 < result.final_avg_price < 3.0
    assert result.nash_price > 0
    assert result.monopoly_price > result.nash_price


def test_run_simulation_reproducible():
    """Same seed should produce identical results."""
    config = ExperimentConfig(matchup="QQ", memory=1, preset="e-commerce",
                              shocks=False, seed=42, total_rounds=500)
    r1 = run_simulation(config)
    r2 = run_simulation(config)
    np.testing.assert_array_equal(r1.price_history, r2.price_history)


def test_run_simulation_with_shocks():
    """Simulation with shocks should run without error."""
    config = ExperimentConfig(matchup="QQ", memory=1, preset="e-commerce",
                              shocks=True, seed=42, total_rounds=1000)
    result = run_simulation(config)
    assert result.price_history.shape == (1000, 2)


def test_all_matchups_valid():
    """All 6 matchup codes should be defined."""
    expected = {"QQ", "SS", "PG-PG", "QS", "Q-TFT", "Q-Competitive"}
    assert set(MATCHUPS.keys()) == expected


def test_run_simulation_saves_agent_states():
    """Simulation should save agent states at T*0.9 for counterfactual."""
    config = ExperimentConfig(matchup="QQ", memory=1, preset="e-commerce",
                              shocks=False, seed=42, total_rounds=1000)
    result = run_simulation(config)
    assert result.saved_states is not None
    assert len(result.saved_states) == 2
