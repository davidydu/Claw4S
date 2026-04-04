"""Tests for the information-sharing environment."""

import numpy as np
import pytest

from src.environment import EnvConfig, Environment


def test_env_config_defaults():
    c = EnvConfig()
    assert c.n_agents == 4
    assert c.state_dim == 8
    assert 0 < c.obs_fraction <= 1.0


def test_env_obs_masks_cover_all_dims():
    """Each dimension should be observed by at least one agent."""
    rng = np.random.default_rng(42)
    config = EnvConfig(n_agents=4, state_dim=8, obs_fraction=0.25)
    env = Environment(config, rng)
    combined = np.zeros(config.state_dim, dtype=bool)
    for mask in env.obs_masks:
        combined |= mask
    assert combined.all(), "Not all dimensions covered by agents"


def test_env_step_shapes():
    """Environment step returns correct shapes."""
    rng = np.random.default_rng(42)
    config = EnvConfig(n_agents=4)
    env = Environment(config, rng)
    disclosures = np.array([0.5, 0.5, 0.5, 0.5])
    payoffs, sharing, errors, welfare = env.step(disclosures)
    assert payoffs.shape == (4,)
    assert sharing.shape == (4,)
    assert errors.shape == (4,)
    assert isinstance(welfare, float)


def test_full_sharing_improves_decisions():
    """Full sharing should generally reduce errors vs no sharing."""
    rng = np.random.default_rng(42)
    config = EnvConfig(n_agents=4, competition=0.0, complementarity=0.9)
    env = Environment(config, rng)

    errors_sharing = []
    errors_hoarding = []
    for _ in range(200):
        _, _, errs_s, _ = env.step(np.ones(4))
        _, _, errs_h, _ = env.step(np.zeros(4))
        errors_sharing.append(np.mean(errs_s))
        errors_hoarding.append(np.mean(errs_h))

    assert np.mean(errors_sharing) < np.mean(errors_hoarding), (
        "Full sharing should reduce errors vs no sharing"
    )


def test_deterministic_with_same_seed():
    """Same seed should produce identical results."""
    config = EnvConfig(n_agents=4)
    for _ in range(3):
        rng = np.random.default_rng(99)
        env = Environment(config, rng)
        p1, _, _, _ = env.step(np.array([0.5, 0.5, 0.5, 0.5]))

    rng2 = np.random.default_rng(99)
    env2 = Environment(config, rng2)
    p2, _, _, _ = env2.step(np.array([0.5, 0.5, 0.5, 0.5]))
    np.testing.assert_array_equal(p1, p2)
