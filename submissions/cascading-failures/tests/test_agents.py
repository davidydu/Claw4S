"""Tests for agent processing functions."""

import math
from src.agents import robust_agent, fragile_agent, averaging_agent, CLIP_BOUND, DECAY


def test_robust_clips_extreme():
    # Large positive inputs should be clipped to CLIP_BOUND
    result = robust_agent([100.0, -100.0], noise=0.0)
    # After clipping to [-2, 2], mean = 0, tanh(0) = 0
    assert abs(result) < 0.01


def test_robust_passes_normal():
    result = robust_agent([0.5, 0.5], noise=0.0)
    # DECAY * tanh(0.5) ~ 0.95 * 0.462 ~ 0.439
    assert 0.3 < result < 0.6


def test_fragile_relays_linearly():
    # Fragile agent takes mean and relays with decay, no tanh
    result = fragile_agent([5.0], noise=0.0)
    # DECAY * 5.0 = 4.75
    assert abs(result - DECAY * 5.0) < 0.01


def test_fragile_passes_garbage():
    # Large error passes through linearly (scaled by decay)
    result = fragile_agent([50.0], noise=0.0)
    assert abs(result - DECAY * 50.0) < 0.01


def test_fragile_more_vulnerable_than_robust():
    # For large inputs, fragile should produce larger output magnitude
    inputs = [10.0, 10.0, 10.0]
    r_frag = abs(fragile_agent(inputs, noise=0.0))
    r_rob = abs(robust_agent(inputs, noise=0.0))
    assert r_frag > r_rob


def test_averaging_dampens():
    # One outlier among normal values
    result = averaging_agent([0.0, 0.0, 0.0, 10.0], noise=0.0)
    # Mean = 2.5, DECAY * tanh(2.5) ~ 0.95 * 0.987 ~ 0.937
    assert 0.8 < result < 1.0


def test_averaging_small_inputs():
    result = averaging_agent([0.1, 0.2, 0.3], noise=0.0)
    # Mean = 0.2, DECAY * tanh(0.2) ~ 0.95 * 0.197 ~ 0.187
    assert 0.1 < result < 0.3


def test_empty_inputs():
    """All agent types should handle no neighbors gracefully."""
    assert robust_agent([], noise=0.0) == 0.0
    assert fragile_agent([], noise=0.0) == 0.0
    assert averaging_agent([], noise=0.0) == 0.0


def test_noise_affects_output():
    r1 = fragile_agent([0.5], noise=0.0)
    r2 = fragile_agent([0.5], noise=0.1)
    assert abs(r2 - r1 - 0.1) < 0.01
