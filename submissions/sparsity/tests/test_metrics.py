"""Tests for src/metrics.py."""

import torch
from src.metrics import (
    dead_neuron_fraction,
    near_dead_fraction,
    zero_fraction,
    activation_entropy,
    mean_activation_magnitude,
    compute_all_metrics,
)


def test_dead_fraction_all_alive():
    """No dead neurons when all activations are positive."""
    acts = torch.ones(10, 5)
    frac = dead_neuron_fraction(acts)
    assert frac == 0.0


def test_dead_fraction_all_dead():
    """All neurons dead when all activations are zero."""
    acts = torch.zeros(10, 5)
    frac = dead_neuron_fraction(acts)
    assert frac == 1.0


def test_dead_fraction_partial():
    """Correct fraction when some neurons are dead."""
    acts = torch.zeros(10, 4)
    acts[:, 0] = 1.0  # Only neuron 0 is alive
    acts[:, 2] = 0.5  # Neuron 2 is also alive
    frac = dead_neuron_fraction(acts)
    assert frac == 0.5  # 2 out of 4 are dead


def test_dead_fraction_single_sample_alive():
    """Neuron alive if even one sample has nonzero activation."""
    acts = torch.zeros(10, 3)
    acts[5, 1] = 0.001  # One sample, one neuron
    frac = dead_neuron_fraction(acts)
    # Neurons 0 and 2 are dead, neuron 1 is alive
    assert abs(frac - 2.0 / 3.0) < 1e-6


def test_near_dead_fraction_all_large():
    """No near-dead neurons when all have high mean activation."""
    acts = torch.ones(10, 5) * 10.0
    frac = near_dead_fraction(acts, threshold=1e-3)
    assert frac == 0.0


def test_near_dead_fraction_all_tiny():
    """All near-dead when mean activation is very small."""
    acts = torch.ones(10, 5) * 1e-5
    frac = near_dead_fraction(acts, threshold=1e-3)
    assert frac == 1.0


def test_zero_fraction_all_zero():
    """Zero fraction is 1.0 when all values are zero."""
    acts = torch.zeros(10, 5)
    frac = zero_fraction(acts)
    assert frac == 1.0


def test_zero_fraction_none_zero():
    """Zero fraction is 0.0 when no values are zero."""
    acts = torch.ones(10, 5)
    frac = zero_fraction(acts)
    assert frac == 0.0


def test_zero_fraction_half():
    """Zero fraction correct when half values are zero."""
    acts = torch.zeros(10, 4)
    acts[:, :2] = 1.0  # Half the neurons are active
    frac = zero_fraction(acts)
    assert abs(frac - 0.5) < 1e-6


def test_entropy_uniform():
    """Entropy is positive for spread-out activations."""
    torch.manual_seed(42)
    acts = torch.rand(100, 50)
    ent = activation_entropy(acts)
    assert ent > 0.0


def test_entropy_constant():
    """Entropy is zero for constant activations."""
    acts = torch.full((10, 5), 3.0)
    ent = activation_entropy(acts)
    assert ent == 0.0


def test_mean_magnitude_positive():
    """Mean magnitude is positive for nonzero activations."""
    acts = torch.ones(10, 5) * 2.0
    mag = mean_activation_magnitude(acts)
    assert abs(mag - 2.0) < 1e-6


def test_mean_magnitude_zero():
    """Mean magnitude is zero for zero activations."""
    acts = torch.zeros(10, 5)
    mag = mean_activation_magnitude(acts)
    assert mag == 0.0


def test_compute_all_metrics_keys():
    """compute_all_metrics returns all expected keys."""
    acts = torch.rand(10, 5)
    m = compute_all_metrics(acts)
    assert "dead_neuron_fraction" in m
    assert "near_dead_fraction" in m
    assert "zero_fraction" in m
    assert "activation_entropy" in m
    assert "mean_activation_magnitude" in m


def test_compute_all_metrics_empty():
    """compute_all_metrics handles empty tensor."""
    acts = torch.zeros(0, 5)
    m = compute_all_metrics(acts)
    assert m["dead_neuron_fraction"] == 0.0
    assert m["mean_activation_magnitude"] == 0.0
