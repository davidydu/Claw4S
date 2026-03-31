"""Tests for the ModularMLP model."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import ModularMLP, PRIME


def test_model_output_shape():
    """Output shape should be (batch, p)."""
    model = ModularMLP(p=PRIME, embed_dim=32, hidden_dim=64)
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    out = model(a, b)
    assert out.shape == (3, PRIME)


def test_model_output_range():
    """Output logits should be finite."""
    model = ModularMLP(p=PRIME)
    a = torch.randint(0, PRIME, (10,))
    b = torch.randint(0, PRIME, (10,))
    out = model(a, b)
    assert torch.isfinite(out).all()


def test_model_deterministic():
    """Same seed should produce same model outputs."""
    torch.manual_seed(42)
    model1 = ModularMLP(p=PRIME)
    torch.manual_seed(42)
    model2 = ModularMLP(p=PRIME)

    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    assert torch.allclose(model1(a, b), model2(a, b))


def test_model_small_prime():
    """Model should work with smaller primes."""
    model = ModularMLP(p=5, embed_dim=8, hidden_dim=16)
    a = torch.tensor([0, 1, 2, 3, 4])
    b = torch.tensor([4, 3, 2, 1, 0])
    out = model(a, b)
    assert out.shape == (5, 5)
