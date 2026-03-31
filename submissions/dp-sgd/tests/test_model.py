"""Tests for MLP model."""

import torch
import pytest

from src.model import MLP, create_model


class TestMLP:
    """Tests for MLP architecture."""

    def test_output_shape(self):
        model = MLP(n_features=10, n_hidden=64, n_classes=5)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 5)

    def test_single_sample(self):
        model = MLP(n_features=10, n_hidden=64, n_classes=5)
        x = torch.randn(1, 10)
        out = model(x)
        assert out.shape == (1, 5)

    def test_parameter_count(self):
        model = MLP(n_features=10, n_hidden=64, n_classes=5)
        n_params = sum(p.numel() for p in model.parameters())
        # fc1: 10*64 + 64 = 704, fc2: 64*5 + 5 = 325, total = 1029
        assert n_params == 1029


class TestCreateModel:
    """Tests for model creation with seed control."""

    def test_reproducibility(self):
        m1 = create_model(seed=42)
        m2 = create_model(seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = create_model(seed=42)
        m2 = create_model(seed=123)
        params_equal = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        assert not params_equal
