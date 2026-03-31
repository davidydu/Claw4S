"""Tests for models (RandomFeaturesModel and MLP)."""

import torch
import pytest

from src.model import (
    RandomFeaturesModel,
    MLP,
    count_parameters,
    get_interpolation_threshold,
    create_mlp,
)


class TestRandomFeaturesModel:
    """Tests for RandomFeaturesModel."""

    def test_transform_shape(self):
        model = RandomFeaturesModel(input_dim=10, n_features=50)
        X = torch.randn(5, 10)
        phi = model.transform(X)
        assert phi.shape == (5, 50)

    def test_fit_and_predict(self):
        model = RandomFeaturesModel(input_dim=10, n_features=50)
        X = torch.randn(20, 10)
        y = torch.randn(20, 1)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (20, 1)

    def test_predict_without_fit_raises(self):
        model = RandomFeaturesModel(input_dim=10, n_features=50)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(torch.randn(5, 10))

    def test_deterministic_with_seed(self):
        m1 = RandomFeaturesModel(input_dim=10, n_features=50, seed=42)
        m2 = RandomFeaturesModel(input_dim=10, n_features=50, seed=42)
        X = torch.randn(5, 10)
        assert torch.allclose(m1.transform(X), m2.transform(X))

    def test_different_seeds_differ(self):
        m1 = RandomFeaturesModel(input_dim=10, n_features=50, seed=42)
        m2 = RandomFeaturesModel(input_dim=10, n_features=50, seed=99)
        X = torch.randn(5, 10)
        assert not torch.allclose(m1.transform(X), m2.transform(X))

    def test_n_params(self):
        model = RandomFeaturesModel(input_dim=10, n_features=50)
        assert model.n_params == 50

    def test_relu_nonnegativity(self):
        model = RandomFeaturesModel(input_dim=10, n_features=50)
        X = torch.randn(100, 10)
        phi = model.transform(X)
        assert (phi >= 0).all(), "ReLU output should be non-negative"

    def test_overparameterized_fits_exactly(self):
        """When p > n, should achieve near-zero training error."""
        n = 20
        model = RandomFeaturesModel(input_dim=5, n_features=100, seed=42)
        X = torch.randn(n, 5)
        y = torch.randn(n, 1)
        model.fit(X, y)
        pred = model.predict(X)
        mse = ((pred - y) ** 2).mean().item()
        assert mse < 0.01, f"Overparameterized should fit nearly exactly, got MSE={mse}"


class TestMLP:
    """Tests for MLP class."""

    def test_forward_pass_shape(self):
        model = MLP(input_dim=10, hidden_width=20)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 1)

    def test_different_widths(self):
        for h in [1, 5, 50, 500]:
            model = MLP(input_dim=10, hidden_width=h)
            x = torch.randn(3, 10)
            out = model(x)
            assert out.shape == (3, 1)


class TestCountParameters:
    """Tests for count_parameters."""

    def test_formula(self):
        # MLP(d, h): params = h*(d+1) + (h+1) = h*(d+2) + 1
        for d, h in [(10, 5), (10, 20), (5, 100), (20, 50)]:
            model = MLP(input_dim=d, hidden_width=h)
            expected = h * (d + 2) + 1
            assert count_parameters(model) == expected


class TestGetInterpolationThreshold:
    """Tests for get_interpolation_threshold."""

    def test_returns_n_train(self):
        # For random features, threshold = n_train
        assert get_interpolation_threshold(200, 10) == 200
        assert get_interpolation_threshold(100, 20) == 100


class TestCreateMlp:
    """Tests for create_mlp."""

    def test_deterministic(self):
        m1 = create_mlp(10, 20, seed=42)
        m2 = create_mlp(10, 20, seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_cpu_device(self):
        model = create_mlp(10, 20)
        for p in model.parameters():
            assert p.device.type == "cpu"
