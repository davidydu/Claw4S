"""Tests for MLP model construction and parameter counting."""

import pytest
import torch

from src.models import FlexibleMLP, compute_width_for_budget, count_parameters


class TestComputeWidth:
    """Tests for compute_width_for_budget."""

    def test_single_hidden_layer(self):
        """Width computed for 1 hidden layer should yield close to budget."""
        width = compute_width_for_budget(
            input_dim=10, output_dim=5, num_hidden_layers=1, param_budget=1000
        )
        assert width > 0
        model = FlexibleMLP(10, width, 5, 1)
        actual = count_parameters(model)
        assert abs(actual - 1000) / 1000 < 0.15

    def test_multiple_hidden_layers(self):
        """Width computed for 4 hidden layers should yield close to budget."""
        width = compute_width_for_budget(
            input_dim=20, output_dim=10, num_hidden_layers=4, param_budget=5000
        )
        assert width > 0
        model = FlexibleMLP(20, width, 10, 4)
        actual = count_parameters(model)
        assert abs(actual - 5000) / 5000 < 0.15

    def test_depth_1_vs_depth_8_same_budget(self):
        """Deeper networks should have narrower layers for same budget."""
        w1 = compute_width_for_budget(
            input_dim=50, output_dim=10, num_hidden_layers=1, param_budget=10000
        )
        w8 = compute_width_for_budget(
            input_dim=50, output_dim=10, num_hidden_layers=8, param_budget=10000
        )
        assert w1 > w8, "Shallow network should be wider than deep one"

    def test_invalid_depth_raises(self):
        """Depth < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
            compute_width_for_budget(10, 5, 0, 1000)


class TestFlexibleMLP:
    """Tests for FlexibleMLP architecture."""

    def test_output_shape_classification(self):
        """Output shape should match output_dim for classification."""
        model = FlexibleMLP(194, 64, 97, 2)
        x = torch.randn(32, 194)
        out = model(x)
        assert out.shape == (32, 97)

    def test_output_shape_regression(self):
        """Output shape should match output_dim=1 for regression."""
        model = FlexibleMLP(8, 32, 1, 3)
        x = torch.randn(16, 8)
        out = model(x)
        assert out.shape == (16, 1)

    def test_depth_1(self):
        """Model with depth 1 should work correctly."""
        model = FlexibleMLP(10, 20, 5, 1)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_depth_8(self):
        """Model with depth 8 should work correctly."""
        model = FlexibleMLP(10, 16, 5, 8)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_invalid_depth_raises(self):
        """Depth < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_hidden_layers must be >= 1"):
            FlexibleMLP(10, 20, 5, 0)

    def test_deterministic_output(self):
        """Same seed should produce same model weights."""
        torch.manual_seed(42)
        m1 = FlexibleMLP(10, 20, 5, 2)
        torch.manual_seed(42)
        m2 = FlexibleMLP(10, 20, 5, 2)
        x = torch.randn(4, 10)
        assert torch.allclose(m1(x), m2(x))
