"""Tests for neural network models."""

import torch
from src.models import TwoLayerMLP


class TestTwoLayerMLP:
    def test_forward_classification(self):
        model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 5)

    def test_forward_regression(self):
        model = TwoLayerMLP(input_dim=1, hidden_dim=32, output_dim=1)
        x = torch.randn(8, 1)
        out = model(x)
        assert out.shape == (8, 1)

    def test_layer_names(self):
        model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
        names = model.get_layer_names()
        assert "layer1" in names
        assert "layer2" in names
        assert len(names) == 2

    def test_gradients_flow(self):
        model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
        x = torch.randn(4, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name in model.get_layer_names():
            layer = getattr(model, name)
            for p in layer.parameters():
                assert p.grad is not None
                assert p.grad.norm() > 0

    def test_parameter_count(self):
        model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
        # layer1: 10*32 + 32 = 352, layer2: 32*5 + 5 = 165
        total = sum(p.numel() for p in model.parameters())
        assert total == 352 + 165
