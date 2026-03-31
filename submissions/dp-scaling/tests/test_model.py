"""Tests for model definition and parameter counting."""

import torch

from src.model import MLP, count_parameters


class TestMLP:
    """Tests for the MLP model."""

    def test_forward_shape(self):
        model = MLP(n_features=10, hidden_size=32, n_classes=5)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 5)

    def test_single_sample(self):
        model = MLP(n_features=10, hidden_size=16, n_classes=3)
        x = torch.randn(1, 10)
        out = model(x)
        assert out.shape == (1, 3)

    def test_output_is_logits(self):
        """Output should be raw logits (not softmax), can be any real number."""
        model = MLP(n_features=5, hidden_size=16, n_classes=3)
        x = torch.randn(4, 5)
        out = model(x)
        # Logits can be negative, so check they are not all positive (like softmax)
        assert out.min().item() < out.max().item()


class TestCountParameters:
    """Tests for count_parameters."""

    def test_known_count(self):
        """MLP(10, 16, 5): fc1 = 10*16+16=176, fc2 = 16*5+5=85, total=261."""
        model = MLP(n_features=10, hidden_size=16, n_classes=5)
        assert count_parameters(model) == 261

    def test_increasing_with_hidden_size(self):
        counts = []
        for h in [16, 32, 64, 128]:
            model = MLP(n_features=10, hidden_size=h, n_classes=5)
            counts.append(count_parameters(model))
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)  # All different
