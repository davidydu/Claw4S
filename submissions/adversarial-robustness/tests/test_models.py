"""Tests for MLP models."""

import torch

from src.models import TwoLayerMLP, build_model, HIDDEN_WIDTHS


class TestTwoLayerMLP:
    def test_forward_shape(self):
        model = TwoLayerMLP(input_dim=2, hidden_width=32)
        x = torch.randn(10, 2)
        out = model(x)
        assert out.shape == (10, 2)

    def test_param_count_increases(self):
        counts = []
        for w in [16, 32, 64]:
            model = TwoLayerMLP(input_dim=2, hidden_width=w)
            counts.append(model.param_count())
        assert counts[0] < counts[1] < counts[2]

    def test_param_count_formula(self):
        # For input_dim=2, hidden_width=w:
        # Layer 1: 2*w + w (weights + bias)
        # Layer 2: w*w + w
        # Layer 3: w*2 + 2
        # Total: 2w + w + w^2 + w + 2w + 2 = w^2 + 6w + 2
        for w in [16, 32, 64]:
            model = TwoLayerMLP(input_dim=2, hidden_width=w)
            expected = w * w + 6 * w + 2
            assert model.param_count() == expected, \
                f"Width {w}: expected {expected}, got {model.param_count()}"


class TestBuildModel:
    def test_reproducibility(self):
        m1 = build_model(hidden_width=32, seed=42)
        m2 = build_model(hidden_width=32, seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seeds_differ(self):
        m1 = build_model(hidden_width=32, seed=42)
        m2 = build_model(hidden_width=32, seed=99)
        params_equal = all(
            torch.allclose(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        assert not params_equal

    def test_hidden_widths_constant(self):
        assert HIDDEN_WIDTHS == [16, 32, 64, 128, 256, 512]
