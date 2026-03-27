"""Tests for MLP model construction."""

import pytest
import torch

from src.models import MLP, build_model, build_width_model, count_parameters, ARCH_CONFIGS


class TestMLP:
    def test_forward_shape(self):
        model = MLP(in_features=10, hidden_width=64, n_hidden_layers=2, n_classes=5)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 5)

    def test_single_hidden_layer(self):
        model = MLP(in_features=10, hidden_width=100, n_hidden_layers=1, n_classes=5)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 5)

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="n_hidden_layers must be >= 1"):
            MLP(in_features=10, hidden_width=32, n_hidden_layers=0, n_classes=5)


class TestBuildModel:
    def test_all_arch_configs(self):
        for arch_name in ARCH_CONFIGS:
            model = build_model(arch_name)
            x = torch.randn(8, 10)
            out = model(x)
            assert out.shape == (8, 5), f"Failed for {arch_name}"

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unknown arch"):
            build_model("nonexistent-arch")

    def test_param_counts_reasonable(self):
        """All named architectures should have roughly ~10K params."""
        for arch_name in ARCH_CONFIGS:
            model = build_model(arch_name)
            n = count_parameters(model)
            assert 1000 < n < 100000, (
                f"{arch_name} has {n} params, expected ~10K"
            )


class TestBuildWidthModel:
    def test_forward_shape(self):
        model = build_width_model(width=64)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 5)

    def test_width_affects_param_count(self):
        small = count_parameters(build_width_model(16))
        large = count_parameters(build_width_model(256))
        assert large > small
