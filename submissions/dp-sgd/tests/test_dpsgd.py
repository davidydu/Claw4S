"""Tests for DP-SGD implementation."""

import math
import torch
import torch.nn as nn
import pytest

from src.model import create_model
from src.dpsgd import (
    compute_per_sample_gradients,
    clip_gradients,
    aggregate_and_noise,
    dpsgd_step,
    compute_epsilon_rdp,
    train_dpsgd,
    train_non_private,
)
from src.data import make_dataloaders


class TestPerSampleGradients:
    """Tests for per-sample gradient computation."""

    def test_correct_count(self):
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        X = torch.randn(5, 4)
        y = torch.tensor([0, 1, 2, 0, 1])
        grads = compute_per_sample_gradients(model, loss_fn, X, y)
        assert len(grads) == 5  # one per sample

    def test_gradient_shapes_match_params(self):
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        X = torch.randn(3, 4)
        y = torch.tensor([0, 1, 2])
        grads = compute_per_sample_gradients(model, loss_fn, X, y)
        param_shapes = [p.shape for p in model.parameters()]
        for sample_grads in grads:
            for grad, expected_shape in zip(sample_grads, param_shapes):
                assert grad.shape == expected_shape

    def test_different_samples_different_grads(self):
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        X = torch.randn(2, 4)
        y = torch.tensor([0, 1])
        grads = compute_per_sample_gradients(model, loss_fn, X, y)
        # Gradients for different samples should differ
        g0_flat = torch.cat([g.flatten() for g in grads[0]])
        g1_flat = torch.cat([g.flatten() for g in grads[1]])
        assert not torch.allclose(g0_flat, g1_flat)


class TestClipGradients:
    """Tests for gradient clipping."""

    def test_clipping_reduces_norm(self):
        # Create a gradient with large norm
        grads = [[torch.randn(4, 8) * 10, torch.randn(8) * 10]]
        clipped = clip_gradients(grads, max_norm=1.0)
        total_norm = sum(g.norm() ** 2 for g in clipped[0]) ** 0.5
        assert total_norm.item() <= 1.0 + 1e-6

    def test_small_grad_unchanged(self):
        # Create a gradient with very small norm
        grads = [[torch.randn(4, 8) * 0.001, torch.randn(8) * 0.001]]
        clipped = clip_gradients(grads, max_norm=100.0)
        for orig, clip in zip(grads[0], clipped[0]):
            assert torch.allclose(orig, clip, atol=1e-10)

    def test_multiple_samples(self):
        grads = [
            [torch.randn(4, 8) * 10],
            [torch.randn(4, 8) * 0.01],
        ]
        clipped = clip_gradients(grads, max_norm=1.0)
        assert len(clipped) == 2
        # First should be clipped (large), second should be ~unchanged
        norm0 = clipped[0][0].norm().item()
        assert norm0 <= 1.0 + 1e-6


class TestAggregateAndNoise:
    """Tests for gradient aggregation and noise addition."""

    def test_output_shapes(self):
        clipped = [
            [torch.randn(4, 8), torch.randn(8)],
            [torch.randn(4, 8), torch.randn(8)],
        ]
        result = aggregate_and_noise(clipped, noise_multiplier=1.0, max_norm=1.0)
        assert len(result) == 2
        assert result[0].shape == (4, 8)
        assert result[1].shape == (8,)

    def test_zero_noise_equals_mean(self):
        torch.manual_seed(42)
        g1 = [torch.tensor([1.0, 2.0, 3.0])]
        g2 = [torch.tensor([4.0, 5.0, 6.0])]
        clipped = [g1, g2]

        # With zero noise multiplier, result should be the average
        result = aggregate_and_noise(clipped, noise_multiplier=0.0, max_norm=1.0,
                                     seed=42)
        expected = torch.tensor([2.5, 3.5, 4.5])
        assert torch.allclose(result[0], expected)

    def test_noise_adds_variance(self):
        clipped = [[torch.zeros(100)] for _ in range(10)]
        # With large noise, result should not be near zero
        result = aggregate_and_noise(
            clipped, noise_multiplier=100.0, max_norm=1.0, seed=42,
        )
        assert result[0].abs().mean().item() > 0.1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty gradient list"):
            aggregate_and_noise([], noise_multiplier=1.0, max_norm=1.0)


class TestDpsgdStep:
    """Tests for a single DP-SGD update step."""

    def test_parameters_change(self):
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        old_params = [p.clone() for p in model.parameters()]

        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        X = torch.randn(4, 4)
        y = torch.tensor([0, 1, 2, 0])

        dpsgd_step(model, loss_fn, X, y,
                    learning_rate=0.1, max_norm=1.0, noise_multiplier=0.1)

        params_changed = any(
            not torch.equal(old, new)
            for old, new in zip(old_params, model.parameters())
        )
        assert params_changed

    def test_returns_loss(self):
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        X = torch.randn(4, 4)
        y = torch.tensor([0, 1, 2, 0])

        loss = dpsgd_step(model, loss_fn, X, y,
                          learning_rate=0.1, max_norm=1.0, noise_multiplier=0.1)
        assert isinstance(loss, float)
        assert loss >= 0


class TestPrivacyAccounting:
    """Tests for RDP-based privacy accounting."""

    def test_epsilon_positive(self):
        eps = compute_epsilon_rdp(
            noise_multiplier=1.0, sample_rate=0.1,
            n_steps=100, delta=1e-5,
        )
        assert eps > 0

    def test_more_noise_less_epsilon(self):
        eps_low_noise = compute_epsilon_rdp(
            noise_multiplier=0.5, sample_rate=0.1,
            n_steps=100, delta=1e-5,
        )
        eps_high_noise = compute_epsilon_rdp(
            noise_multiplier=5.0, sample_rate=0.1,
            n_steps=100, delta=1e-5,
        )
        assert eps_high_noise < eps_low_noise

    def test_more_steps_more_epsilon(self):
        eps_few = compute_epsilon_rdp(
            noise_multiplier=1.0, sample_rate=0.1,
            n_steps=10, delta=1e-5,
        )
        eps_many = compute_epsilon_rdp(
            noise_multiplier=1.0, sample_rate=0.1,
            n_steps=1000, delta=1e-5,
        )
        assert eps_many > eps_few

    def test_zero_noise_infinite_epsilon(self):
        eps = compute_epsilon_rdp(
            noise_multiplier=0.0, sample_rate=0.1,
            n_steps=100, delta=1e-5,
        )
        assert eps == float("inf")

    def test_finite_epsilon_values(self):
        """Common configs should give finite epsilon."""
        for sigma in [0.5, 1.0, 2.0, 5.0]:
            eps = compute_epsilon_rdp(
                noise_multiplier=sigma, sample_rate=0.16,
                n_steps=120, delta=1e-5,
            )
            assert math.isfinite(eps), f"sigma={sigma} gave non-finite epsilon"
            assert eps > 0


class TestTrainDpsgd:
    """Integration tests for DP-SGD training."""

    def test_returns_expected_keys(self):
        train_loader, test_loader, n_train, _ = make_dataloaders(
            n_samples=50, n_features=4, n_classes=3, batch_size=16,
        )
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        result = train_dpsgd(
            model=model, train_loader=train_loader,
            test_loader=test_loader, n_epochs=2,
            learning_rate=0.1, max_norm=1.0,
            noise_multiplier=1.0, n_train=n_train,
        )
        expected_keys = {
            "accuracy", "epsilon", "delta",
            "noise_multiplier", "max_norm", "n_epochs", "seed",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_accuracy_in_range(self):
        train_loader, test_loader, n_train, _ = make_dataloaders(
            n_samples=50, n_features=4, n_classes=3, batch_size=16,
        )
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        result = train_dpsgd(
            model=model, train_loader=train_loader,
            test_loader=test_loader, n_epochs=2,
            learning_rate=0.1, max_norm=1.0,
            noise_multiplier=1.0, n_train=n_train,
        )
        assert 0.0 <= result["accuracy"] <= 1.0


class TestTrainNonPrivate:
    """Tests for non-private baseline training."""

    def test_returns_expected_keys(self):
        train_loader, test_loader, _, _ = make_dataloaders(
            n_samples=50, n_features=4, n_classes=3, batch_size=16,
        )
        model = create_model(n_features=4, n_hidden=8, n_classes=3, seed=42)
        result = train_non_private(
            model=model, train_loader=train_loader,
            test_loader=test_loader, n_epochs=5,
            learning_rate=0.1,
        )
        assert "accuracy" in result
        assert result["epsilon"] == float("inf")

    def test_learns_something(self):
        """Non-private training should achieve above-chance accuracy."""
        train_loader, test_loader, _, _ = make_dataloaders(
            n_samples=200, n_features=10, n_classes=5,
            batch_size=32, seed=42,
        )
        model = create_model(n_features=10, n_hidden=64, n_classes=5, seed=42)
        result = train_non_private(
            model=model, train_loader=train_loader,
            test_loader=test_loader, n_epochs=30,
            learning_rate=0.1,
        )
        # Random chance for 5 classes = 0.2
        assert result["accuracy"] > 0.3, (
            f"Non-private accuracy {result['accuracy']:.4f} is too low"
        )
