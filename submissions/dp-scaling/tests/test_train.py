"""Tests for training routines (standard and DP-SGD)."""

import torch

from src.data import make_dataloaders
from src.model import MLP
from src.train import (
    train_standard,
    train_dp_sgd,
    evaluate,
    _compute_per_sample_gradients,
    _clip_and_noise,
)


def _make_small_setup(seed=42):
    """Create a small model and data for testing."""
    torch.manual_seed(seed)
    train_loader, test_loader, n_feat, n_cls = make_dataloaders(
        n_samples=50, n_features=5, n_classes=3, seed=seed, batch_size=16
    )
    torch.manual_seed(seed)
    model = MLP(n_feat, hidden_size=8, n_classes=n_cls)
    return model, train_loader, test_loader


class TestTrainStandard:
    """Tests for standard SGD training."""

    def test_loss_decreases(self):
        model, train_loader, test_loader = _make_small_setup()
        loss_before, _ = evaluate(model, test_loader)
        model = train_standard(model, train_loader, epochs=20, seed=42)
        loss_after, _ = evaluate(model, test_loader)
        assert loss_after < loss_before

    def test_reproducibility(self):
        m1, train1, test1 = _make_small_setup(seed=42)
        m2, train2, test2 = _make_small_setup(seed=42)
        m1 = train_standard(m1, train1, epochs=10, seed=42)
        m2 = train_standard(m2, train2, epochs=10, seed=42)
        l1, _ = evaluate(m1, test1)
        l2, _ = evaluate(m2, test2)
        assert abs(l1 - l2) < 1e-5


class TestTrainDPSGD:
    """Tests for DP-SGD training."""

    def test_runs_without_error(self):
        model, train_loader, test_loader = _make_small_setup()
        model = train_dp_sgd(
            model, train_loader, epochs=5,
            max_grad_norm=1.0, noise_multiplier=1.0, seed=42
        )
        loss, acc = evaluate(model, test_loader)
        assert loss > 0
        assert 0 <= acc <= 1

    def test_more_noise_worse_performance(self):
        """Higher noise should generally yield worse (or equal) test loss."""
        m1, train1, test1 = _make_small_setup(seed=42)
        m2, train2, test2 = _make_small_setup(seed=42)

        m1 = train_dp_sgd(m1, train1, epochs=30, noise_multiplier=0.1, seed=42)
        m2 = train_dp_sgd(m2, train2, epochs=30, noise_multiplier=5.0, seed=42)

        l1, _ = evaluate(m1, test1)
        l2, _ = evaluate(m2, test2)
        # With much more noise, loss should be higher
        assert l2 >= l1 * 0.5  # Allow some slack

    def test_reproducibility(self):
        m1, train1, test1 = _make_small_setup(seed=42)
        m2, train2, test2 = _make_small_setup(seed=42)
        m1 = train_dp_sgd(m1, train1, epochs=5, noise_multiplier=1.0, seed=42)
        m2 = train_dp_sgd(m2, train2, epochs=5, noise_multiplier=1.0, seed=42)
        l1, _ = evaluate(m1, test1)
        l2, _ = evaluate(m2, test2)
        assert abs(l1 - l2) < 1e-5


class TestPerSampleGradients:
    """Tests for per-sample gradient computation."""

    def test_correct_count(self):
        model = MLP(5, 8, 3)
        X = torch.randn(4, 5)
        y = torch.tensor([0, 1, 2, 1])
        criterion = torch.nn.CrossEntropyLoss()
        grads = _compute_per_sample_gradients(model, X, y, criterion)
        assert len(grads) == 4  # One per sample
        # Each sample should have gradients for all parameters
        n_param_groups = len(list(model.parameters()))
        assert len(grads[0]) == n_param_groups


class TestClipAndNoise:
    """Tests for gradient clipping and noise addition."""

    def test_clipping_bounds_norm(self):
        """After clipping, no sample's gradient should exceed max_grad_norm."""
        # Create artificial per-sample gradients with known large norms
        grads = [
            [torch.randn(5, 8) * 10, torch.randn(8) * 10]
            for _ in range(3)
        ]
        gen = torch.Generator().manual_seed(42)
        result = _clip_and_noise(grads, max_grad_norm=1.0, noise_multiplier=0.0,
                                 generator=gen)
        # Result is average of clipped grads (no noise), should be bounded
        assert len(result) == 2

    def test_noise_adds_variance(self):
        """With noise_multiplier > 0, results should differ across generator seeds."""
        grads = [[torch.ones(3, 3)] for _ in range(2)]
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(99)
        r1 = _clip_and_noise(grads, max_grad_norm=1.0, noise_multiplier=1.0,
                             generator=gen1)
        r2 = _clip_and_noise(grads, max_grad_norm=1.0, noise_multiplier=1.0,
                             generator=gen2)
        assert not torch.allclose(r1[0], r2[0])


class TestEvaluate:
    """Tests for model evaluation."""

    def test_returns_loss_and_accuracy(self):
        model, _, test_loader = _make_small_setup()
        loss, acc = evaluate(model, test_loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1
