"""Tests for DP-SGD implementation."""

import torch
import torch.nn as nn
import numpy as np

from src.model import MLP
from src.dp_sgd import (
    DPConfig,
    compute_per_sample_gradients,
    clip_per_sample_gradients,
    add_noise_and_aggregate,
    dp_sgd_step,
    compute_epsilon,
)


def test_per_sample_gradients_shape():
    """Per-sample gradients have shape (batch_size, *param_shape)."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=8, num_classes=3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 2, 0])

    grads = compute_per_sample_gradients(model, loss_fn, x, y)

    for name, g in grads.items():
        assert g.shape[0] == 4, f"Batch dim wrong for {name}: {g.shape}"


def test_clipping_reduces_norm():
    """Clipped gradients have L2 norm <= max_grad_norm."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=8, num_classes=3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 2, 0])

    grads = compute_per_sample_gradients(model, loss_fn, x, y)
    max_norm = 0.5
    clipped = clip_per_sample_gradients(grads, max_norm)

    # Check per-sample norms
    for i in range(4):
        sq_sum = sum((clipped[n][i] ** 2).sum().item() for n in clipped)
        norm = sq_sum ** 0.5
        assert norm <= max_norm + 1e-5, f"Sample {i} norm {norm:.4f} > {max_norm}"


def test_noise_changes_gradients():
    """Adding noise (sigma > 0) changes the aggregated gradient."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=8, num_classes=3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 2, 0])

    grads = compute_per_sample_gradients(model, loss_fn, x, y)
    clipped = clip_per_sample_gradients(grads, max_grad_norm=1.0)

    no_noise = add_noise_and_aggregate(clipped, noise_multiplier=0.0, max_grad_norm=1.0, batch_size=4)
    with_noise = add_noise_and_aggregate(clipped, noise_multiplier=5.0, max_grad_norm=1.0, batch_size=4)

    # At least one parameter should differ
    differs = False
    for name in no_noise:
        if not torch.allclose(no_noise[name], with_noise[name], atol=1e-6):
            differs = True
            break
    assert differs, "Noise should change the aggregated gradient"


def test_dp_sgd_step_runs():
    """DP-SGD step completes without error."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=8, num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 2, 0])

    config = DPConfig(noise_multiplier=1.0, max_grad_norm=1.0)
    loss = dp_sgd_step(model, optimizer, loss_fn, x, y, config)
    assert isinstance(loss, float)
    assert loss >= 0


def test_non_private_step_runs():
    """Non-private step (sigma=0) completes."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=8, num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 2, 0])

    config = DPConfig(noise_multiplier=0.0)
    loss = dp_sgd_step(model, optimizer, loss_fn, x, y, config)
    assert isinstance(loss, float)


def test_compute_epsilon_non_private():
    """Non-private has infinite epsilon."""
    eps = compute_epsilon(noise_multiplier=0.0, n_steps=100, batch_size=32, n_samples=250)
    assert eps == float("inf")


def test_compute_epsilon_increases_with_steps():
    """More steps = larger epsilon (less privacy)."""
    eps_few = compute_epsilon(noise_multiplier=1.0, n_steps=10, batch_size=32, n_samples=250)
    eps_many = compute_epsilon(noise_multiplier=1.0, n_steps=1000, batch_size=32, n_samples=250)
    assert eps_many > eps_few


def test_compute_epsilon_decreases_with_noise():
    """More noise = smaller epsilon (more privacy)."""
    eps_low = compute_epsilon(noise_multiplier=0.5, n_steps=100, batch_size=32, n_samples=250)
    eps_high = compute_epsilon(noise_multiplier=5.0, n_steps=100, batch_size=32, n_samples=250)
    assert eps_high < eps_low
