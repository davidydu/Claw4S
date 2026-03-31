"""DP-SGD implementation from scratch.

Implements differentially private stochastic gradient descent following
Abadi et al. (2016) "Deep Learning with Differential Privacy":
  1. Per-sample gradient computation
  2. Per-sample gradient clipping to max norm C
  3. Gaussian noise addition: N(0, sigma^2 * C^2 * I)
  4. Privacy accounting via Renyi Differential Privacy (RDP)

No external DP libraries (e.g., opacus) are used.
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


def compute_per_sample_gradients(
    model: nn.Module,
    loss_fn: nn.Module,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> list[list[torch.Tensor]]:
    """Compute gradients for each sample individually.

    This is the naive loop-based approach: for each sample in the batch,
    compute the loss and backpropagate to get that sample's gradient.

    Args:
        model: The neural network.
        loss_fn: Loss function (e.g., CrossEntropyLoss with reduction='sum').
        X_batch: Input batch, shape (batch_size, n_features).
        y_batch: Label batch, shape (batch_size,).

    Returns:
        List of per-sample gradient lists. Each inner list contains
        gradient tensors matching model.parameters() order.
    """
    batch_size = X_batch.shape[0]
    per_sample_grads: list[list[torch.Tensor]] = []

    for i in range(batch_size):
        model.zero_grad()
        xi = X_batch[i : i + 1]  # shape (1, n_features)
        yi = y_batch[i : i + 1]  # shape (1,)
        loss_i = loss_fn(model(xi), yi)
        loss_i.backward()

        sample_grads = []
        for param in model.parameters():
            if param.grad is not None:
                sample_grads.append(param.grad.detach().clone())
            else:
                sample_grads.append(torch.zeros_like(param))
        per_sample_grads.append(sample_grads)

    return per_sample_grads


def clip_gradients(
    per_sample_grads: list[list[torch.Tensor]],
    max_norm: float,
) -> list[list[torch.Tensor]]:
    """Clip each per-sample gradient to have L2 norm at most max_norm.

    Args:
        per_sample_grads: List of per-sample gradient lists.
        max_norm: Maximum L2 norm (clipping threshold C).

    Returns:
        Clipped per-sample gradients.
    """
    clipped = []
    for sample_grads in per_sample_grads:
        # Compute total L2 norm across all parameter gradients for this sample
        total_norm_sq = sum(g.norm() ** 2 for g in sample_grads)
        total_norm = torch.sqrt(total_norm_sq)

        # Clip factor: min(1, max_norm / total_norm)
        clip_factor = min(1.0, max_norm / (total_norm.item() + 1e-12))

        clipped_grads = [g * clip_factor for g in sample_grads]
        clipped.append(clipped_grads)

    return clipped


def aggregate_and_noise(
    clipped_grads: list[list[torch.Tensor]],
    noise_multiplier: float,
    max_norm: float,
    seed: Optional[int] = None,
) -> list[torch.Tensor]:
    """Average clipped gradients and add Gaussian noise.

    Noise scale: sigma * C / batch_size, where sigma = noise_multiplier.
    Each gradient component gets independent Gaussian noise.

    Args:
        clipped_grads: List of clipped per-sample gradient lists.
        noise_multiplier: Noise multiplier sigma (higher = more private).
        max_norm: Clipping norm C.
        seed: Optional RNG seed for noise generation.

    Returns:
        List of noised average gradient tensors, one per parameter.
    """
    batch_size = len(clipped_grads)
    if batch_size == 0:
        raise ValueError("Empty gradient list")

    n_params = len(clipped_grads[0])

    if seed is not None:
        torch.manual_seed(seed)

    noised_avg_grads = []
    for p_idx in range(n_params):
        # Sum gradients across samples
        grad_sum = torch.zeros_like(clipped_grads[0][p_idx])
        for sample_grads in clipped_grads:
            grad_sum += sample_grads[p_idx]

        # Average
        grad_avg = grad_sum / batch_size

        # Add Gaussian noise: N(0, (sigma * C / batch_size)^2 * I)
        noise_std = noise_multiplier * max_norm / batch_size
        noise = torch.randn_like(grad_avg) * noise_std
        noised_avg_grads.append(grad_avg + noise)

    return noised_avg_grads


def dpsgd_step(
    model: nn.Module,
    loss_fn: nn.Module,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    learning_rate: float,
    max_norm: float,
    noise_multiplier: float,
    noise_seed: Optional[int] = None,
) -> float:
    """Execute one DP-SGD update step.

    Args:
        model: Neural network.
        loss_fn: Loss function.
        X_batch: Input batch.
        y_batch: Label batch.
        learning_rate: SGD learning rate.
        max_norm: Gradient clipping norm C.
        noise_multiplier: Noise multiplier sigma.
        noise_seed: Optional seed for noise RNG.

    Returns:
        Batch loss value (float).
    """
    # 1. Per-sample gradients
    per_sample_grads = compute_per_sample_gradients(model, loss_fn, X_batch, y_batch)

    # 2. Clip each per-sample gradient
    clipped = clip_gradients(per_sample_grads, max_norm)

    # 3. Aggregate + add noise
    noised_grads = aggregate_and_noise(clipped, noise_multiplier, max_norm, noise_seed)

    # 4. Apply gradients (SGD update)
    with torch.no_grad():
        for param, grad in zip(model.parameters(), noised_grads):
            param -= learning_rate * grad

    # Compute loss for logging
    with torch.no_grad():
        loss = loss_fn(model(X_batch), y_batch).item()

    return loss


def train_dpsgd(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    learning_rate: float,
    max_norm: float,
    noise_multiplier: float,
    n_train: int,
    delta: float = 1e-5,
    seed: int = 42,
) -> dict:
    """Train a model using DP-SGD and track metrics.

    Args:
        model: Neural network to train.
        train_loader: Training data loader.
        test_loader: Test data loader.
        n_epochs: Number of training epochs.
        learning_rate: Learning rate.
        max_norm: Gradient clipping norm C.
        noise_multiplier: Noise multiplier sigma.
        n_train: Number of training samples (for privacy accounting).
        delta: Privacy parameter delta.
        seed: Random seed.

    Returns:
        Dictionary with training results including epsilon, accuracy, etc.
    """
    torch.manual_seed(seed)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    total_steps = 0
    batch_size = train_loader.batch_size or 64

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            dpsgd_step(
                model=model,
                loss_fn=loss_fn,
                X_batch=X_batch,
                y_batch=y_batch,
                learning_rate=learning_rate,
                max_norm=max_norm,
                noise_multiplier=noise_multiplier,
            )
            total_steps += 1

    # Evaluate
    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()

    # Privacy accounting
    q = batch_size / n_train  # sampling probability
    epsilon = compute_epsilon_rdp(
        noise_multiplier=noise_multiplier,
        sample_rate=q,
        n_steps=total_steps,
        delta=delta,
    )

    return {
        "accuracy": accuracy,
        "epsilon": epsilon,
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "max_norm": max_norm,
        "n_epochs": n_epochs,
        "total_steps": total_steps,
        "seed": seed,
    }


def train_non_private(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    learning_rate: float,
    seed: int = 42,
) -> dict:
    """Train a model with standard (non-private) SGD as baseline.

    Args:
        model: Neural network.
        train_loader: Training data loader.
        test_loader: Test data loader.
        n_epochs: Number of epochs.
        learning_rate: Learning rate.
        seed: Random seed.

    Returns:
        Dictionary with training results.
    """
    torch.manual_seed(seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()

    return {
        "accuracy": accuracy,
        "epsilon": float("inf"),
        "delta": 0.0,
        "noise_multiplier": 0.0,
        "max_norm": float("inf"),
        "n_epochs": n_epochs,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Privacy Accounting via Renyi Differential Privacy (RDP)
# ---------------------------------------------------------------------------

def _compute_rdp_single_order(
    alpha: float,
    noise_multiplier: float,
    sample_rate: float,
    n_steps: int,
) -> float:
    """Compute RDP guarantee at a single order alpha.

    Uses the RDP bound for the subsampled Gaussian mechanism from
    Mironov et al. (2017) "Renyi Differential Privacy of the Sampled
    Gaussian Mechanism".

    For the Gaussian mechanism with noise multiplier sigma,
    RDP at order alpha is: alpha / (2 * sigma^2).

    With Poisson subsampling at rate q, the RDP is bounded by:
    log(1 + q^2 * C(alpha)) / (alpha - 1)
    where C(alpha) involves binomial terms.

    We use the simpler (but looser) bound:
    rdp_single_step <= log(1 + q^2 * (exp((alpha-1)/sigma^2) - 1)) / (alpha - 1)

    Then compose over n_steps: rdp_total = n_steps * rdp_single_step.

    Args:
        alpha: RDP order (must be > 1).
        noise_multiplier: Noise multiplier sigma.
        sample_rate: Subsampling probability q.
        n_steps: Number of gradient steps.

    Returns:
        Total RDP epsilon at order alpha.
    """
    if noise_multiplier <= 0:
        return float("inf")

    sigma = noise_multiplier

    # RDP of Gaussian mechanism at order alpha: alpha / (2 * sigma^2)
    rdp_gaussian = alpha / (2.0 * sigma ** 2)

    if sample_rate >= 1.0:
        # No subsampling, just compose
        return n_steps * rdp_gaussian

    # Subsampled Gaussian mechanism RDP bound
    # Use the tighter bound from Mironov (2017), Proposition 3:
    # For numerical stability, compute in log-space
    q = sample_rate

    # Simplified subsampled RDP bound:
    # rdp <= (1/(alpha-1)) * log(1 + q^2 * binom(alpha,2) * min(4/(sigma^2), exp(2/sigma^2)-1))
    # But a commonly used practical bound is:
    # rdp <= (1/(alpha-1)) * log((1-q)^alpha * (1 + q*exp((alpha-1)/(sigma^2))) ... )
    # We use the simple bound: rdp <= q^2 * alpha / (2 * sigma^2) for small q
    # (first-order Taylor expansion)
    # This is conservative for small q.

    # More precisely, for numerical stability:
    exponent = (alpha - 1.0) / (sigma ** 2)
    if exponent > 500:
        # Overflow guard: use log-space
        rdp_step = q ** 2 * alpha / (2.0 * sigma ** 2)
    else:
        exp_term = math.exp(exponent) - 1.0
        log_term = math.log1p(q * q * math.comb(int(alpha), 2) * min(
            4.0 * (math.exp(1.0 / sigma ** 2) - 1.0),
            exp_term,
        ))
        rdp_step = log_term / (alpha - 1.0)

    return n_steps * rdp_step


def _rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    """Convert RDP guarantee to (epsilon, delta)-DP.

    Uses the conversion: epsilon = rdp - log(delta) / (alpha - 1).

    Args:
        rdp: RDP epsilon at order alpha.
        alpha: RDP order.
        delta: Target delta.

    Returns:
        Epsilon for (epsilon, delta)-DP.
    """
    if rdp == float("inf"):
        return float("inf")
    return rdp - math.log(delta) / (alpha - 1.0)


def compute_epsilon_rdp(
    noise_multiplier: float,
    sample_rate: float,
    n_steps: int,
    delta: float = 1e-5,
    orders: Optional[list[float]] = None,
) -> float:
    """Compute (epsilon, delta)-DP guarantee using RDP accountant.

    Computes RDP at multiple orders and converts to the tightest
    (epsilon, delta)-DP guarantee by optimizing over orders.

    Args:
        noise_multiplier: Noise multiplier sigma.
        sample_rate: Subsampling probability q = batch_size / n.
        n_steps: Total number of gradient update steps.
        delta: Target delta for (epsilon, delta)-DP.
        orders: RDP orders to evaluate. If None, uses a default range.

    Returns:
        Epsilon value for (epsilon, delta)-DP.
    """
    if noise_multiplier <= 0:
        return float("inf")

    if orders is None:
        # Standard range of RDP orders
        orders = [1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20,
                  24, 32, 48, 64, 96, 128, 256, 512, 1024]

    best_epsilon = float("inf")
    for alpha in orders:
        rdp = _compute_rdp_single_order(alpha, noise_multiplier, sample_rate, n_steps)
        eps = _rdp_to_epsilon(rdp, alpha, delta)
        if eps < best_epsilon:
            best_epsilon = eps

    return best_epsilon
