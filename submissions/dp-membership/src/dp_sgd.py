"""DP-SGD implementation from scratch.

Implements differentially private stochastic gradient descent following
Abadi et al. (2016) "Deep Learning with Differential Privacy":
  1. Per-sample gradient computation via vmap
  2. Per-sample gradient clipping (L2 norm)
  3. Gaussian noise addition calibrated to clipping norm
  4. Simple privacy accounting (RDP -> (epsilon, delta) conversion)

No external DP library (e.g., Opacus) is used.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap


@dataclass
class DPConfig:
    """Configuration for DP-SGD training.

    Args:
        noise_multiplier: Ratio of noise std to clipping norm (sigma).
            0 means non-private training.
        max_grad_norm: L2 norm bound for per-sample gradient clipping (C).
        delta: Target delta for (epsilon, delta)-DP.
    """
    noise_multiplier: float = 0.0  # sigma; 0 = non-private
    max_grad_norm: float = 1.0     # C
    delta: float = 1e-5


def compute_per_sample_gradients(
    model: nn.Module,
    loss_fn: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-sample gradients using torch.func.vmap.

    For a batch of size B, returns a dict mapping parameter names to
    tensors of shape (B, *param_shape).

    Args:
        model: The neural network.
        loss_fn: Loss function (e.g., CrossEntropyLoss with reduction='none').
        batch_x: Input batch of shape (B, input_dim).
        batch_y: Label batch of shape (B,).

    Returns:
        Dictionary of per-sample gradients keyed by parameter name.
    """
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}

    def compute_loss_for_single_sample(params_dict, x_single, y_single):
        """Loss for one sample, as a function of parameters."""
        x_batch = x_single.unsqueeze(0)
        y_batch = y_single.unsqueeze(0)
        logits = functional_call(model, params_dict, (x_batch,))
        loss = loss_fn(logits, y_batch)
        return loss.squeeze()

    # grad w.r.t. first arg (params), then vmap over samples
    grad_fn = grad(compute_loss_for_single_sample, argnums=0)
    per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0))(params, batch_x, batch_y)

    return per_sample_grads


def clip_per_sample_gradients(
    per_sample_grads: dict[str, torch.Tensor],
    max_grad_norm: float,
) -> dict[str, torch.Tensor]:
    """Clip each sample's gradient to have L2 norm at most max_grad_norm.

    Args:
        per_sample_grads: Dict of per-sample gradients, each of shape (B, *param_shape).
        max_grad_norm: Maximum L2 norm (C).

    Returns:
        Clipped per-sample gradients.
    """
    # Compute per-sample L2 norm across all parameters
    batch_size = None
    sq_norms = []
    for name, g in per_sample_grads.items():
        if batch_size is None:
            batch_size = g.shape[0]
        # Flatten each sample's gradient and compute squared norm
        flat = g.reshape(batch_size, -1)
        sq_norms.append((flat ** 2).sum(dim=1))

    # Total norm per sample: shape (B,)
    total_sq_norm = torch.stack(sq_norms, dim=0).sum(dim=0)
    total_norm = torch.sqrt(total_sq_norm)

    # Clipping factor: min(1, C / ||g||)
    clip_factor = torch.clamp(max_grad_norm / (total_norm + 1e-8), max=1.0)

    clipped = {}
    for name, g in per_sample_grads.items():
        # Reshape clip_factor to broadcast: (B, 1, 1, ...)
        shape = [batch_size] + [1] * (g.dim() - 1)
        clipped[name] = g * clip_factor.reshape(shape)

    return clipped


def add_noise_and_aggregate(
    clipped_grads: dict[str, torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Aggregate clipped gradients and add calibrated Gaussian noise.

    Noise std = noise_multiplier * max_grad_norm / batch_size.

    Args:
        clipped_grads: Clipped per-sample gradients.
        noise_multiplier: Sigma parameter.
        max_grad_norm: Clipping norm C.
        batch_size: Number of samples in the batch (for averaging).

    Returns:
        Noisy aggregated gradients (one gradient per parameter).
    """
    noisy_agg = {}
    for name, g in clipped_grads.items():
        # Sum over samples
        summed = g.sum(dim=0)

        if noise_multiplier > 0:
            noise_std = noise_multiplier * max_grad_norm
            noise = torch.randn_like(summed) * noise_std
            summed = summed + noise

        # Average
        noisy_agg[name] = summed / batch_size

    return noisy_agg


def dp_sgd_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    dp_config: DPConfig,
) -> float:
    """Perform one DP-SGD training step.

    Steps:
        1. Compute per-sample gradients
        2. Clip per-sample gradients to max_grad_norm
        3. Aggregate and add Gaussian noise
        4. Apply optimizer step

    Args:
        model: Target model.
        optimizer: Optimizer (typically SGD).
        loss_fn: Loss function with reduction='none'.
        batch_x: Input batch.
        batch_y: Label batch.
        dp_config: DP configuration.

    Returns:
        Batch loss (float).
    """
    optimizer.zero_grad()
    batch_size = batch_x.shape[0]

    if dp_config.noise_multiplier == 0:
        # Non-private: standard training
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Step 1: Per-sample gradients
    per_sample_grads = compute_per_sample_gradients(model, loss_fn, batch_x, batch_y)

    # Step 2: Clip
    clipped = clip_per_sample_gradients(per_sample_grads, dp_config.max_grad_norm)

    # Step 3: Noise + aggregate
    noisy_grads = add_noise_and_aggregate(
        clipped, dp_config.noise_multiplier, dp_config.max_grad_norm, batch_size
    )

    # Step 4: Set gradients and step
    for name, p in model.named_parameters():
        if p.requires_grad and name in noisy_grads:
            p.grad = noisy_grads[name]

    optimizer.step()

    # Compute loss for reporting (no grad needed)
    with torch.no_grad():
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y).mean().item()

    return loss


def compute_epsilon(
    noise_multiplier: float,
    n_steps: int,
    batch_size: int,
    n_samples: int,
    delta: float = 1e-5,
) -> float:
    """Simple RDP-based privacy accounting.

    Uses Renyi Differential Privacy (RDP) at order alpha, then converts
    to (epsilon, delta)-DP. This is a simplified version of the moments
    accountant from Abadi et al. (2016).

    Args:
        noise_multiplier: Sigma parameter.
        n_steps: Number of training steps.
        batch_size: Batch size per step.
        n_samples: Total number of training samples.
        delta: Target delta.

    Returns:
        Estimated epsilon. Returns float('inf') for non-private (sigma=0).
    """
    if noise_multiplier == 0:
        return float("inf")

    # Sampling probability
    q = batch_size / n_samples

    # RDP at multiple orders, pick the tightest
    best_epsilon = float("inf")
    for alpha in [2, 5, 10, 20, 50, 100]:
        # RDP guarantee per step for subsampled Gaussian mechanism
        # Simplified bound: rdp_per_step <= q^2 * alpha / (2 * sigma^2)
        rdp_per_step = q ** 2 * alpha / (2.0 * noise_multiplier ** 2)

        # Composition over n_steps
        total_rdp = n_steps * rdp_per_step

        # Convert RDP to (epsilon, delta)-DP
        eps = total_rdp + math.log(1.0 / delta) / (alpha - 1)
        best_epsilon = min(best_epsilon, eps)

    return best_epsilon
