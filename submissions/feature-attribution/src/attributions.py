"""Gradient-based feature attribution methods.

Implements three standard attribution methods:
  1. Vanilla Gradient:      |dL/dx|
  2. Gradient x Input:      |x * dL/dx|
  3. Integrated Gradients:  sum_alpha (dL/dx_alpha) * delta_x  (Sundararajan et al. 2017)

All methods return absolute attribution magnitudes per feature.
"""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np


def _compute_gradient(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    """Compute gradient of the target class logit w.r.t. input x.

    Args:
        model: Trained model.
        x: Single input sample (1, d), requires_grad=True.
        target_class: Class index for which to compute the gradient.

    Returns:
        Gradient tensor of shape (d,).
    """
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    score = logits[0, target_class]
    score.backward()
    assert x.grad is not None, "Gradient computation failed"
    return x.grad[0].detach()


def vanilla_gradient(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Vanilla gradient attribution: |dL/dx|.

    Args:
        model: Trained model.
        x: Single input (1, d).
        target_class: Target class index.

    Returns:
        Absolute gradient magnitudes (d,) as numpy array.
    """
    grad = _compute_gradient(model, x, target_class)
    return torch.abs(grad).numpy()


def gradient_times_input(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Gradient x Input attribution: |x * dL/dx|.

    Args:
        model: Trained model.
        x: Single input (1, d).
        target_class: Target class index.

    Returns:
        Absolute element-wise product (d,) as numpy array.
    """
    grad = _compute_gradient(model, x, target_class)
    return torch.abs(x[0] * grad).numpy()


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Integrated Gradients attribution (Sundararajan et al. 2017).

    Approximates the path integral from a baseline (default: zero vector)
    to the input using the trapezoidal rule.

    IG(x) = (x - x') * integral_0^1 (dF/dx)(x' + alpha*(x - x')) d_alpha

    Args:
        model: Trained model.
        x: Single input (1, d).
        target_class: Target class index.
        n_steps: Number of interpolation steps (50 default per paper).
        baseline: Baseline input (1, d). Defaults to zero vector.

    Returns:
        Absolute integrated gradients (d,) as numpy array.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    delta = x - baseline
    grads = []

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * delta
        grad = _compute_gradient(model, interpolated, target_class)
        grads.append(grad)

    # Trapezoidal rule: average of consecutive pairs
    stacked = torch.stack(grads, dim=0)  # (n_steps+1, d)
    avg_grad = (stacked[:-1] + stacked[1:]).sum(dim=0) / (2.0 * n_steps)
    ig = delta[0] * avg_grad
    return torch.abs(ig).numpy()


METHODS = {
    "vanilla_gradient": vanilla_gradient,
    "gradient_x_input": gradient_times_input,
    "integrated_gradients": integrated_gradients,
}

METHOD_PAIRS = [
    ("vanilla_gradient", "gradient_x_input"),
    ("vanilla_gradient", "integrated_gradients"),
    ("gradient_x_input", "integrated_gradients"),
]


def compute_all_attributions(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute all three attribution methods for a single sample.

    Args:
        model: Trained model.
        x: Single input (1, d).
        target_class: Target class index.
        n_steps: Integration steps for IG.

    Returns:
        Dict mapping method name to attribution array (d,).
    """
    model.eval()
    return {
        "vanilla_gradient": vanilla_gradient(model, x, target_class),
        "gradient_x_input": gradient_times_input(model, x, target_class),
        "integrated_gradients": integrated_gradients(
            model, x, target_class, n_steps=n_steps
        ),
    }
