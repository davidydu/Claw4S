"""MLP architectures with configurable depth and width."""

import math
import torch
import torch.nn as nn


def compute_width_for_budget(
    input_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    param_budget: int,
) -> int:
    """Compute hidden width to match a target parameter budget.

    For an MLP with L hidden layers of width W:
        params = input_dim*W + W  (first layer + bias)
               + (L-1)*(W*W + W)  (hidden-to-hidden layers + biases)
               + W*output_dim + output_dim  (output layer + bias)

    We solve for W given L and the budget.

    Args:
        input_dim: Size of input features.
        output_dim: Size of output layer.
        num_hidden_layers: Number of hidden layers (L >= 1).
        param_budget: Target total parameter count.

    Returns:
        Hidden layer width (rounded to nearest positive integer).

    Raises:
        ValueError: If num_hidden_layers < 1 or no positive width solution exists.
    """
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1")

    # Subtract output bias from budget (it doesn't depend on W)
    remaining = param_budget - output_dim

    if num_hidden_layers == 1:
        # params = input_dim*W + W + W*output_dim + output_dim
        # remaining = W*(input_dim + 1 + output_dim)
        divisor = input_dim + 1 + output_dim
        width = remaining / divisor
    else:
        # Quadratic in W:
        # remaining = (input_dim + 1)*W + (L-1)*(W^2 + W) + output_dim*W
        # remaining = (L-1)*W^2 + (input_dim + 1 + (L-1) + output_dim)*W
        a = num_hidden_layers - 1
        b = input_dim + 1 + (num_hidden_layers - 1) + output_dim
        c = -remaining
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            raise ValueError(
                f"No positive width solution for budget={param_budget}, "
                f"depth={num_hidden_layers}"
            )
        width = (-b + math.sqrt(discriminant)) / (2 * a)

    width = max(1, round(width))
    return width


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FlexibleMLP(nn.Module):
    """MLP with configurable depth and width.

    Architecture: input -> [hidden layers with ReLU] -> output
    Uses Kaiming initialization for stable training across depths.

    Args:
        input_dim: Input feature size.
        hidden_width: Width of each hidden layer.
        output_dim: Output size.
        num_hidden_layers: Number of hidden layers (>= 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        output_dim: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_width, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming initialization for stable deep training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
