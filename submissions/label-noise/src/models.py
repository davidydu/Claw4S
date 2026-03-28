"""MLP architectures with configurable depth and width."""

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations.

    Parameters
    ----------
    in_features : int
        Input dimension.
    hidden_width : int
        Width of every hidden layer.
    n_hidden_layers : int
        Number of hidden layers (depth).
    n_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        in_features: int,
        hidden_width: int,
        n_hidden_layers: int,
        n_classes: int,
    ):
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev, hidden_width))
            layers.append(nn.ReLU())
            prev = hidden_width
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Pre-defined architecture configs in the same small-model regime
# (3.2K-6.1K trainable params with 10 input features and 5 classes).
# ---------------------------------------------------------------------------

ARCH_CONFIGS = {
    # name: (n_hidden_layers, hidden_width, description)
    "shallow-wide": (1, 200, "1 hidden layer, width 200"),
    "medium":       (2, 70,  "2 hidden layers, width 70"),
    "deep-narrow":  (4, 35,  "4 hidden layers, width 35"),
}

WIDTH_SWEEP_WIDTHS = [16, 32, 64, 128, 256]
WIDTH_SWEEP_DEPTH = 2  # fixed depth for width sweep


def build_model(
    arch_name: str,
    in_features: int = 10,
    n_classes: int = 5,
) -> MLP:
    """Instantiate an MLP from a named architecture config."""
    if arch_name not in ARCH_CONFIGS:
        raise ValueError(
            f"Unknown arch '{arch_name}'. Choose from {list(ARCH_CONFIGS.keys())}"
        )
    depth, width, _ = ARCH_CONFIGS[arch_name]
    return MLP(in_features, width, depth, n_classes)


def build_width_model(
    width: int,
    in_features: int = 10,
    n_classes: int = 5,
) -> MLP:
    """Instantiate a depth-2 MLP with the given width (for width sweep)."""
    return MLP(in_features, width, WIDTH_SWEEP_DEPTH, n_classes)
