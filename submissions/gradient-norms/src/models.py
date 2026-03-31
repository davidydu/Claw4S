"""Neural network models for gradient norm phase transition experiments."""

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Simple 2-layer MLP for studying training dynamics.

    Architecture: input -> Linear -> ReLU -> Linear -> output
    Named layers: 'layer1' (input->hidden), 'layer2' (hidden->output)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.relu(self.layer1(x)))

    def get_layer_names(self) -> list[str]:
        """Return names of parameterized layers (for gradient tracking)."""
        return ["layer1", "layer2"]
