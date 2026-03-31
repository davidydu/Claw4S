"""Tiny MLP model for RMT analysis."""

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """Three-layer MLP: input -> hidden1 -> hidden2 -> output.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width (same for both hidden layers).
        output_dim: Output dimension (num classes or 1 for regression).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weight_matrices(self) -> list[tuple[str, torch.Tensor]]:
        """Return list of (layer_name, weight_matrix) tuples.

        Returns weight tensors detached from the computation graph.
        Each tensor has shape (out_features, in_features).
        """
        return [
            ("fc1", self.fc1.weight.detach().clone()),
            ("fc2", self.fc2.weight.detach().clone()),
            ("fc3", self.fc3.weight.detach().clone()),
        ]
