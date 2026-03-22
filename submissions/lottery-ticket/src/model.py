"""Two-layer ReLU MLP for lottery ticket experiments."""

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Simple 2-layer ReLU MLP.

    Architecture: input_dim -> hidden_dim (ReLU) -> output_dim
    Baseline with hidden=128 has ~25K parameters.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_nonzero_parameters(self) -> int:
        """Count number of non-zero parameters (after pruning)."""
        return sum((p != 0).sum().item() for p in self.parameters())
