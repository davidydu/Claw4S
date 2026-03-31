"""Two-layer ReLU MLP for modular addition with partially symmetric init."""

import torch
import torch.nn as nn
from typing import Optional


class SymmetricMLP(nn.Module):
    """Two-layer ReLU MLP with configurable hidden-layer symmetry.

    Architecture: input -> Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)

    Symmetric init sets all rows of the first hidden layer to the same value,
    then adds a small perturbation epsilon * N(0,1). The readout layer keeps
    a seeded Kaiming initialization, so only the incoming hidden weights start
    symmetric.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        epsilon: float,
        seed: int = 42,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self._symmetric_init(seed)

    def _symmetric_init(self, seed: int) -> None:
        """Initialize fc1 with symmetric weights + epsilon perturbation.

        All rows of fc1.weight are set to the same base vector, then
        a perturbation of scale epsilon is added. fc1.bias is set to zero.
        fc2 is initialized with standard Kaiming uniform (seeded).
        """
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Base vector: use a fixed value (0.1) for all elements
        base_weight = torch.full(
            (1, self.input_dim), 0.1, dtype=torch.float32
        )

        # Expand to all rows (all neurons start identical)
        symmetric_weight = base_weight.expand(self.hidden_dim, self.input_dim).clone()

        # Add perturbation
        if self.epsilon > 0:
            perturbation = torch.randn(
                self.hidden_dim, self.input_dim, generator=gen
            )
            symmetric_weight = symmetric_weight + self.epsilon * perturbation

        self.fc1.weight.data = symmetric_weight
        self.fc1.bias.data.zero_()

        # Second layer: Kaiming uniform (standard), seeded for reproducibility
        gen2 = torch.Generator()
        gen2.manual_seed(seed + 1000)
        nn.init.kaiming_uniform_(self.fc2.weight, generator=gen2)
        fan_in = self.fc2.weight.size(1)
        bound = 1.0 / (fan_in ** 0.5)
        self.fc2.bias.data.uniform_(-bound, bound, generator=gen2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> fc1 -> ReLU -> fc2."""
        return self.fc2(self.relu(self.fc1(x)))
