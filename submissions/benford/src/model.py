"""Tiny MLP model definition for Benford's Law experiments."""

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """A small multi-layer perceptron with ReLU activations.

    Architecture: Input -> [Linear -> ReLU] x n_hidden -> Linear -> Output

    Args:
        d_in: Input dimension.
        d_hidden: Hidden layer dimension.
        d_out: Output dimension.
        n_hidden: Number of hidden layers (default 2).
    """

    def __init__(self, d_in, d_hidden, d_out, n_hidden=2):
        super().__init__()
        layers = []
        # First hidden layer
        layers.append(nn.Linear(d_in, d_hidden))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_weight_layers(self):
        """Return dict mapping layer names to weight tensors (excluding biases).

        Returns:
            Dict of {name: tensor} for each Linear layer's weight parameter.
        """
        result = {}
        for name, param in self.named_parameters():
            if "weight" in name:
                result[name] = param.data.clone()
        return result

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
