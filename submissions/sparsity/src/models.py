"""ReLU MLP models for activation sparsity experiments."""

import torch
import torch.nn as nn


class ReLUMLP(nn.Module):
    """Two-layer MLP with ReLU activation for sparsity tracking.

    Architecture: input -> Linear -> ReLU -> Linear -> output
    The hidden activations after ReLU are the ones we track for sparsity.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dim : int
        Number of neurons in the hidden layer.
    output_dim : int
        Dimension of the output.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._last_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, storing hidden activations for sparsity analysis."""
        pre_act = self.fc1(x)
        hidden = self.relu(pre_act)
        self._last_hidden = hidden.detach()
        output = self.fc2(hidden)
        return output

    def get_last_hidden(self) -> torch.Tensor:
        """Return the most recent hidden-layer activations (post-ReLU).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, hidden_dim). Values >= 0 due to ReLU.

        Raises
        ------
        RuntimeError
            If called before any forward pass.
        """
        if self._last_hidden is None:
            raise RuntimeError("No forward pass has been performed yet.")
        return self._last_hidden


def create_model(input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = 42) -> ReLUMLP:
    """Create a ReLU MLP with deterministic initialization.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer width.
    output_dim : int
        Output dimension.
    seed : int
        Random seed for reproducible weight initialization.

    Returns
    -------
    ReLUMLP
        Initialized model ready for training.
    """
    torch.manual_seed(seed)
    model = ReLUMLP(input_dim, hidden_dim, output_dim)
    return model
