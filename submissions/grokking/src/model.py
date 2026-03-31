"""Tiny MLP model for modular arithmetic grokking experiments.

Architecture: Two learned embeddings (one per input), concatenated,
fed through a 1-hidden-layer MLP with ReLU activation.
"""

import torch
import torch.nn as nn


class GrokkingMLP(nn.Module):
    """Small MLP for learning modular arithmetic.

    Architecture:
        Embed(a) || Embed(b) -> Linear -> ReLU -> Linear -> output(p)

    Args:
        p: Prime modulus (vocabulary size and number of output classes).
        embed_dim: Dimension of each input embedding.
        hidden_dim: Number of hidden units in the MLP.
    """

    def __init__(self, p: int, embed_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Long tensor of shape (batch_size, 2), columns are (a, b).

        Returns:
            Logits of shape (batch_size, p).
        """
        a = self.embed_a(x[:, 0])  # (batch, embed_dim)
        b = self.embed_b(x[:, 1])  # (batch, embed_dim)
        h = torch.cat([a, b], dim=1)  # (batch, 2 * embed_dim)
        h = self.relu(self.fc1(h))  # (batch, hidden_dim)
        return self.fc2(h)  # (batch, p)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
