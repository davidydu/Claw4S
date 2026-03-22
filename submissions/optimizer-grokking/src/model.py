"""Two-layer MLP for modular arithmetic.

Architecture: Embedding(p, embed_dim) x2 -> concat -> Linear -> ReLU -> Linear -> p classes.
Follows the standard grokking setup from Power et al. (2022).
"""

import torch
import torch.nn as nn

from data import PRIME


class ModularMLP(nn.Module):
    """MLP for learning (a + b) mod p.

    Each input (a, b) is embedded separately, embeddings are concatenated,
    then passed through a 2-layer MLP with ReLU activation.
    """

    def __init__(
        self,
        p: int = PRIME,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.p = p
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            a: Integer tensor of shape (batch,), values in [0, p).
            b: Integer tensor of shape (batch,), values in [0, p).

        Returns:
            Logits of shape (batch, p).
        """
        ea = self.embed_a(a)
        eb = self.embed_b(b)
        x = torch.cat([ea, eb], dim=-1)
        return self.mlp(x)
