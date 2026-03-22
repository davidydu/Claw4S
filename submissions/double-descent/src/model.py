"""Models for double descent experiments.

Provides two model types:
1. RandomFeaturesModel: Random ReLU features with least-squares fit.
   This gives the cleanest double descent (Belkin et al. 2019).
2. MLP: Trainable 2-layer neural network for comparison.
"""

import torch
import torch.nn as nn


class RandomFeaturesModel:
    """Random ReLU features model with minimum-norm least-squares fit.

    The first layer (random projection + ReLU) is fixed.
    The second layer is fit via pseudo-inverse: beta = pinv(Phi) @ y.

    This setup produces a very clean double descent because:
    - At the interpolation threshold (p = n_train), the system is exactly
      determined and forced to fit noise exactly.
    - Below threshold: underdetermined, classical bias-variance tradeoff.
    - Above threshold: minimum-norm solution is smoother, generalizes better.

    Args:
        input_dim: Input feature dimension (d).
        n_features: Number of random features (p). This is the "width".
        seed: Random seed for the random projection.
    """

    def __init__(self, input_dim: int, n_features: int, seed: int = 42):
        self.input_dim = input_dim
        self.n_features = n_features
        self.seed = seed

        gen = torch.Generator()
        gen.manual_seed(seed)

        # Fixed random first layer
        self.W = torch.randn(input_dim, n_features, generator=gen)
        self.b = torch.randn(n_features, generator=gen)

        # Trainable second layer (set by fit())
        self.beta = None

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply random feature transformation: ReLU(X @ W + b)."""
        return torch.relu(X @ self.W + self.b)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the second layer via minimum-norm least squares."""
        Phi = self.transform(X)
        self.beta = torch.linalg.lstsq(Phi, y).solution

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict using fitted model."""
        if self.beta is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        Phi = self.transform(X)
        return Phi @ self.beta

    @property
    def n_params(self) -> int:
        """Number of trainable parameters (second layer only)."""
        return self.n_features  # beta has n_features entries


class MLP(nn.Module):
    """Two-layer MLP: Linear(d, h) -> ReLU -> Linear(h, 1).

    Used as a comparison to the random features model to show that
    double descent also occurs in trained neural networks.

    Args:
        input_dim: Number of input features (d).
        hidden_width: Number of hidden units (h).
    """

    def __init__(self, input_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc2(self.relu(self.fc1(x)))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_interpolation_threshold(n_train: int, input_dim: int) -> int:
    """Estimate hidden width at the interpolation threshold.

    For RandomFeaturesModel: threshold at p = n_train.
    For MLP: threshold at h*(d+2)+1 ~ n_train, so h ~ (n_train-1)/(d+2).

    Args:
        n_train: Number of training samples.
        input_dim: Input dimensionality.

    Returns:
        Estimated hidden width at interpolation threshold.
    """
    return n_train  # For random features, threshold is exactly n_train


def create_mlp(input_dim: int, hidden_width: int, seed: int = 42) -> MLP:
    """Create an MLP with deterministic initialization.

    Args:
        input_dim: Number of input features.
        hidden_width: Number of hidden units.
        seed: Random seed for weight initialization.

    Returns:
        Initialized MLP model on CPU.
    """
    torch.manual_seed(seed)
    model = MLP(input_dim, hidden_width)
    model.to(torch.device("cpu"))
    return model
