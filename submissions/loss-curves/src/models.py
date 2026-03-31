"""MLP model definitions for training experiments."""

import torch
import torch.nn as nn


class EmbeddingMLP(nn.Module):
    """MLP with embedding layers for discrete-input tasks (modular arithmetic)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_inputs: int,
        hidden_size: int,
        n_classes: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(n_inputs * embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_inputs) of integer indices
        emb = self.embedding(x)  # (batch, n_inputs, embed_dim)
        emb = emb.view(emb.size(0), -1)  # (batch, n_inputs * embed_dim)
        return self.net(emb)


class ContinuousMLP(nn.Module):
    """MLP for continuous-input tasks (regression, classification)."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(task_config: dict, hidden_size: int) -> nn.Module:
    """Build the appropriate model for a task configuration."""
    if task_config["input_type"] == "embedding":
        return EmbeddingMLP(
            vocab_size=task_config["vocab_size"],
            embed_dim=task_config["embed_dim"],
            n_inputs=task_config["n_inputs"],
            hidden_size=hidden_size,
            n_classes=task_config["n_classes"],
        )
    elif task_config["task_type"] == "regression":
        return ContinuousMLP(
            input_dim=task_config["input_dim"],
            hidden_size=hidden_size,
            output_dim=1,
        )
    elif task_config["task_type"] == "classification":
        return ContinuousMLP(
            input_dim=task_config["input_dim"],
            hidden_size=hidden_size,
            output_dim=task_config["n_classes"],
        )
    else:
        raise ValueError(f"Unknown task type: {task_config['task_type']}")
