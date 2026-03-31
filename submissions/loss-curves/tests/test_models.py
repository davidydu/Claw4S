"""Tests for model construction and forward pass."""

import torch
from src.models import EmbeddingMLP, ContinuousMLP, build_model
from src.tasks import TASK_REGISTRY


def test_embedding_mlp_forward():
    model = EmbeddingMLP(
        vocab_size=10, embed_dim=8, n_inputs=2, hidden_size=32, n_classes=10
    )
    x = torch.randint(0, 10, (4, 2))
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_continuous_mlp_regression():
    model = ContinuousMLP(input_dim=20, hidden_size=32, output_dim=1)
    x = torch.randn(4, 20)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_continuous_mlp_classification():
    model = ContinuousMLP(input_dim=20, hidden_size=64, output_dim=5)
    x = torch.randn(4, 20)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"


def test_build_model_all_tasks():
    for task_name, config in TASK_REGISTRY.items():
        model = build_model(config, hidden_size=32)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, f"Model for {task_name} has 0 parameters"


def test_param_count_increases_with_hidden_size():
    config = TASK_REGISTRY["regression"]
    sizes = [32, 64, 128]
    param_counts = []
    for hs in sizes:
        model = build_model(config, hs)
        param_counts.append(sum(p.numel() for p in model.parameters()))
    assert param_counts[0] < param_counts[1] < param_counts[2]
