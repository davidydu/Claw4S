"""Tests for task data generation."""

import torch
from src.tasks import (
    make_modular_addition_data,
    make_modular_multiplication_data,
    make_regression_data,
    make_classification_data,
    TASK_REGISTRY,
)


def test_modular_addition_shapes():
    X, y = make_modular_addition_data(p=7)
    assert X.shape == (49, 2), f"Expected (49, 2), got {X.shape}"
    assert y.shape == (49,), f"Expected (49,), got {y.shape}"
    assert y.min() >= 0
    assert y.max() < 7


def test_modular_addition_correctness():
    X, y = make_modular_addition_data(p=5)
    for i in range(len(X)):
        a, b = X[i].tolist()
        assert y[i].item() == (a + b) % 5


def test_modular_multiplication_shapes():
    X, y = make_modular_multiplication_data(p=7)
    assert X.shape == (49, 2)
    assert y.shape == (49,)
    assert y.max() < 7


def test_modular_multiplication_correctness():
    X, y = make_modular_multiplication_data(p=5)
    for i in range(len(X)):
        a, b = X[i].tolist()
        assert y[i].item() == (a * b) % 5


def test_regression_shapes():
    X, y = make_regression_data(n_samples=100, input_dim=10)
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


def test_classification_shapes():
    X, y = make_classification_data(n_samples=100, input_dim=10, n_classes=3)
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert y.min() >= 0
    assert y.max() < 3


def test_reproducibility():
    X1, y1 = make_modular_addition_data(seed=123)
    X2, y2 = make_modular_addition_data(seed=123)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_task_registry_completeness():
    expected_tasks = {"mod_add", "mod_mul", "regression", "classification"}
    assert set(TASK_REGISTRY.keys()) == expected_tasks
    for name, config in TASK_REGISTRY.items():
        assert "make_data" in config
        assert "task_type" in config
