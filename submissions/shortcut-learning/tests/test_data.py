"""Tests for synthetic data generation."""

import numpy as np
import torch
from src.data import generate_dataset


def test_dataset_shapes():
    """Verify that datasets have correct shapes."""
    data = generate_dataset(n_train=100, n_test=50, n_genuine=10, seed=42)

    train_X, train_y = data["train_dataset"].tensors
    assert train_X.shape == (100, 11), f"Expected (100, 11), got {train_X.shape}"
    assert train_y.shape == (100,), f"Expected (100,), got {train_y.shape}"

    test_w_X, test_w_y = data["test_with_shortcut"].tensors
    assert test_w_X.shape == (50, 11)
    assert test_w_y.shape == (50,)

    test_wo_X, test_wo_y = data["test_without_shortcut"].tensors
    assert test_wo_X.shape == (50, 11)
    assert test_wo_y.shape == (50,)


def test_shortcut_correlation_in_training():
    """In training data, shortcut feature (last col) should equal label."""
    data = generate_dataset(n_train=200, n_test=100, n_genuine=10, seed=42)
    train_X, train_y = data["train_dataset"].tensors
    shortcut_col = train_X[:, -1]
    assert torch.all(shortcut_col == train_y.float()), \
        "Shortcut feature should perfectly match labels in training set"


def test_shortcut_randomized_in_test():
    """In test_without_shortcut, shortcut should NOT perfectly match labels."""
    data = generate_dataset(n_train=200, n_test=500, n_genuine=10, seed=42)
    test_X, test_y = data["test_without_shortcut"].tensors
    shortcut_col = test_X[:, -1]
    match_rate = (shortcut_col == test_y.float()).float().mean().item()
    # Random should be close to 0.5, definitely not 1.0
    assert match_rate < 0.7, \
        f"Shortcut should be randomized in test, but match rate is {match_rate:.3f}"


def test_reproducibility():
    """Same seed should produce identical data."""
    d1 = generate_dataset(n_train=50, n_test=20, seed=42)
    d2 = generate_dataset(n_train=50, n_test=20, seed=42)
    X1, y1 = d1["train_dataset"].tensors
    X2, y2 = d2["train_dataset"].tensors
    assert torch.equal(X1, X2), "Same seed should produce identical features"
    assert torch.equal(y1, y2), "Same seed should produce identical labels"


def test_different_seeds_differ():
    """Different seeds should produce different data."""
    d1 = generate_dataset(n_train=50, n_test=20, seed=42)
    d2 = generate_dataset(n_train=50, n_test=20, seed=99)
    X1, _ = d1["train_dataset"].tensors
    X2, _ = d2["train_dataset"].tensors
    assert not torch.equal(X1, X2), "Different seeds should produce different data"


def test_metadata():
    """Metadata should contain all generation parameters."""
    data = generate_dataset(n_train=100, n_test=50, n_genuine=10, seed=42)
    meta = data["metadata"]
    assert meta["n_train"] == 100
    assert meta["n_test"] == 50
    assert meta["n_genuine"] == 10
    assert meta["n_total_features"] == 11
    assert meta["shortcut_index"] == 10
    assert meta["seed"] == 42
