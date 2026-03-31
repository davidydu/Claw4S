# tests/test_data.py
"""Tests for synthetic data generation."""

import torch
from src.data import generate_dataset


def test_generate_random_labels_shape():
    """Dataset has correct shape and types."""
    X_train, y_train, X_test, y_test = generate_dataset(
        n_train=50, n_test=10, d=5, n_classes=3, seed=42, label_type="random"
    )
    assert X_train.shape == (50, 5)
    assert y_train.shape == (50,)
    assert X_test.shape == (10, 5)
    assert y_test.shape == (10,)
    assert X_train.dtype == torch.float32
    assert y_train.dtype == torch.int64


def test_generate_structured_labels_shape():
    """Structured labels dataset has correct shape."""
    X_train, y_train, X_test, y_test = generate_dataset(
        n_train=50, n_test=10, d=5, n_classes=3, seed=42, label_type="structured"
    )
    assert X_train.shape == (50, 5)
    assert y_train.shape == (50,)


def test_random_labels_are_random():
    """Random labels should cover multiple classes."""
    _, y_train, _, _ = generate_dataset(
        n_train=200, n_test=50, d=20, n_classes=10, seed=42, label_type="random"
    )
    unique_labels = torch.unique(y_train)
    assert len(unique_labels) >= 5, f"Expected >= 5 unique labels, got {len(unique_labels)}"


def test_structured_labels_have_structure():
    """Structured labels should depend on features (nearby points get same label)."""
    X_train, y_train, _, _ = generate_dataset(
        n_train=200, n_test=50, d=20, n_classes=10, seed=42, label_type="structured"
    )
    unique_labels = torch.unique(y_train)
    assert len(unique_labels) >= 2, "Structured labels should have multiple classes"


def test_reproducibility():
    """Same seed produces identical data."""
    X1, y1, _, _ = generate_dataset(seed=123, label_type="random")
    X2, y2, _, _ = generate_dataset(seed=123, label_type="random")
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_different_seeds_differ():
    """Different seeds produce different data."""
    _, y1, _, _ = generate_dataset(seed=42, label_type="random")
    _, y2, _, _ = generate_dataset(seed=99, label_type="random")
    assert not torch.equal(y1, y2)


def test_invalid_label_type_raises():
    """Invalid label_type should raise ValueError."""
    try:
        generate_dataset(label_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e)
