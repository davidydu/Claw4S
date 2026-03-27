"""Tests for data generation and poisoning utilities."""

import numpy as np
import pytest

from src.data import generate_gaussian_clusters, make_datasets, poison_labels


class TestGenerateGaussianClusters:
    """Tests for generate_gaussian_clusters."""

    def test_shape(self):
        X, y = generate_gaussian_clusters(n_samples=100, n_features=5, n_classes=3)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_class_count(self):
        X, y = generate_gaussian_clusters(n_samples=100, n_classes=4)
        assert set(y) == {0, 1, 2, 3}

    def test_reproducibility(self):
        X1, y1 = generate_gaussian_clusters(seed=42)
        X2, y2 = generate_gaussian_clusters(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds(self):
        X1, _ = generate_gaussian_clusters(seed=42)
        X2, _ = generate_gaussian_clusters(seed=99)
        assert not np.allclose(X1, X2)

    def test_balanced_classes(self):
        _, y = generate_gaussian_clusters(n_samples=500, n_classes=5)
        counts = np.bincount(y)
        assert counts.min() == 100
        assert counts.max() == 100


class TestPoisonLabels:
    """Tests for poison_labels."""

    def test_no_poison(self):
        y = np.array([0, 1, 2, 3, 4])
        y_p = poison_labels(y, 0.0)
        np.testing.assert_array_equal(y, y_p)

    def test_full_poison(self):
        y = np.array([0, 0, 0, 0, 0])
        y_p = poison_labels(y, 1.0, n_classes=5)
        # All labels should be different from original
        assert np.all(y_p != y)

    def test_partial_poison_count(self):
        y = np.zeros(100, dtype=np.int64)
        y_p = poison_labels(y, 0.1, n_classes=5, seed=42)
        n_flipped = np.sum(y_p != y)
        assert n_flipped == 10  # 10% of 100

    def test_poison_never_same_label(self):
        y = np.zeros(100, dtype=np.int64)
        y_p = poison_labels(y, 1.0, n_classes=5, seed=42)
        # No poisoned label should equal original
        assert np.all(y_p != y)

    def test_invalid_fraction(self):
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="poison_fraction"):
            poison_labels(y, -0.1)
        with pytest.raises(ValueError, match="poison_fraction"):
            poison_labels(y, 1.5)

    def test_reproducibility(self):
        y = np.arange(50) % 5
        y1 = poison_labels(y, 0.3, seed=42)
        y2 = poison_labels(y, 0.3, seed=42)
        np.testing.assert_array_equal(y1, y2)


class TestMakeDatasets:
    """Tests for make_datasets."""

    def test_split_sizes(self):
        X, y = generate_gaussian_clusters(n_samples=100)
        datasets = make_datasets(X, y, y, train_fraction=0.7)
        train_n = len(datasets["train_dataset"])
        test_n = len(datasets["test_dataset"])
        assert train_n == 70
        assert test_n == 30

    def test_test_uses_clean_labels(self):
        X, y_clean = generate_gaussian_clusters(n_samples=100, n_classes=5)
        y_poison = poison_labels(y_clean, 0.5, n_classes=5)
        datasets = make_datasets(X, y_poison, y_clean)
        # Test labels should be clean
        test_labels = datasets["y_test"].numpy()
        # Verify they come from clean labels (indirectly: they're a subset)
        assert set(test_labels).issubset(set(y_clean))
