"""Tests for synthetic data generation."""

import numpy as np
import pytest

from src.data import generate_gaussian_clusters, make_dataloaders


class TestGenerateGaussianClusters:
    """Tests for generate_gaussian_clusters."""

    def test_output_shapes(self):
        X, y = generate_gaussian_clusters(n_samples=100, n_features=5, n_classes=3)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_correct_labels(self):
        X, y = generate_gaussian_clusters(n_samples=100, n_features=5, n_classes=3)
        unique_labels = set(y.tolist())
        assert unique_labels == {0, 1, 2}

    def test_reproducibility(self):
        X1, y1 = generate_gaussian_clusters(seed=42)
        X2, y2 = generate_gaussian_clusters(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_gaussian_clusters(seed=42)
        X2, _ = generate_gaussian_clusters(seed=123)
        assert not np.allclose(X1, X2)

    def test_default_parameters(self):
        X, y = generate_gaussian_clusters()
        assert X.shape == (500, 10)
        assert len(set(y.tolist())) == 5


class TestMakeDataloaders:
    """Tests for make_dataloaders."""

    def test_returns_tuple(self):
        result = make_dataloaders(n_samples=100, n_features=5, n_classes=3)
        assert len(result) == 4  # train_loader, test_loader, n_train, n_test

    def test_correct_split(self):
        _, _, n_train, n_test = make_dataloaders(
            n_samples=100, test_fraction=0.2
        )
        assert n_train == 80
        assert n_test == 20

    def test_batch_shapes(self):
        train_loader, test_loader, _, _ = make_dataloaders(
            n_samples=100, n_features=5, n_classes=3, batch_size=32,
        )
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.shape[1] == 5
        assert len(y_batch.shape) == 1

    def test_normalized(self):
        """Data should be approximately normalized after processing."""
        train_loader, _, _, _ = make_dataloaders(n_samples=500)
        all_X = []
        for X_batch, _ in train_loader:
            all_X.append(X_batch.numpy())
        X = np.concatenate(all_X, axis=0)
        # Mean should be close to 0, std close to 1 (approximately)
        assert abs(X.mean()) < 0.5
        assert abs(X.std() - 1.0) < 0.5

    def test_normalization_uses_training_split_statistics(self, monkeypatch):
        """Normalization should be fit on the training split only."""
        X = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 10.0],
        ])
        y = np.array([0, 0, 1, 1], dtype=np.int64)

        def fake_generate_gaussian_clusters(**kwargs):
            return X.copy(), y.copy()

        monkeypatch.setattr("src.data.generate_gaussian_clusters", fake_generate_gaussian_clusters)

        train_loader, test_loader, _, _ = make_dataloaders(
            n_samples=4,
            n_features=2,
            n_classes=2,
            test_fraction=0.5,
            batch_size=2,
        )

        X_train, _ = next(iter(train_loader))
        X_test, _ = next(iter(test_loader))

        assert abs(X_train.numpy().mean()) < 1e-6
        assert X_test.numpy().mean() > 1.0
