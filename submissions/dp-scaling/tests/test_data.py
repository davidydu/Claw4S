"""Tests for synthetic data generation."""

import numpy as np
import torch

from src.data import generate_gaussian_clusters, make_dataloaders


class TestGenerateGaussianClusters:
    """Tests for generate_gaussian_clusters."""

    def test_output_shapes(self):
        X, y = generate_gaussian_clusters(n_samples=100, n_features=5, n_classes=3)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_correct_classes(self):
        X, y = generate_gaussian_clusters(n_samples=100, n_features=5, n_classes=3)
        unique = set(y.tolist())
        assert unique == {0, 1, 2}

    def test_reproducibility(self):
        X1, y1 = generate_gaussian_clusters(seed=42)
        X2, y2 = generate_gaussian_clusters(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_gaussian_clusters(seed=42)
        X2, _ = generate_gaussian_clusters(seed=99)
        assert not np.allclose(X1, X2)

    def test_default_params(self):
        X, y = generate_gaussian_clusters()
        assert X.shape == (500, 10)
        assert y.shape == (500,)
        assert len(set(y.tolist())) == 5


class TestMakeDataloaders:
    """Tests for make_dataloaders."""

    def test_returns_correct_types(self):
        train_loader, test_loader, n_feat, n_cls = make_dataloaders(
            n_samples=100, n_features=5, n_classes=3
        )
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)
        assert n_feat == 5
        assert n_cls == 3

    def test_train_test_split_size(self):
        train_loader, test_loader, _, _ = make_dataloaders(
            n_samples=100, train_fraction=0.8
        )
        train_samples = sum(len(batch[0]) for batch in train_loader)
        test_samples = sum(len(batch[0]) for batch in test_loader)
        assert train_samples == 80
        assert test_samples == 20

    def test_data_types(self):
        train_loader, _, _, _ = make_dataloaders(n_samples=50)
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.long
