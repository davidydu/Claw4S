"""Tests for synthetic data generators."""

import numpy as np
import torch

from src.data import make_circles, make_moons, make_dataloaders


class TestMakeCircles:
    def test_output_shape(self):
        X, y = make_circles(n_samples=100, seed=42)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_binary_labels(self):
        X, y = make_circles(n_samples=200, seed=42)
        unique_labels = set(y.tolist())
        assert unique_labels == {0, 1}

    def test_balanced_classes(self):
        X, y = make_circles(n_samples=200, seed=42)
        assert abs(np.sum(y == 0) - np.sum(y == 1)) <= 1

    def test_reproducibility(self):
        X1, y1 = make_circles(n_samples=100, seed=42)
        X2, y2 = make_circles(n_samples=100, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = make_circles(n_samples=100, seed=42)
        X2, _ = make_circles(n_samples=100, seed=99)
        assert not np.allclose(X1, X2)

    def test_dtype(self):
        X, y = make_circles(n_samples=50, seed=42)
        assert X.dtype == np.float32
        assert y.dtype == np.int64


class TestMakeMoons:
    def test_output_shape(self):
        X, y = make_moons(n_samples=100, seed=42)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_binary_labels(self):
        X, y = make_moons(n_samples=200, seed=42)
        unique_labels = set(y.tolist())
        assert unique_labels == {0, 1}

    def test_reproducibility(self):
        X1, y1 = make_moons(n_samples=100, seed=42)
        X2, y2 = make_moons(n_samples=100, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestMakeDataloaders:
    def test_returns_four_items(self):
        result = make_dataloaders(dataset="circles", n_samples=100, seed=42)
        assert len(result) == 4

    def test_test_size(self):
        _, _, X_test, y_test = make_dataloaders(
            dataset="circles", n_samples=100, test_fraction=0.2, seed=42)
        assert len(X_test) == 20
        assert len(y_test) == 20

    def test_tensor_types(self):
        _, _, X_test, y_test = make_dataloaders(
            dataset="circles", n_samples=100, seed=42)
        assert isinstance(X_test, torch.Tensor)
        assert isinstance(y_test, torch.Tensor)

    def test_invalid_dataset(self):
        try:
            make_dataloaders(dataset="invalid", n_samples=100, seed=42)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
