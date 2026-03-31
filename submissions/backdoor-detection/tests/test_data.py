"""Tests for synthetic data generation and backdoor injection."""

import numpy as np
import pytest

from src.data import generate_clean_data, inject_backdoor, make_datasets


class TestGenerateCleanData:
    """Tests for generate_clean_data function."""

    def test_output_shapes(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        assert X.shape == (100, 10)
        assert y.shape == (100,)

    def test_correct_classes(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        unique = set(y)
        assert unique == {0, 1, 2, 3, 4}

    def test_deterministic(self):
        X1, y1 = generate_clean_data(seed=42)
        X2, y2 = generate_clean_data(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds(self):
        X1, _ = generate_clean_data(seed=42)
        X2, _ = generate_clean_data(seed=99)
        assert not np.array_equal(X1, X2)

    def test_float32(self):
        X, _ = generate_clean_data()
        assert X.dtype == np.float32


class TestInjectBackdoor:
    """Tests for inject_backdoor function."""

    def test_poison_count(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        _, _, mask = inject_backdoor(X, y, poison_fraction=0.1)
        assert mask.sum() == 10

    def test_trigger_applied(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        X_p, _, mask = inject_backdoor(X, y, trigger_strength=10.0,
                                        trigger_features=(0, 1, 2))
        poisoned_features = X_p[mask, :3]
        expected_val = 10.0
        np.testing.assert_array_almost_equal(
            poisoned_features, np.full_like(poisoned_features, expected_val)
        )

    def test_labels_changed(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        _, y_p, mask = inject_backdoor(X, y, target_class=0)
        assert all(y_p[mask] == 0)

    def test_clean_samples_unchanged(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        X_p, y_p, mask = inject_backdoor(X, y)
        clean = ~mask
        np.testing.assert_array_equal(X_p[clean], X[clean])

    def test_deterministic(self):
        X, y = generate_clean_data(seed=42)
        _, _, m1 = inject_backdoor(X, y, seed=42)
        _, _, m2 = inject_backdoor(X, y, seed=42)
        np.testing.assert_array_equal(m1, m2)


class TestMakeDatasets:
    """Tests for make_datasets function."""

    def test_tensor_dataset(self):
        X, y = generate_clean_data(n_samples=50, n_features=10, n_classes=5)
        ds = make_datasets(X, y)
        assert len(ds) == 50
        x_t, y_t = ds[0]
        assert x_t.shape == (10,)
        assert y_t.ndim == 0
