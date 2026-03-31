"""Tests for data generation and label noise injection."""

import numpy as np
import pytest

from src.data import make_gaussian_clusters, inject_label_noise, build_datasets


class TestMakeGaussianClusters:
    def test_output_shapes(self):
        X, y = make_gaussian_clusters(n_samples=100, n_features=5, n_classes=3)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_label_range(self):
        X, y = make_gaussian_clusters(n_samples=200, n_features=10, n_classes=5)
        assert set(np.unique(y)) == {0, 1, 2, 3, 4}

    def test_deterministic(self):
        X1, y1 = make_gaussian_clusters(seed=42)
        X2, y2 = make_gaussian_clusters(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, y1 = make_gaussian_clusters(seed=42)
        X2, y2 = make_gaussian_clusters(seed=99)
        assert not np.array_equal(X1, X2)


class TestInjectLabelNoise:
    def test_zero_noise_unchanged(self):
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_noisy = inject_label_noise(y, 0.0, n_classes=5)
        np.testing.assert_array_equal(y, y_noisy)

    def test_noise_flips_labels(self):
        y = np.zeros(100, dtype=np.int64)
        y_noisy = inject_label_noise(y, 0.5, n_classes=5, seed=42)
        n_flipped = (y != y_noisy).sum()
        # Should flip approximately 50 out of 100
        assert 40 <= n_flipped <= 60

    def test_flipped_labels_are_different_class(self):
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        y_noisy = inject_label_noise(y, 0.3, n_classes=5, seed=42)
        flipped = y != y_noisy
        # Where flipped, the new label must differ from original
        for i in range(len(y)):
            if flipped[i]:
                assert y_noisy[i] != y[i]

    def test_invalid_noise_frac(self):
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="noise_frac must be in"):
            inject_label_noise(y, -0.1, n_classes=3)
        with pytest.raises(ValueError, match="noise_frac must be in"):
            inject_label_noise(y, 1.5, n_classes=3)

    def test_deterministic(self):
        y = np.arange(50, dtype=np.int64) % 5
        y1 = inject_label_noise(y, 0.3, n_classes=5, seed=42)
        y2 = inject_label_noise(y, 0.3, n_classes=5, seed=42)
        np.testing.assert_array_equal(y1, y2)


class TestBuildDatasets:
    def test_train_test_split_sizes(self):
        train_ds, test_ds, y_clean = build_datasets(
            n_samples=100, train_frac=0.7
        )
        assert len(train_ds) == 70
        assert len(test_ds) == 30
        assert len(y_clean) == 70

    def test_noisy_labels_differ_from_clean(self):
        train_ds, _, y_clean = build_datasets(
            n_samples=200, noise_frac=0.5, seed=42
        )
        y_noisy = train_ds.tensors[1].numpy()
        n_flipped = (y_noisy != y_clean).sum()
        assert n_flipped > 0, "Expected some labels to be flipped at 50% noise"

    def test_test_set_has_clean_labels(self):
        """Test set labels should be identical across noise levels."""
        _, test_clean, _ = build_datasets(noise_frac=0.0, seed=42)
        _, test_noisy, _ = build_datasets(noise_frac=0.5, seed=42)
        np.testing.assert_array_equal(
            test_clean.tensors[1].numpy(),
            test_noisy.tensors[1].numpy(),
        )
