"""Tests for task data generation."""

import torch

from src.tasks import make_sparse_parity_data, make_regression_data


class TestSparseParity:
    """Tests for sparse parity dataset."""

    def test_shapes(self):
        """Data shapes should match requested sizes."""
        data = make_sparse_parity_data(
            n_bits=20, k_relevant=5, n_train=100, n_test=50, seed=42
        )
        assert data["x_train"].shape == (100, 20)
        assert data["y_train"].shape == (100,)
        assert data["x_test"].shape == (50, 20)
        assert data["y_test"].shape == (50,)

    def test_inputs_are_binary(self):
        """Inputs should be in {-1, +1}."""
        data = make_sparse_parity_data(
            n_bits=10, k_relevant=3, n_train=100, n_test=50, seed=42
        )
        unique_vals = data["x_train"].unique().sort()[0]
        assert torch.allclose(unique_vals, torch.tensor([-1.0, 1.0]))

    def test_labels_are_binary(self):
        """Labels should be in {0, 1}."""
        data = make_sparse_parity_data(
            n_bits=10, k_relevant=3, n_train=200, n_test=100, seed=42
        )
        assert set(data["y_train"].unique().tolist()) <= {0, 1}
        assert set(data["y_test"].unique().tolist()) <= {0, 1}

    def test_parity_correctness(self):
        """Labels should equal parity of first k bits."""
        data = make_sparse_parity_data(
            n_bits=10, k_relevant=3, n_train=50, n_test=20, seed=42
        )
        x = data["x_train"]
        y = data["y_train"]
        # Compute parity manually: product of first 3 columns
        parity = x[:, 0] * x[:, 1] * x[:, 2]
        expected_y = ((parity + 1) / 2).long()
        assert torch.equal(y, expected_y)

    def test_reproducibility(self):
        """Same seed should produce same data."""
        d1 = make_sparse_parity_data(seed=42)
        d2 = make_sparse_parity_data(seed=42)
        assert torch.equal(d1["x_train"], d2["x_train"])
        assert torch.equal(d1["y_train"], d2["y_train"])

    def test_metadata(self):
        """Metadata fields should be correct."""
        data = make_sparse_parity_data(
            n_bits=20, k_relevant=5, seed=42
        )
        assert data["input_dim"] == 20
        assert data["output_dim"] == 2
        assert data["task_type"] == "classification"
        assert data["task_name"] == "sparse_parity"
        assert data["k_relevant"] == 5

    def test_roughly_balanced(self):
        """Labels should be roughly 50/50 for large samples."""
        data = make_sparse_parity_data(
            n_bits=10, k_relevant=3, n_train=2000, n_test=1000, seed=42
        )
        frac_positive = data["y_train"].float().mean().item()
        assert 0.4 < frac_positive < 0.6, f"Class balance: {frac_positive}"


class TestRegression:
    """Tests for regression dataset."""

    def test_shapes(self):
        """Data shapes should match requested sizes."""
        data = make_regression_data(
            n_train=100, n_test=50, input_dim=8, seed=42
        )
        assert data["x_train"].shape == (100, 8)
        assert data["y_train"].shape == (100,)
        assert data["x_test"].shape == (50, 8)
        assert data["y_test"].shape == (50,)

    def test_target_is_finite(self):
        """All target values should be finite."""
        data = make_regression_data(seed=42)
        assert torch.isfinite(data["y_train"]).all()
        assert torch.isfinite(data["y_test"]).all()

    def test_reproducibility(self):
        """Same seed should produce same data."""
        d1 = make_regression_data(seed=42)
        d2 = make_regression_data(seed=42)
        assert torch.equal(d1["x_train"], d2["x_train"])
        assert torch.equal(d1["y_train"], d2["y_train"])

    def test_metadata(self):
        """Metadata fields should be correct."""
        data = make_regression_data(seed=42)
        assert data["input_dim"] == 8
        assert data["output_dim"] == 1
        assert data["task_type"] == "regression"
        assert data["task_name"] == "smooth_regression"
