"""Tests for data generation modules."""

import torch
from src.data import make_modular_addition_dataset, make_regression_dataset, MODULUS


class TestModularAddition:
    def test_shapes(self):
        d = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        total = MODULUS * MODULUS
        n_train = int(total * 0.7)
        n_test = total - n_train

        assert d["x_train"].shape == (n_train, 2 * MODULUS)
        assert d["y_train"].shape == (n_train,)
        assert d["x_test"].shape == (n_test, 2 * MODULUS)
        assert d["y_test"].shape == (n_test,)

    def test_onehot_encoding(self):
        d = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        # Each row should have exactly 2 ones (one per input number)
        row_sums = d["x_train"].sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0))

    def test_labels_in_range(self):
        d = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        assert d["y_train"].min() >= 0
        assert d["y_train"].max() < MODULUS
        assert d["y_test"].min() >= 0
        assert d["y_test"].max() < MODULUS

    def test_no_overlap(self):
        """Train and test sets should not share any input pairs."""
        d = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        train_set = set(tuple(row.tolist()) for row in d["x_train"])
        test_set = set(tuple(row.tolist()) for row in d["x_test"])
        assert len(train_set & test_set) == 0

    def test_reproducibility(self):
        d1 = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        d2 = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        assert torch.equal(d1["x_train"], d2["x_train"])
        assert torch.equal(d1["y_train"], d2["y_train"])

    def test_metadata(self):
        d = make_modular_addition_dataset(modulus=MODULUS, frac=0.7, seed=42)
        assert d["input_dim"] == 2 * MODULUS
        assert d["output_dim"] == MODULUS
        assert d["task_name"] == "modular_addition"


class TestRegression:
    def test_shapes(self):
        d = make_regression_dataset(n_samples=100, frac=0.8, seed=42)
        assert d["x_train"].shape == (80, 1)
        assert d["y_train"].shape == (80, 1)
        assert d["x_test"].shape == (20, 1)
        assert d["y_test"].shape == (20, 1)

    def test_metadata(self):
        d = make_regression_dataset(n_samples=100, frac=0.8, seed=42)
        assert d["input_dim"] == 1
        assert d["output_dim"] == 1
        assert d["task_name"] == "regression"

    def test_reproducibility(self):
        d1 = make_regression_dataset(n_samples=100, frac=0.8, seed=42)
        d2 = make_regression_dataset(n_samples=100, frac=0.8, seed=42)
        assert torch.equal(d1["x_train"], d2["x_train"])
        assert torch.equal(d1["y_train"], d2["y_train"])
