"""Tests for data generation modules."""

from src.data import generate_modular_addition, generate_regression


def test_modular_addition_shapes():
    """Verify modular addition dataset has correct shapes."""
    data = generate_modular_addition(p=7, seed=42)
    n_total = 7 * 7  # 49 pairs
    n_train = int(n_total * 0.7)
    n_test = n_total - n_train

    assert data["X_train"].shape == (n_train, 14)  # 2*p = 14
    assert data["y_train"].shape == (n_train,)
    assert data["X_test"].shape == (n_test, 14)
    assert data["y_test"].shape == (n_test,)
    assert data["input_dim"] == 14
    assert data["output_dim"] == 7


def test_modular_addition_labels():
    """Verify labels are in correct range [0, p)."""
    data = generate_modular_addition(p=7, seed=42)
    assert data["y_train"].min() >= 0
    assert data["y_train"].max() < 7
    assert data["y_test"].min() >= 0
    assert data["y_test"].max() < 7


def test_modular_addition_one_hot():
    """Verify one-hot encoding sums to 2 per row."""
    data = generate_modular_addition(p=7, seed=42)
    row_sums = data["X_train"].sum(dim=1)
    assert (row_sums == 2.0).all()


def test_regression_shapes():
    """Verify regression dataset has correct shapes."""
    data = generate_regression(n_samples=100, seed=42)
    n_train = int(100 * 0.7)
    n_test = 100 - n_train

    assert data["X_train"].shape == (n_train, 3)
    assert data["y_train"].shape == (n_train, 1)
    assert data["X_test"].shape == (n_test, 3)
    assert data["y_test"].shape == (n_test, 1)


def test_regression_deterministic():
    """Verify regression is deterministic with same seed."""
    d1 = generate_regression(n_samples=50, seed=42)
    d2 = generate_regression(n_samples=50, seed=42)
    assert (d1["X_train"] == d2["X_train"]).all()
    assert (d1["y_train"] == d2["y_train"]).all()
