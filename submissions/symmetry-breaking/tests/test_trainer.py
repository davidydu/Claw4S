"""Tests for the training loop (fast, reduced epochs)."""

from src.trainer import train_single_run


def test_train_single_run_structure():
    """A short training run should return all expected fields."""
    result = train_single_run(
        hidden_dim=8,
        epsilon=0.01,
        num_epochs=100,
        batch_size=512,
        lr=0.1,
        log_interval=50,
        seed=42,
        modulus=17,  # Small modulus for fast test
    )

    required_keys = [
        "hidden_dim", "epsilon", "seed", "epochs_logged",
        "symmetry_values", "loss_values", "final_test_acc",
        "final_train_acc", "initial_symmetry", "final_symmetry",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_train_symmetry_values_logged():
    """Symmetry should be logged at each log_interval."""
    result = train_single_run(
        hidden_dim=8,
        epsilon=0.01,
        num_epochs=200,
        batch_size=512,
        lr=0.1,
        log_interval=100,
        seed=42,
        modulus=17,
    )
    # Should have: epoch 0, 100, 200 = 3 values
    assert len(result["symmetry_values"]) == 3, (
        f"Expected 3 symmetry values, got {len(result['symmetry_values'])}"
    )
    assert result["epochs_logged"] == [0, 100, 200]


def test_train_reproducibility():
    """Same parameters should produce identical results."""
    r1 = train_single_run(
        hidden_dim=8, epsilon=0.01, num_epochs=50,
        batch_size=512, lr=0.1, log_interval=50, seed=42, modulus=17,
    )
    r2 = train_single_run(
        hidden_dim=8, epsilon=0.01, num_epochs=50,
        batch_size=512, lr=0.1, log_interval=50, seed=42, modulus=17,
    )
    assert r1["final_test_acc"] == r2["final_test_acc"]
    assert r1["symmetry_values"] == r2["symmetry_values"]


def test_symmetric_init_high_initial_symmetry():
    """With epsilon=0, initial symmetry should be ~1.0."""
    result = train_single_run(
        hidden_dim=8, epsilon=0.0, num_epochs=50,
        batch_size=512, lr=0.1, log_interval=50, seed=42, modulus=17,
    )
    assert result["initial_symmetry"] > 0.99, (
        f"Expected initial symmetry ~1.0, got {result['initial_symmetry']}"
    )
