"""Tests for the training loop."""

from src.trainer import train_run


def test_train_run_short():
    """Test training for a few epochs."""
    result = train_run(
        task_name="regression",
        hidden_size=32,
        n_epochs=10,
        seed=42,
    )
    assert result["task"] == "regression"
    assert result["hidden_size"] == 32
    assert len(result["epochs"]) == 10
    assert len(result["losses"]) == 10
    assert result["n_params"] > 0
    # Loss should be finite
    assert all(0 <= loss < 1e6 for loss in result["losses"])


def test_train_run_classification():
    result = train_run(
        task_name="classification",
        hidden_size=32,
        n_epochs=10,
        seed=42,
    )
    assert result["task"] == "classification"
    assert len(result["losses"]) == 10


def test_train_run_modular_addition():
    result = train_run(
        task_name="mod_add",
        hidden_size=32,
        n_epochs=10,
        seed=42,
    )
    assert result["task"] == "mod_add"
    assert len(result["losses"]) == 10
    # Initial loss for 97-class problem should be around -ln(1/97) ~ 4.57
    assert result["losses"][0] > 2.0


def test_train_run_reproducible():
    r1 = train_run("regression", hidden_size=32, n_epochs=5, seed=42)
    r2 = train_run("regression", hidden_size=32, n_epochs=5, seed=42)
    assert r1["losses"] == r2["losses"]


def test_loss_decreases():
    """Loss should generally decrease over training."""
    result = train_run("regression", hidden_size=64, n_epochs=50, seed=42)
    # Compare first few epochs avg to last few epochs avg
    early = sum(result["losses"][:5]) / 5
    late = sum(result["losses"][-5:]) / 5
    assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"
