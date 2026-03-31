# tests/test_sweep.py
"""Tests for the model size sweep."""

from src.sweep import run_sweep


def test_sweep_small():
    """Run a minimal sweep and check structure."""
    results = run_sweep(
        hidden_dims=[5, 10],
        n_train=30,
        n_test=10,
        d=5,
        n_classes=3,
        max_epochs=100,
    )

    assert "metadata" in results
    assert "results" in results

    # 2 hidden dims x 2 label types = 4 runs
    assert len(results["results"]) == 4

    for r in results["results"]:
        assert "label_type" in r
        assert "hidden_dim" in r
        assert "n_params" in r
        assert "train_acc" in r
        assert "test_acc" in r
        assert r["label_type"] in ("random", "structured")
        assert 0.0 <= r["train_acc"] <= 1.0
        assert 0.0 <= r["test_acc"] <= 1.0


def test_sweep_larger_model_higher_acc():
    """Larger models should generally have higher training accuracy."""
    results = run_sweep(
        hidden_dims=[5, 80],
        n_train=30,
        n_test=10,
        d=5,
        n_classes=3,
        max_epochs=500,
    )

    random_results = [r for r in results["results"] if r["label_type"] == "random"]
    small = [r for r in random_results if r["hidden_dim"] == 5][0]
    large = [r for r in random_results if r["hidden_dim"] == 80][0]

    # Larger model should have >= accuracy (may not always hold for very small epochs
    # but with 500 epochs it should)
    assert large["train_acc"] >= small["train_acc"] - 0.1, (
        f"Larger model ({large['train_acc']:.3f}) should have >= accuracy "
        f"than smaller ({small['train_acc']:.3f})"
    )
