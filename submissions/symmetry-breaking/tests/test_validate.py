"""Tests for scientific sanity checks in validate.py."""

from validate import validate_results_content


def _make_run(hidden_dim: int, epsilon: float, final_test_acc: float) -> dict:
    return {
        "hidden_dim": hidden_dim,
        "epsilon": epsilon,
        "seed": 42,
        "epochs_logged": [0, 100, 200],
        "symmetry_values": [1.0, 0.8, 0.3],
        "loss_values": [2.0, 1.8],
        "train_acc_values": [0.2, 0.5],
        "final_test_acc": final_test_acc,
        "final_train_acc": 0.6,
        "initial_symmetry": 1.0,
        "final_symmetry": 0.3,
        "num_epochs": 200,
        "batch_size": 256,
        "lr": 0.1,
        "modulus": 97,
    }


def test_validate_results_content_flags_weak_high_epsilon_signal():
    results = [
        _make_run(16, 0.0, 0.03),
        _make_run(16, 0.1, 0.04),
        _make_run(32, 0.0, 0.02),
        _make_run(32, 0.1, 0.03),
    ]
    errors, _ = validate_results_content(
        results=results,
        expected_hidden_dims=[16, 32],
        expected_epsilons=[0.0, 0.1],
    )

    assert any("highest epsilon" in e for e in errors)


def test_validate_results_content_accepts_strong_high_epsilon_signal():
    results = [
        _make_run(16, 0.0, 0.03),
        _make_run(16, 0.1, 0.22),
        _make_run(32, 0.0, 0.02),
        _make_run(32, 0.1, 0.31),
    ]
    errors, diagnostics = validate_results_content(
        results=results,
        expected_hidden_dims=[16, 32],
        expected_epsilons=[0.0, 0.1],
    )

    assert errors == []
    assert diagnostics["best_test_acc_at_max_epsilon"] == 0.31
