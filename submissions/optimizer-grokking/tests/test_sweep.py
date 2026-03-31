"""Tests for sweep orchestration and metadata."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sweep


def _fake_result(optimizer: str, lr: float, weight_decay: float, outcome: str = "failure") -> dict:
    """Build a minimal run result with required keys."""
    return {
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "history": [{
            "epoch": 1,
            "train_acc": 0.1,
            "test_acc": 0.1,
            "train_loss": 1.0,
            "test_loss": 1.0,
        }],
        "final_train_acc": 0.1,
        "final_test_acc": 0.1,
        "memorization_epoch": None,
        "generalization_epoch": None,
        "grokking_epoch": None,
        "outcome": outcome,
    }


def test_run_sweep_resumes_cached_runs(tmp_path, monkeypatch):
    """Existing runs should be reused when resume mode is enabled."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    save_path = results_dir / "sweep_results.json"

    cached = _fake_result("sgd", 0.1, 0.0)
    save_path.write_text(json.dumps({"metadata": {"total_seconds": 0.0}, "runs": [cached]}))

    calls = []

    def fake_train_run(
        optimizer_name, lr, weight_decay, train_a, train_b, train_t,
        test_a, test_b, test_t, max_epochs, batch_size, p, seed, log_interval,
    ):
        calls.append((optimizer_name, lr, weight_decay))
        return _fake_result(optimizer_name, lr, weight_decay)

    monkeypatch.setattr(sweep, "train_run", fake_train_run)

    runs = sweep.run_sweep(
        optimizers=["sgd", "adam"],
        learning_rates=[0.1],
        weight_decays=[0.0],
        max_epochs=1,
        log_interval=1,
        results_dir=str(results_dir),
        resume=True,
    )

    assert calls == [("adam", 0.1, 0.0)]
    assert len(runs) == 2
    assert {(r["optimizer"], r["lr"], r["weight_decay"]) for r in runs} == {
        ("sgd", 0.1, 0.0),
        ("adam", 0.1, 0.0),
    }


def test_run_sweep_writes_provenance_metadata(tmp_path, monkeypatch):
    """Saved metadata should include environment provenance fields."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    def fake_train_run(
        optimizer_name, lr, weight_decay, train_a, train_b, train_t,
        test_a, test_b, test_t, max_epochs, batch_size, p, seed, log_interval,
    ):
        return _fake_result(optimizer_name, lr, weight_decay, outcome="grokking")

    monkeypatch.setattr(sweep, "train_run", fake_train_run)

    sweep.run_sweep(
        optimizers=["adamw"],
        learning_rates=[0.03],
        weight_decays=[0.01],
        max_epochs=5,
        log_interval=5,
        results_dir=str(results_dir),
        resume=False,
    )

    data = json.loads((results_dir / "sweep_results.json").read_text())
    meta = data["metadata"]

    assert meta["num_runs"] == 1
    assert meta["completed_runs"] == 1
    assert meta["train_examples"] + meta["test_examples"] == sweep.PRIME * sweep.PRIME
    assert meta["python_version"]
    assert meta["torch_version"]
    assert meta["numpy_version"]
    assert meta["platform"]
