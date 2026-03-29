"""Tests for experiment orchestration and reproducibility metadata."""

from src import experiment


def test_run_full_experiment_records_config_and_passes_overrides(monkeypatch):
    """run_full_experiment should pass explicit config and record it in metadata."""
    calls = []

    monkeypatch.setattr(
        experiment,
        "PRIVACY_LEVELS",
        [
            experiment.PrivacyLevel("non-private", 0.0, "baseline"),
            experiment.PrivacyLevel("strong-dp", 5.0, "strong"),
        ],
    )

    def fake_run_single_experiment(
        privacy_level,
        seed,
        n_samples,
        n_features,
        n_classes,
        hidden_dim,
        epochs,
        batch_size,
        lr,
        n_shadows,
        max_grad_norm,
        cluster_std,
        delta,
    ):
        calls.append(
            {
                "privacy_level": privacy_level.name,
                "seed": seed,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "n_shadows": n_shadows,
                "max_grad_norm": max_grad_norm,
                "cluster_std": cluster_std,
                "delta": delta,
            }
        )
        return {
            "privacy_level": privacy_level.name,
            "noise_multiplier": privacy_level.noise_multiplier,
            "seed": seed,
            "epsilon": float("inf") if privacy_level.noise_multiplier == 0 else 3.0,
            "train_accuracy": 0.9,
            "test_accuracy": 0.8,
            "train_loss": 0.2,
            "test_loss": 0.3,
            "generalization_gap": 0.1,
            "attack_auc": 0.6,
            "attack_accuracy": 0.55,
        }

    monkeypatch.setattr(experiment, "run_single_experiment", fake_run_single_experiment)

    results = experiment.run_full_experiment(
        seeds=[7, 9],
        n_samples=120,
        n_features=6,
        n_classes=3,
        hidden_dim=33,
        epochs=5,
        batch_size=16,
        lr=0.07,
        n_shadows=2,
        max_grad_norm=0.7,
        cluster_std=1.8,
        delta=1e-6,
    )

    assert len(calls) == 4
    assert all(c["hidden_dim"] == 33 for c in calls)
    assert all(c["epochs"] == 5 for c in calls)
    assert all(c["batch_size"] == 16 for c in calls)
    assert all(c["lr"] == 0.07 for c in calls)
    assert all(c["n_shadows"] == 2 for c in calls)
    assert all(c["max_grad_norm"] == 0.7 for c in calls)
    assert all(c["cluster_std"] == 1.8 for c in calls)
    assert all(c["delta"] == 1e-6 for c in calls)

    metadata = results["metadata"]
    assert metadata["hidden_dim"] == 33
    assert metadata["epochs"] == 5
    assert metadata["batch_size"] == 16
    assert metadata["lr"] == 0.07
    assert metadata["n_shadows"] == 2
    assert metadata["max_grad_norm"] == 0.7
    assert metadata["cluster_std"] == 1.8
    assert metadata["delta"] == 1e-6
