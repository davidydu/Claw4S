"""Full experiment runner for membership inference under differential privacy.

Orchestrates the complete pipeline:
  1. For each privacy level (non-private, weak, moderate, strong DP):
     a. For each seed (3 seeds for statistical robustness):
        - Generate synthetic data with member/non-member split
        - Train target model with specified DP config
        - Train 3 shadow models with same DP config
        - Train attack classifier on shadow data
        - Run attack on target model
        - Record metrics: attack AUC, accuracy, model utility, epsilon
  2. Aggregate results across seeds (mean +/- std)
  3. Save results to JSON and generate plots
"""

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from src.attack import run_attack, train_attack_classifier, train_shadow_models
from src.data import generate_gaussian_clusters, make_member_nonmember_split
from src.dp_sgd import DPConfig, compute_epsilon
from src.train import evaluate_model, train_model


@dataclass
class PrivacyLevel:
    """A named privacy configuration."""
    name: str
    noise_multiplier: float  # sigma
    description: str


# Four privacy levels as specified
PRIVACY_LEVELS = [
    PrivacyLevel("non-private", 0.0, "No DP (baseline)"),
    PrivacyLevel("weak-dp", 0.5, "Weak DP (sigma=0.5, large epsilon)"),
    PrivacyLevel("moderate-dp", 2.0, "Moderate DP (sigma=2.0)"),
    PrivacyLevel("strong-dp", 5.0, "Strong DP (sigma=5.0, small epsilon)"),
]


def run_single_experiment(
    privacy_level: PrivacyLevel,
    seed: int,
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    hidden_dim: int = 128,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 0.1,
    n_shadows: int = 3,
    max_grad_norm: float = 1.0,
    cluster_std: float = 2.5,
    delta: float = 1e-5,
) -> dict:
    """Run one experiment: train target + shadows, attack, measure.

    Args:
        privacy_level: Privacy configuration to test.
        seed: Random seed for this trial.
        n_samples: Number of data samples.
        n_features: Feature dimensionality.
        n_classes: Number of classes.
        hidden_dim: MLP hidden layer width.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        n_shadows: Number of shadow models.
        max_grad_norm: Gradient clipping norm.
        cluster_std: Std dev of Gaussian clusters (larger = more overlap = harder).
        delta: Target delta for (epsilon, delta)-DP accounting.

    Returns:
        Dictionary of metrics for this trial.
    """
    dp_config = DPConfig(
        noise_multiplier=privacy_level.noise_multiplier,
        max_grad_norm=max_grad_norm,
        delta=delta,
    )

    # Generate target data
    X, y = generate_gaussian_clusters(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        cluster_std=cluster_std,
        seed=seed,
    )
    member_ds, nonmember_ds = make_member_nonmember_split(
        X, y, member_ratio=0.5, seed=seed
    )

    # Train target model
    target_model, train_losses = train_model(
        train_dataset=member_ds,
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_classes=n_classes,
        dp_config=dp_config if privacy_level.noise_multiplier > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )

    # Evaluate target model utility
    train_acc, train_loss = evaluate_model(target_model, member_ds)
    test_acc, test_loss = evaluate_model(target_model, nonmember_ds)

    # Compute epsilon
    n_member = len(member_ds)
    n_steps = epochs * (n_member // batch_size + (1 if n_member % batch_size else 0))
    epsilon = compute_epsilon(
        noise_multiplier=privacy_level.noise_multiplier,
        n_steps=n_steps,
        batch_size=batch_size,
        n_samples=n_member,
        delta=delta,
    )

    # Train shadow models + attack classifier
    shadow_dp = dp_config if privacy_level.noise_multiplier > 0 else None
    attack_X, attack_y = train_shadow_models(
        n_shadows=n_shadows,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        cluster_std=cluster_std,
        dp_config=shadow_dp,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        base_seed=seed * 100,
    )

    attack_clf = train_attack_classifier(
        attack_X, attack_y, epochs=50, lr=0.01, seed=seed + 5000
    )

    # Run attack
    attack_results = run_attack(attack_clf, target_model, member_ds, nonmember_ds)

    return {
        "privacy_level": privacy_level.name,
        "noise_multiplier": privacy_level.noise_multiplier,
        "seed": seed,
        "epsilon": epsilon,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "generalization_gap": train_acc - test_acc,
        "attack_auc": attack_results["attack_auc"],
        "attack_accuracy": attack_results["attack_accuracy"],
    }


def run_full_experiment(
    seeds: list[int] | None = None,
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    hidden_dim: int = 128,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 0.1,
    n_shadows: int = 3,
    max_grad_norm: float = 1.0,
    cluster_std: float = 2.5,
    delta: float = 1e-5,
) -> dict:
    """Run the full experiment across all privacy levels and seeds.

    Args:
        seeds: List of random seeds. Defaults to [42, 123, 456].
        n_samples: Samples per dataset.
        n_features: Feature dimensionality.
        n_classes: Number of classes.
        hidden_dim: Hidden layer width for target/shadow MLPs.
        epochs: Training epochs for target/shadow models.
        batch_size: Training batch size.
        lr: SGD learning rate.
        n_shadows: Number of shadow models per trial.
        max_grad_norm: DP-SGD clipping norm C.
        cluster_std: Standard deviation of Gaussian clusters.
        delta: Target delta for (epsilon, delta)-DP accounting.

    Returns:
        Complete results dictionary with per-trial and aggregated metrics.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    all_results = []
    start_time = time.time()

    total_runs = len(PRIVACY_LEVELS) * len(seeds)
    run_idx = 0

    for pl in PRIVACY_LEVELS:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {pl.name} (sigma={pl.noise_multiplier}), seed={seed}")

            result = run_single_experiment(
                privacy_level=pl,
                seed=seed,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                n_shadows=n_shadows,
                max_grad_norm=max_grad_norm,
                cluster_std=cluster_std,
                delta=delta,
            )

            print(f"  epsilon={result['epsilon']:.2f}, "
                  f"test_acc={result['test_accuracy']:.3f}, "
                  f"attack_auc={result['attack_auc']:.3f}")

            all_results.append(result)

    elapsed = time.time() - start_time

    # Aggregate results per privacy level
    aggregated = {}
    for pl in PRIVACY_LEVELS:
        pl_results = [r for r in all_results if r["privacy_level"] == pl.name]
        metrics = {}
        for key in ["epsilon", "train_accuracy", "test_accuracy",
                     "generalization_gap", "attack_auc", "attack_accuracy"]:
            values = [r[key] for r in pl_results if r[key] != float("inf")]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
            else:
                metrics[key] = {
                    "mean": float("inf"),
                    "std": 0.0,
                    "values": [float("inf")] * len(pl_results),
                }
        aggregated[pl.name] = {
            "noise_multiplier": pl.noise_multiplier,
            "description": pl.description,
            "metrics": metrics,
        }

    return {
        "metadata": {
            "n_privacy_levels": len(PRIVACY_LEVELS),
            "n_seeds": len(seeds),
            "seeds": seeds,
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
            "total_runs": total_runs,
            "elapsed_seconds": elapsed,
        },
        "per_trial": all_results,
        "aggregated": aggregated,
    }


def print_summary(results: dict) -> str:
    """Generate a human-readable summary of the experiment.

    Args:
        results: Results dictionary from run_full_experiment.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("MEMBERSHIP INFERENCE UNDER DIFFERENTIAL PRIVACY — RESULTS")
    lines.append("=" * 72)
    meta = results["metadata"]
    lines.append(f"Samples: {meta['n_samples']} | Features: {meta['n_features']} | "
                 f"Classes: {meta['n_classes']} | Seeds: {meta['n_seeds']}")
    lines.append(f"Runtime: {meta['elapsed_seconds']:.1f}s")
    lines.append("")

    header = f"{'Privacy Level':<16} {'sigma':>6} {'epsilon':>10} {'Test Acc':>10} {'Attack AUC':>12} {'Attack Acc':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for pl in PRIVACY_LEVELS:
        agg = results["aggregated"][pl.name]
        m = agg["metrics"]
        eps_str = (f"{m['epsilon']['mean']:.1f}" if m["epsilon"]["mean"] != float("inf")
                   else "inf")
        lines.append(
            f"{pl.name:<16} {agg['noise_multiplier']:>6.1f} "
            f"{eps_str:>10} "
            f"{m['test_accuracy']['mean']:.3f}+/-{m['test_accuracy']['std']:.3f} "
            f"{m['attack_auc']['mean']:.3f}+/-{m['attack_auc']['std']:.3f} "
            f"{m['attack_accuracy']['mean']:.3f}+/-{m['attack_accuracy']['std']:.3f}"
        )

    lines.append("")

    # Key finding
    non_priv_auc = results["aggregated"]["non-private"]["metrics"]["attack_auc"]["mean"]
    strong_auc = results["aggregated"]["strong-dp"]["metrics"]["attack_auc"]["mean"]
    reduction = non_priv_auc - strong_auc

    lines.append("KEY FINDINGS:")
    lines.append(f"  - Non-private attack AUC: {non_priv_auc:.3f}")
    lines.append(f"  - Strong DP attack AUC:   {strong_auc:.3f}")
    lines.append(f"  - AUC reduction:          {reduction:.3f}")
    lines.append(f"  - Strong DP brings AUC to {'near-random (0.5)' if strong_auc < 0.6 else 'above random'}")
    lines.append("")
    lines.append("THESIS: DP-SGD with strong privacy (small epsilon) reduces")
    lines.append("membership inference attack AUC toward random guessing (0.5).")
    lines.append("=" * 72)

    return "\n".join(lines)
