"""Run the full poisoning sweep experiment."""

import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data import generate_gaussian_clusters, make_datasets, poison_labels
from src.model import MLP, evaluate_accuracy, train_model


@dataclass
class ExperimentConfig:
    """Configuration for the poisoning sensitivity experiment."""

    n_samples: int = 500
    n_features: int = 10
    n_classes: int = 5
    cluster_std: float = 2.0
    poison_fractions: tuple = (0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    hidden_widths: tuple = (32, 64, 128)
    seeds: tuple = (42, 123, 7)
    n_epochs: int = 200
    lr: float = 0.05
    batch_size: int = 64
    train_fraction: float = 0.7
    data_seed: int = 42


@dataclass
class RunResult:
    """Result of a single training run."""

    poison_fraction: float
    hidden_width: int
    seed: int
    train_accuracy: float
    test_accuracy: float
    train_clean_accuracy: float
    generalization_gap: float
    final_loss: float
    elapsed_seconds: float


def run_single(
    config: ExperimentConfig,
    poison_fraction: float,
    hidden_width: int,
    seed: int,
    X: np.ndarray,
    y_clean: np.ndarray,
) -> RunResult:
    """Run a single training experiment.

    Args:
        config: Experiment configuration.
        poison_fraction: Fraction of training labels to flip.
        hidden_width: MLP hidden layer width.
        seed: Random seed for this run.
        X: Feature array (shared across runs).
        y_clean: Clean label array (shared across runs).

    Returns:
        RunResult with all metrics.
    """
    t0 = time.time()

    # Poison labels with a combined seed for reproducibility
    poison_seed = seed + int(poison_fraction * 1000)
    y_poisoned = poison_labels(y_clean, poison_fraction, config.n_classes, seed=poison_seed)

    # Split into train/test
    datasets = make_datasets(
        X, y_poisoned, y_clean,
        train_fraction=config.train_fraction,
        seed=config.data_seed,
    )

    # Build and train model
    torch.manual_seed(seed)
    model = MLP(config.n_features, hidden_width, config.n_classes)
    losses = train_model(
        model, datasets["train_dataset"],
        n_epochs=config.n_epochs, lr=config.lr,
        batch_size=config.batch_size, seed=seed,
    )

    # Evaluate
    train_acc = evaluate_accuracy(model, datasets["X_train"], datasets["train_dataset"].tensors[1])
    test_acc = evaluate_accuracy(model, datasets["X_test"], datasets["y_test"])
    train_clean_acc = evaluate_accuracy(model, datasets["X_train"], datasets["train_clean_labels"])
    gen_gap = train_acc - test_acc

    elapsed = time.time() - t0

    return RunResult(
        poison_fraction=poison_fraction,
        hidden_width=hidden_width,
        seed=seed,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        train_clean_accuracy=train_clean_acc,
        generalization_gap=gen_gap,
        final_loss=losses[-1],
        elapsed_seconds=elapsed,
    )


def run_sweep(config: ExperimentConfig) -> list[RunResult]:
    """Run the full poisoning sweep across all configurations.

    Args:
        config: Experiment configuration.

    Returns:
        List of RunResult objects for all (fraction, width, seed) combos.
    """
    # Generate data once
    X, y_clean = generate_gaussian_clusters(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        seed=config.data_seed,
        cluster_std=config.cluster_std,
    )

    results = []
    total = len(config.poison_fractions) * len(config.hidden_widths) * len(config.seeds)
    done = 0

    for pf in config.poison_fractions:
        for hw in config.hidden_widths:
            for s in config.seeds:
                result = run_single(config, pf, hw, s, X, y_clean)
                results.append(result)
                done += 1
                if done % 9 == 0 or done == total:
                    print(f"  [{done}/{total}] poison={pf:.0%} width={hw} "
                          f"test_acc={result.test_accuracy:.3f}")

    return results
