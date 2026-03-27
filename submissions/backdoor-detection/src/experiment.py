"""Experiment runner: sweep over configurations and collect results."""

import time
import itertools
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from src.data import generate_clean_data, inject_backdoor, make_datasets
from src.model import train_model, extract_activations
from src.spectral import compute_spectral_scores, compute_detection_auc, compute_eigenvalue_ratio


@dataclass
class ExperimentConfig:
    """Configuration for a single backdoor detection experiment."""
    poison_fraction: float
    trigger_strength: float
    hidden_dim: int
    n_samples: int = 500
    n_features: int = 10
    n_classes: int = 5
    target_class: int = 0
    trigger_features: tuple[int, ...] = (0, 1, 2)
    epochs: int = 50
    lr: float = 0.01
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: dict
    detection_auc: float
    eigenvalue_ratio: float
    top_5_eigenvalues: list[float]
    clean_model_accuracy: float
    backdoored_model_accuracy: float
    backdoor_success_rate: float
    n_poisoned: int
    n_total: int
    elapsed_seconds: float


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single backdoor detection experiment.

    Steps:
    1. Generate clean synthetic data
    2. Inject backdoor into a copy
    3. Train a clean model and a backdoored model
    4. Extract penultimate-layer activations from the backdoored model
    5. Compute spectral signatures and detection AUC

    Args:
        config: Experiment configuration.

    Returns:
        ExperimentResult with metrics and timing.
    """
    start_time = time.time()

    # 1. Generate clean data
    X_clean, y_clean = generate_clean_data(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        seed=config.seed,
    )

    # 2. Inject backdoor
    X_poison, y_poison, poison_mask = inject_backdoor(
        X_clean, y_clean,
        poison_fraction=config.poison_fraction,
        target_class=config.target_class,
        trigger_strength=config.trigger_strength,
        trigger_features=config.trigger_features,
        seed=config.seed,
    )

    # 3. Train clean model
    clean_dataset = make_datasets(X_clean, y_clean)
    clean_model = train_model(
        clean_dataset,
        input_dim=config.n_features,
        hidden_dim=config.hidden_dim,
        n_classes=config.n_classes,
        epochs=config.epochs,
        lr=config.lr,
        seed=config.seed,
    )

    # 4. Train backdoored model
    poison_dataset = make_datasets(X_poison, y_poison)
    backdoor_model = train_model(
        poison_dataset,
        input_dim=config.n_features,
        hidden_dim=config.hidden_dim,
        n_classes=config.n_classes,
        epochs=config.epochs,
        lr=config.lr,
        seed=config.seed + 1,  # Different seed for model init
    )

    # 5. Compute accuracies
    with torch.no_grad():
        # Clean model accuracy on clean data
        X_tensor = torch.from_numpy(X_clean).float()
        y_tensor = torch.from_numpy(y_clean).long()
        clean_preds = clean_model(X_tensor).argmax(dim=1)
        clean_acc = (clean_preds == y_tensor).float().mean().item()

        # Backdoored model accuracy on clean data (should still be high)
        bd_preds = backdoor_model(X_tensor).argmax(dim=1)
        bd_acc = (bd_preds == y_tensor).float().mean().item()

        # Backdoor success rate: accuracy of trigger on non-target samples
        non_target = y_clean != config.target_class
        X_triggered = X_clean[non_target].copy()
        for feat_idx in config.trigger_features:
            X_triggered[:, feat_idx] = config.trigger_strength
        X_trig_tensor = torch.from_numpy(X_triggered).float()
        trig_preds = backdoor_model(X_trig_tensor).argmax(dim=1)
        target_tensor = torch.full((len(X_triggered),), config.target_class, dtype=torch.long)
        backdoor_sr = (trig_preds == target_tensor).float().mean().item()

    # 6. Extract activations from backdoored model on poisoned data
    activations = extract_activations(backdoor_model, poison_dataset)
    act_np = activations.numpy()

    # 7. Spectral analysis
    scores, eigenvalues, _ = compute_spectral_scores(act_np)
    detection_auc = compute_detection_auc(scores, poison_mask)
    eig_ratio = compute_eigenvalue_ratio(eigenvalues)

    elapsed = time.time() - start_time

    return ExperimentResult(
        config=asdict(config),
        detection_auc=detection_auc,
        eigenvalue_ratio=eig_ratio,
        top_5_eigenvalues=[float(e) for e in eigenvalues[:5]],
        clean_model_accuracy=clean_acc,
        backdoored_model_accuracy=bd_acc,
        backdoor_success_rate=backdoor_sr,
        n_poisoned=int(poison_mask.sum()),
        n_total=len(poison_mask),
        elapsed_seconds=elapsed,
    )


def run_sweep(
    poison_fractions: list[float] = [0.05, 0.10, 0.20, 0.30],
    trigger_strengths: list[float] = [3.0, 5.0, 10.0],
    hidden_dims: list[int] = [64, 128, 256],
    seed: int = 42,
    progress_callback: Optional[callable] = None,
) -> list[ExperimentResult]:
    """Run a full parameter sweep.

    Iterates over all combinations of poison_fraction, trigger_strength,
    and hidden_dim.

    Args:
        poison_fractions: List of poison fractions to test.
        trigger_strengths: List of trigger strengths to test.
        hidden_dims: List of hidden dimensions to test.
        seed: Base random seed.
        progress_callback: Optional function called with (i, total, result).

    Returns:
        List of ExperimentResult for each configuration.
    """
    configs = list(itertools.product(poison_fractions, trigger_strengths, hidden_dims))
    total = len(configs)
    results = []

    for i, (pf, ts, hd) in enumerate(configs):
        config = ExperimentConfig(
            poison_fraction=pf,
            trigger_strength=ts,
            hidden_dim=hd,
            seed=seed,
        )
        result = run_single_experiment(config)
        results.append(result)

        if progress_callback is not None:
            progress_callback(i + 1, total, result)

    return results
