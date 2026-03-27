"""Experiment runner: trains MLPs and evaluates calibration across shifts.

Orchestrates the full experimental pipeline: data generation, model training,
and calibration evaluation for all (width, shift, seed) configurations.
"""

import time
import torch
import numpy as np
from typing import Any

from src.data import make_datasets, N_FEATURES, N_CLASSES
from src.models import TwoLayerMLP, train_model, predict_proba
from src.metrics import (expected_calibration_error, brier_score,
                         confidence_histogram, accuracy)

# Experiment configuration
HIDDEN_WIDTHS = [16, 32, 64, 128, 256]
SHIFT_MAGNITUDES = [0.0, 0.5, 1.0, 2.0, 4.0]
SEEDS = [42, 43, 44]
N_BINS = 10
TRAIN_EPOCHS = 200
LEARNING_RATE = 0.01


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_single_experiment(hidden_width: int, seed: int,
                          shift_magnitudes: list[float] | None = None,
                          ) -> dict[str, Any]:
    """Run one experiment: train a model and evaluate on all shifts.

    Args:
        hidden_width: Width of hidden layer.
        seed: Random seed.
        shift_magnitudes: List of shift magnitudes to test.

    Returns:
        Dict with training info and per-shift calibration metrics.
    """
    if shift_magnitudes is None:
        shift_magnitudes = SHIFT_MAGNITUDES

    set_seed(seed)
    datasets = make_datasets(seed=seed, shift_magnitudes=shift_magnitudes)
    X_train_np, y_train_np = datasets['train']

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)

    # Count parameters
    model = TwoLayerMLP(N_FEATURES, hidden_width, N_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())

    # Train
    set_seed(seed)  # Reset seed before training for reproducibility
    losses = train_model(model, X_train, y_train,
                        lr=LEARNING_RATE, epochs=TRAIN_EPOCHS)

    # Evaluate on training data
    probs_train, _ = predict_proba(model, X_train)
    probs_train_np = probs_train.numpy()
    train_acc = accuracy(probs_train_np, y_train_np)
    train_ece, _ = expected_calibration_error(probs_train_np, y_train_np,
                                              n_bins=N_BINS)

    result = {
        'hidden_width': hidden_width,
        'seed': seed,
        'n_params': n_params,
        'final_train_loss': losses[-1],
        'train_accuracy': train_acc,
        'train_ece': train_ece,
        'shifts': {},
    }

    # Evaluate on each shifted test set
    for mag in shift_magnitudes:
        X_test_np, y_test_np = datasets[f'test_shift_{mag}']
        X_test = torch.tensor(X_test_np, dtype=torch.float32)

        probs_test, _ = predict_proba(model, X_test)
        probs_test_np = probs_test.numpy()

        ece, bin_data = expected_calibration_error(probs_test_np, y_test_np,
                                                   n_bins=N_BINS)
        bs = brier_score(probs_test_np, y_test_np)
        conf_hist = confidence_histogram(probs_test_np, n_bins=N_BINS)
        acc = accuracy(probs_test_np, y_test_np)

        result['shifts'][str(mag)] = {
            'shift_magnitude': mag,
            'accuracy': acc,
            'ece': ece,
            'brier_score': bs,
            'mean_confidence': conf_hist['mean_confidence'],
            'reliability': bin_data,
            'confidence_histogram': conf_hist,
        }

    return result


def run_all_experiments() -> dict[str, Any]:
    """Run the full experiment grid: widths x shifts x seeds.

    Returns:
        Dict with metadata, raw results, and aggregated statistics.
    """
    start_time = time.time()
    all_results = []
    total = len(HIDDEN_WIDTHS) * len(SEEDS)
    completed = 0

    for width in HIDDEN_WIDTHS:
        for seed in SEEDS:
            completed += 1
            print(f"  [{completed}/{total}] width={width}, seed={seed}")
            result = run_single_experiment(width, seed)
            all_results.append(result)

    elapsed = time.time() - start_time

    # Aggregate: compute mean and std across seeds for each (width, shift)
    aggregated = aggregate_results(all_results)

    return {
        'metadata': {
            'hidden_widths': HIDDEN_WIDTHS,
            'shift_magnitudes': SHIFT_MAGNITUDES,
            'seeds': SEEDS,
            'n_bins': N_BINS,
            'train_epochs': TRAIN_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'n_features': N_FEATURES,
            'n_classes': N_CLASSES,
            'elapsed_seconds': round(elapsed, 2),
            'n_experiments': len(all_results),
        },
        'raw_results': all_results,
        'aggregated': aggregated,
    }


def aggregate_results(raw_results: list[dict]) -> list[dict]:
    """Aggregate results across seeds for each (width, shift) pair.

    Args:
        raw_results: List of per-experiment result dicts.

    Returns:
        List of dicts with mean/std for each (width, shift) combination.
    """
    from collections import defaultdict

    # Group by (width, shift)
    groups: dict[tuple, list] = defaultdict(list)
    for r in raw_results:
        width = r['hidden_width']
        for shift_key, shift_data in r['shifts'].items():
            groups[(width, shift_key)].append(shift_data)

    aggregated = []
    for (width, shift_key), entries in sorted(groups.items()):
        eces = [e['ece'] for e in entries]
        briers = [e['brier_score'] for e in entries]
        accs = [e['accuracy'] for e in entries]
        confs = [e['mean_confidence'] for e in entries]

        aggregated.append({
            'hidden_width': width,
            'shift_magnitude': float(shift_key),
            'n_seeds': len(entries),
            'ece_mean': float(np.mean(eces)),
            'ece_std': float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0,
            'brier_mean': float(np.mean(briers)),
            'brier_std': float(np.std(briers, ddof=1)) if len(briers) > 1 else 0.0,
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            'confidence_mean': float(np.mean(confs)),
            'confidence_std': float(np.std(confs, ddof=1)) if len(confs) > 1 else 0.0,
        })

    return aggregated
