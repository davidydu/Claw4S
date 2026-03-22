"""Phase diagram sweep orchestration.

Runs a grid of training experiments over weight_decay, dataset_fraction,
and hidden_dim, collecting results for phase diagram construction.
"""

import time

import numpy as np
import torch

from src.analysis import Phase, classify_phase, compute_grokking_gap
from src.data import generate_modular_addition_data, split_dataset
from src.model import GrokkingMLP
from src.train import TrainConfig, train_model


# Default sweep grid
DEFAULT_WEIGHT_DECAYS = [0.0, 0.001, 0.01, 0.1, 1.0]
DEFAULT_DATASET_FRACTIONS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_HIDDEN_DIMS = [32, 64, 128]

# Fixed hyperparameters
DEFAULT_P = 97
DEFAULT_EMBED_DIM = 16
DEFAULT_LR = 1e-3
DEFAULT_MAX_EPOCHS = 5000
DEFAULT_SEED = 42


def run_single(
    p: int,
    embed_dim: int,
    hidden_dim: int,
    weight_decay: float,
    train_fraction: float,
    max_epochs: int,
    seed: int,
) -> dict:
    """Run a single training experiment and classify the result.

    Returns:
        Dict with config, result metrics, phase classification, and timing.
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate and split data
    data = generate_modular_addition_data(p)
    train_data, test_data = split_dataset(data, train_fraction, seed=seed)

    # Create model
    model = GrokkingMLP(p, embed_dim, hidden_dim)
    param_count = model.count_parameters()

    # Configure training
    config = TrainConfig(
        lr=DEFAULT_LR,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        seed=seed,
    )

    # Train
    start_time = time.time()
    result = train_model(model, train_data, test_data, config)
    elapsed = time.time() - start_time

    # Classify phase
    phase = classify_phase(result)
    gap = compute_grokking_gap(result)

    return {
        "config": {
            "p": p,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "train_fraction": train_fraction,
            "max_epochs": max_epochs,
            "seed": seed,
            "param_count": param_count,
        },
        "metrics": {
            "final_train_acc": result.final_train_acc,
            "final_test_acc": result.final_test_acc,
            "total_epochs": result.total_epochs,
            "epoch_train_95": result.epoch_train_95,
            "epoch_test_95": result.epoch_test_95,
            "train_accs": result.train_accs,
            "test_accs": result.test_accs,
            "train_losses": result.train_losses,
            "test_losses": result.test_losses,
            "logged_epochs": result.logged_epochs,
        },
        "phase": phase.value,
        "grokking_gap": gap,
        "elapsed_seconds": round(elapsed, 2),
    }


def run_sweep(
    weight_decays: list[float] | None = None,
    dataset_fractions: list[float] | None = None,
    hidden_dims: list[int] | None = None,
    p: int = DEFAULT_P,
    embed_dim: int = DEFAULT_EMBED_DIM,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Run full phase diagram sweep over hyperparameter grid.

    Args:
        weight_decays: List of weight decay values. Default: [0, 1e-3, 1e-2, 1e-1, 1.0]
        dataset_fractions: List of train fractions. Default: [0.3, 0.5, 0.7, 0.9]
        hidden_dims: List of hidden layer widths. Default: [32, 64, 128]
        p: Prime modulus.
        embed_dim: Embedding dimension.
        max_epochs: Maximum training epochs per run.
        seed: Global random seed.

    Returns:
        List of result dicts from run_single.
    """
    if weight_decays is None:
        weight_decays = DEFAULT_WEIGHT_DECAYS
    if dataset_fractions is None:
        dataset_fractions = DEFAULT_DATASET_FRACTIONS
    if hidden_dims is None:
        hidden_dims = DEFAULT_HIDDEN_DIMS

    total = len(weight_decays) * len(dataset_fractions) * len(hidden_dims)
    results = []

    print(f"Starting sweep: {total} runs")
    print(f"  Weight decays: {weight_decays}")
    print(f"  Dataset fractions: {dataset_fractions}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Max epochs: {max_epochs}, p={p}")
    print()

    sweep_start = time.time()
    run_idx = 0

    for hd in hidden_dims:
        for wd in weight_decays:
            for frac in dataset_fractions:
                run_idx += 1
                print(
                    f"  [{run_idx}/{total}] hidden={hd}, wd={wd}, "
                    f"frac={frac} ... ",
                    end="",
                    flush=True,
                )

                result = run_single(
                    p=p,
                    embed_dim=embed_dim,
                    hidden_dim=hd,
                    weight_decay=wd,
                    train_fraction=frac,
                    max_epochs=max_epochs,
                    seed=seed,
                )

                phase = result["phase"]
                elapsed = result["elapsed_seconds"]
                train_acc = result["metrics"]["final_train_acc"]
                test_acc = result["metrics"]["final_test_acc"]
                print(
                    f"{phase} (train={train_acc:.1%}, test={test_acc:.1%}) "
                    f"[{elapsed:.1f}s]"
                )

                results.append(result)

    total_time = time.time() - sweep_start
    print(f"\nSweep complete: {total} runs in {total_time:.1f}s")

    return results
