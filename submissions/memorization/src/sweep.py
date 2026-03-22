# src/sweep.py
"""Model size sweep for memorization capacity measurement."""

import torch

from src.data import generate_dataset
from src.model import MLP, count_parameters
from src.train import train_model, compute_accuracy

# Default hidden dimensions to sweep
DEFAULT_HIDDEN_DIMS = [5, 10, 20, 40, 80, 160, 320, 640]

# Default experiment parameters
DEFAULT_N_TRAIN = 200
DEFAULT_N_TEST = 50
DEFAULT_D = 20
DEFAULT_N_CLASSES = 10
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 5000
DEFAULT_LR = 0.001


def run_sweep(
    hidden_dims: list[int] | None = None,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    d: int = DEFAULT_D,
    n_classes: int = DEFAULT_N_CLASSES,
    seed: int = DEFAULT_SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    lr: float = DEFAULT_LR,
) -> dict:
    """Run memorization sweep across model sizes and label types.

    Args:
        hidden_dims: List of hidden layer widths to sweep.
        n_train: Number of training samples.
        n_test: Number of test samples.
        d: Feature dimensionality.
        n_classes: Number of classes.
        seed: Random seed.
        max_epochs: Maximum training epochs per run.
        lr: Learning rate.

    Returns:
        Dictionary with sweep results and metadata.
    """
    if hidden_dims is None:
        hidden_dims = DEFAULT_HIDDEN_DIMS

    label_types = ["random", "structured"]
    results = []

    print(f"Sweeping {len(hidden_dims)} hidden dims x {len(label_types)} label types")
    print(f"  n_train={n_train}, n_test={n_test}, d={d}, n_classes={n_classes}")
    print(f"  max_epochs={max_epochs}, lr={lr}, seed={seed}")
    print()

    for label_type in label_types:
        print(f"--- Label type: {label_type} ---")

        # Generate dataset (same seed for each label type ensures same X)
        X_train, y_train, X_test, y_test = generate_dataset(
            n_train=n_train,
            n_test=n_test,
            d=d,
            n_classes=n_classes,
            seed=seed,
            label_type=label_type,
        )

        for h in hidden_dims:
            print(f"  h={h}: ", end="", flush=True)

            # Set all seeds for reproducibility
            torch.manual_seed(seed + h)

            # Create model
            model = MLP(input_dim=d, hidden_dim=h, num_classes=n_classes)
            n_params = count_parameters(model)

            # Train
            result = train_model(
                model=model,
                X=X_train,
                y=y_train,
                max_epochs=max_epochs,
                lr=lr,
                seed=seed + h,
                log_interval=0,  # suppress per-epoch logging
            )

            # Evaluate on test set
            test_acc = compute_accuracy(model, X_test, y_test)

            print(f"params={n_params:>6d}, "
                  f"train_acc={result.final_train_acc:.4f}, "
                  f"test_acc={test_acc:.4f}, "
                  f"epochs={result.epochs_run}, "
                  f"converged={'yes' if result.convergence_epoch >= 0 else 'no'}")

            results.append({
                "label_type": label_type,
                "hidden_dim": h,
                "n_params": n_params,
                "train_acc": result.final_train_acc,
                "test_acc": test_acc,
                "train_loss": result.final_train_loss,
                "convergence_epoch": result.convergence_epoch,
                "epochs_run": result.epochs_run,
            })

        print()

    return {
        "metadata": {
            "n_train": n_train,
            "n_test": n_test,
            "d": d,
            "n_classes": n_classes,
            "seed": seed,
            "max_epochs": max_epochs,
            "lr": lr,
            "hidden_dims": hidden_dims,
            "label_types": label_types,
        },
        "results": results,
    }


def run_multi_seed_sweep(
    seeds: list[int] | None = None,
    hidden_dims: list[int] | None = None,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    d: int = DEFAULT_D,
    n_classes: int = DEFAULT_N_CLASSES,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    lr: float = DEFAULT_LR,
) -> dict:
    """Run sweep with multiple seeds to quantify variance.

    Args:
        seeds: List of random seeds. Defaults to [42, 43, 44].
        hidden_dims: List of hidden layer widths.
        Other args: same as run_sweep.

    Returns:
        Dictionary with per-seed results and aggregated statistics.
    """
    if seeds is None:
        seeds = [42, 43, 44]
    if hidden_dims is None:
        hidden_dims = DEFAULT_HIDDEN_DIMS

    all_seed_results = []
    for i, s in enumerate(seeds):
        print(f"\n{'='*40} Seed {s} ({i+1}/{len(seeds)}) {'='*40}")
        result = run_sweep(
            hidden_dims=hidden_dims,
            n_train=n_train,
            n_test=n_test,
            d=d,
            n_classes=n_classes,
            seed=s,
            max_epochs=max_epochs,
            lr=lr,
        )
        all_seed_results.append(result)

    # Aggregate: compute mean and std of train_acc and test_acc per (label_type, hidden_dim)
    import numpy as np
    label_types = all_seed_results[0]["metadata"]["label_types"]
    aggregated = []

    for label_type in label_types:
        for h in hidden_dims:
            train_accs = []
            test_accs = []
            n_params = None
            for seed_result in all_seed_results:
                for r in seed_result["results"]:
                    if r["label_type"] == label_type and r["hidden_dim"] == h:
                        train_accs.append(r["train_acc"])
                        test_accs.append(r["test_acc"])
                        n_params = r["n_params"]
                        break

            aggregated.append({
                "label_type": label_type,
                "hidden_dim": h,
                "n_params": n_params,
                "train_acc_mean": float(np.mean(train_accs)),
                "train_acc_std": float(np.std(train_accs, ddof=1)) if len(train_accs) > 1 else 0.0,
                "test_acc_mean": float(np.mean(test_accs)),
                "test_acc_std": float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
                "n_seeds": len(train_accs),
            })

    return {
        "metadata": {
            **all_seed_results[0]["metadata"],
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "per_seed_results": [r["results"] for r in all_seed_results],
        "aggregated": aggregated,
    }
