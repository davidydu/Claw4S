"""Sweep experiments for double descent analysis.

Three experiments:
1. Model-wise: Sweep random-feature width p, observe double descent in test MSE.
2. Noise comparison: Repeat model-wise sweep at different noise levels.
3. Neural network comparison: Train MLPs at varying width for comparison.
4. Epoch-wise: At interpolation threshold, track MLP test loss over epochs.
"""

import time

import torch

from src.data import generate_regression_data
from src.model import (
    RandomFeaturesModel,
    create_mlp,
    count_parameters,
    get_interpolation_threshold,
)
from src.training import fit_random_features, train_mlp


def random_features_sweep(
    widths: list[int],
    n_train: int = 200,
    n_test: int = 200,
    d: int = 20,
    noise_std: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """Sweep random-feature width and record train/test MSE.

    Args:
        widths: List of feature counts (p) to try.
        n_train, n_test, d: Dataset parameters.
        noise_std: Label noise standard deviation.
        seed: Random seed.

    Returns:
        List of dicts with: width, n_params, param_ratio, train_loss, test_loss.
    """
    X_train, y_train, X_test, y_test = generate_regression_data(
        n_train, n_test, d, noise_std, seed
    )

    results = []
    for p in widths:
        model = RandomFeaturesModel(d, p, seed=seed)
        metrics = fit_random_features(model, X_train, y_train, X_test, y_test)

        results.append({
            "width": p,
            "n_params": p,  # Only second layer is "trainable"
            "param_ratio": p / n_train,
            "train_loss": metrics["train_loss"],
            "test_loss": metrics["test_loss"],
        })

    return results


def mlp_sweep(
    widths: list[int],
    n_train: int = 200,
    n_test: int = 200,
    d: int = 20,
    noise_std: float = 1.0,
    epochs: int = 4000,
    lr: float = 0.001,
    seed: int = 42,
) -> list[dict]:
    """Sweep MLP hidden width and record final train/test MSE.

    Args:
        widths: List of hidden widths to try.
        n_train, n_test, d: Dataset parameters.
        noise_std: Label noise standard deviation.
        epochs: Training epochs per model.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        List of dicts with: width, n_params, param_ratio, train_loss, test_loss.
    """
    X_train, y_train, X_test, y_test = generate_regression_data(
        n_train, n_test, d, noise_std, seed
    )

    results = []
    for h in widths:
        model = create_mlp(d, h, seed=seed)
        n_params = count_parameters(model)

        train_result = train_mlp(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, record_every=0,
        )

        results.append({
            "width": h,
            "n_params": n_params,
            "param_ratio": n_params / n_train,
            "train_loss": train_result["final_train_loss"],
            "test_loss": train_result["final_test_loss"],
        })

    return results


def epoch_wise_sweep(
    width: int,
    n_train: int = 200,
    n_test: int = 200,
    d: int = 20,
    noise_std: float = 1.0,
    max_epochs: int = 8000,
    lr: float = 0.001,
    record_every: int = 50,
    seed: int = 42,
) -> dict:
    """Train a single MLP and record loss trajectory over epochs.

    Args:
        width: Hidden width (should be near interpolation threshold).
        n_train, n_test, d: Dataset parameters.
        noise_std: Label noise standard deviation.
        max_epochs: Total training epochs.
        lr: Learning rate.
        record_every: Record losses every this many epochs.
        seed: Random seed.

    Returns:
        Dict with: width, n_params, epochs, train_losses, test_losses.
    """
    X_train, y_train, X_test, y_test = generate_regression_data(
        n_train, n_test, d, noise_std, seed
    )

    model = create_mlp(d, width, seed=seed)
    n_params = count_parameters(model)

    result = train_mlp(
        model, X_train, y_train, X_test, y_test,
        epochs=max_epochs, lr=lr, record_every=record_every,
    )

    return {
        "width": width,
        "n_params": n_params,
        "param_ratio": n_params / n_train,
        "epochs": result["epoch_history"],
        "train_losses": result["train_loss_history"],
        "test_losses": result["test_loss_history"],
    }


def run_all_sweeps(config: dict | None = None) -> dict:
    """Run all experiments and return combined results.

    Experiments:
    1. Random features model-wise sweep at 3 noise levels.
    2. MLP model-wise sweep at highest noise level.
    3. MLP epoch-wise sweep at interpolation threshold.

    Args:
        config: Optional configuration overrides.

    Returns:
        Dict with: random_features, mlp_sweep, epoch_wise, metadata.
    """
    if config is None:
        config = {}

    n_train = config.get("n_train", 200)
    n_test = config.get("n_test", 200)
    d = config.get("d", 20)
    seed = config.get("seed", 42)
    lr = config.get("lr", 0.001)
    mlp_epochs = config.get("mlp_epochs", 4000)
    epoch_wise_max = config.get("epoch_wise_max_epochs", 6000)
    epoch_wise_record = config.get("epoch_wise_record_every", 25)

    noise_levels = config.get("noise_levels", [0.1, 0.5, 1.0])

    # Random features widths: dense near threshold (p=n_train)
    rf_widths = config.get("rf_widths", [
        10, 20, 40, 60, 80, 100, 120, 140, 160, 170, 180, 190, 195,
        200, 205, 210, 220, 240, 280, 320, 400, 500, 700, 1000,
    ])

    # MLP widths (fewer, since each requires training)
    mlp_widths = config.get("mlp_widths", [
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 50, 80, 120, 200,
    ])

    # MLP interpolation threshold: h*(d+2)+1 ~ n_train => h ~ (n-1)/(d+2)
    mlp_threshold = max(1, round((n_train - 1) / (d + 2)))

    start_time = time.time()

    # --- Experiment 1: Random features sweep at each noise level ---
    print("[1/4] Random features model-wise sweep...")
    rf_results = {}
    for noise_std in noise_levels:
        label = f"noise_{noise_std}"
        print(f"  noise_std={noise_std}...")
        rf_results[label] = random_features_sweep(
            widths=rf_widths,
            n_train=n_train, n_test=n_test, d=d,
            noise_std=noise_std, seed=seed,
        )

    # --- Experiment 2: MLP sweep at highest noise ---
    print("[2/4] MLP model-wise sweep...")
    highest_noise = max(noise_levels)
    mlp_results = mlp_sweep(
        widths=mlp_widths,
        n_train=n_train, n_test=n_test, d=d,
        noise_std=highest_noise, epochs=mlp_epochs,
        lr=lr, seed=seed,
    )

    # --- Experiment 3: Epoch-wise sweep ---
    print("[3/4] Epoch-wise double descent sweep...")
    # Use MLP width near interpolation threshold
    epoch_width = mlp_threshold
    epoch_wise_results = {}
    for noise_std in noise_levels:
        label = f"noise_{noise_std}"
        print(f"  width={epoch_width}, noise_std={noise_std}...")
        epoch_wise_results[label] = epoch_wise_sweep(
            width=epoch_width,
            n_train=n_train, n_test=n_test, d=d,
            noise_std=noise_std, max_epochs=epoch_wise_max,
            lr=lr, record_every=epoch_wise_record, seed=seed,
        )

    # --- Experiment 4: Multiple seeds for variance ---
    print("[4/4] Multi-seed variance estimation...")
    variance_seeds = config.get("variance_seeds", [42, 123, 456])
    variance_results = []
    for s in variance_seeds:
        results = random_features_sweep(
            widths=rf_widths,
            n_train=n_train, n_test=n_test, d=d,
            noise_std=1.0, seed=s,
        )
        variance_results.append({
            "seed": s,
            "results": results,
        })

    elapsed = time.time() - start_time
    print(f"All experiments completed in {elapsed:.1f}s")

    return {
        "random_features": rf_results,
        "mlp_sweep": mlp_results,
        "epoch_wise": epoch_wise_results,
        "variance": variance_results,
        "metadata": {
            "n_train": n_train,
            "n_test": n_test,
            "d": d,
            "seed": seed,
            "lr": lr,
            "noise_levels": noise_levels,
            "rf_widths": rf_widths,
            "mlp_widths": mlp_widths,
            "rf_interpolation_threshold": n_train,
            "mlp_interpolation_threshold": mlp_threshold,
            "mlp_epochs": mlp_epochs,
            "epoch_wise_max_epochs": epoch_wise_max,
            "variance_seeds": variance_seeds,
            "runtime_seconds": elapsed,
        },
    }
