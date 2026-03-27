"""Main experiment runner: train models across sizes and privacy levels.

Orchestrates the full experimental sweep:
- 5 model sizes x 3 privacy levels x 3 seeds = 45 training runs
- Aggregates results, fits scaling laws, compares exponents
"""

import json
import os
import time

import numpy as np
import torch

from src.data import make_dataloaders
from src.model import MLP, count_parameters
from src.train import train_standard, train_dp_sgd, evaluate
from src.scaling import fit_scaling_law


# Experiment configuration (all pinned for reproducibility)
HIDDEN_SIZES = [16, 32, 64, 128, 256]
SEEDS = [42, 123, 789]
PRIVACY_CONFIGS = {
    "non_private": {"noise_multiplier": 0.0},
    "moderate_dp": {"noise_multiplier": 1.0},
    "strong_dp": {"noise_multiplier": 3.0},
}
N_SAMPLES = 500
N_FEATURES = 10
N_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.01
MAX_GRAD_NORM = 1.0


def run_single_experiment(
    hidden_size: int,
    privacy_level: str,
    noise_multiplier: float,
    seed: int,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> dict:
    """Run a single training experiment and return metrics.

    Args:
        hidden_size: Number of hidden units in the MLP.
        privacy_level: Name of the privacy configuration.
        noise_multiplier: DP-SGD noise multiplier (0.0 for non-private).
        seed: Random seed.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Dictionary with experiment configuration and results.
    """
    train_loader, test_loader, n_feat, n_cls = make_dataloaders(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        seed=seed,
        batch_size=BATCH_SIZE,
    )

    torch.manual_seed(seed)
    model = MLP(n_feat, hidden_size, n_cls)
    n_params = count_parameters(model)

    t0 = time.time()

    if noise_multiplier == 0.0:
        model = train_standard(model, train_loader, lr=lr, epochs=epochs, seed=seed)
    else:
        model = train_dp_sgd(
            model,
            train_loader,
            lr=lr,
            epochs=epochs,
            max_grad_norm=MAX_GRAD_NORM,
            noise_multiplier=noise_multiplier,
            seed=seed,
        )

    elapsed = time.time() - t0
    test_loss, accuracy = evaluate(model, test_loader)

    return {
        "hidden_size": hidden_size,
        "n_params": n_params,
        "privacy_level": privacy_level,
        "noise_multiplier": noise_multiplier,
        "seed": seed,
        "test_loss": test_loss,
        "accuracy": accuracy,
        "train_time_s": round(elapsed, 2),
    }


def run_full_experiment() -> dict:
    """Run the full experimental sweep: all sizes x privacy levels x seeds.

    Returns:
        Dictionary containing:
            'raw_results': List of per-run result dictionaries.
            'aggregated': Dict mapping privacy_level -> {
                'param_counts': [...],
                'mean_losses': [...],
                'std_losses': [...],
                'mean_accuracies': [...],
            }
            'scaling_fits': Dict mapping privacy_level -> fit results.
            'summary': High-level comparison of scaling exponents.
    """
    raw_results = []
    total_runs = len(HIDDEN_SIZES) * len(PRIVACY_CONFIGS) * len(SEEDS)
    run_idx = 0

    for privacy_level, config in PRIVACY_CONFIGS.items():
        for hidden_size in HIDDEN_SIZES:
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"  [{run_idx}/{total_runs}] "
                    f"hidden={hidden_size}, {privacy_level}, seed={seed}"
                )
                result = run_single_experiment(
                    hidden_size=hidden_size,
                    privacy_level=privacy_level,
                    noise_multiplier=config["noise_multiplier"],
                    seed=seed,
                )
                raw_results.append(result)
                print(
                    f"    -> loss={result['test_loss']:.4f}, "
                    f"acc={result['accuracy']:.3f}, "
                    f"time={result['train_time_s']}s"
                )

    # Aggregate across seeds
    aggregated = {}
    for privacy_level in PRIVACY_CONFIGS:
        level_results = [r for r in raw_results if r["privacy_level"] == privacy_level]
        param_counts = []
        mean_losses = []
        std_losses = []
        mean_accs = []

        for h in HIDDEN_SIZES:
            runs = [r for r in level_results if r["hidden_size"] == h]
            losses = [r["test_loss"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            param_counts.append(runs[0]["n_params"])
            mean_losses.append(float(np.mean(losses)))
            std_losses.append(float(np.std(losses)))
            mean_accs.append(float(np.mean(accs)))

        aggregated[privacy_level] = {
            "param_counts": param_counts,
            "mean_losses": mean_losses,
            "std_losses": std_losses,
            "mean_accuracies": mean_accs,
        }

    # Fit scaling laws
    scaling_fits = {}
    for privacy_level, agg in aggregated.items():
        param_counts = np.array(agg["param_counts"], dtype=np.float64)
        mean_losses = np.array(agg["mean_losses"], dtype=np.float64)
        try:
            fit = fit_scaling_law(param_counts, mean_losses)
            scaling_fits[privacy_level] = fit
        except RuntimeError as e:
            print(f"  Warning: Scaling law fit failed for {privacy_level}: {e}")
            scaling_fits[privacy_level] = {
                "a": None,
                "alpha": None,
                "l_inf": None,
                "r_squared": None,
                "error": str(e),
            }

    # Summary comparison
    summary = {}
    non_private_alpha = scaling_fits.get("non_private", {}).get("alpha")
    for level, fit in scaling_fits.items():
        entry = {
            "alpha": fit.get("alpha"),
            "a": fit.get("a"),
            "l_inf": fit.get("l_inf"),
            "r_squared": fit.get("r_squared"),
        }
        if non_private_alpha is not None and fit.get("alpha") is not None:
            entry["alpha_ratio_vs_non_private"] = round(
                fit["alpha"] / non_private_alpha, 4
            ) if non_private_alpha > 0 else None
        summary[level] = entry

    return {
        "raw_results": raw_results,
        "aggregated": aggregated,
        "scaling_fits": scaling_fits,
        "summary": summary,
        "config": {
            "hidden_sizes": HIDDEN_SIZES,
            "seeds": SEEDS,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "epochs": EPOCHS,
            "lr": LR,
            "max_grad_norm": MAX_GRAD_NORM,
            "privacy_configs": PRIVACY_CONFIGS,
        },
    }


def save_results(results: dict, output_dir: str = "results") -> str:
    """Save experiment results to JSON.

    Args:
        results: Full experiment results dictionary.
        output_dir: Directory to save results.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "experiment_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return path
