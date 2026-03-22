"""Hyperparameter sweep across optimizers, learning rates, and weight decays.

Runs all combinations and collects results for analysis.
Uses tensor-based training (no DataLoader) for speed.
"""

import json
import os
import time

from data import split_data, PRIME, SEED
from train import train_run

# Sweep configuration
OPTIMIZERS = ["sgd", "sgd_momentum", "adam", "adamw"]

LEARNING_RATES = [1e-1, 3e-2, 1e-2]

WEIGHT_DECAYS = [0.0, 0.01, 0.1]

MAX_EPOCHS = 750
LOG_INTERVAL = 75
BATCH_SIZE = 512


def run_sweep(
    optimizers: list[str] | None = None,
    learning_rates: list[float] | None = None,
    weight_decays: list[float] | None = None,
    max_epochs: int = MAX_EPOCHS,
    log_interval: int = LOG_INTERVAL,
    batch_size: int = BATCH_SIZE,
    results_dir: str = "results",
) -> list[dict]:
    """Run the full optimizer sweep.

    Args:
        optimizers: List of optimizer names. Defaults to OPTIMIZERS.
        learning_rates: List of learning rates. Defaults to LEARNING_RATES.
        weight_decays: List of weight decay values. Defaults to WEIGHT_DECAYS.
        max_epochs: Maximum epochs per run.
        log_interval: Epoch interval for logging metrics.
        batch_size: Mini-batch size for training.
        results_dir: Directory to save results.

    Returns:
        List of result dicts, one per configuration.
    """
    if optimizers is None:
        optimizers = OPTIMIZERS
    if learning_rates is None:
        learning_rates = LEARNING_RATES
    if weight_decays is None:
        weight_decays = WEIGHT_DECAYS

    # Create shared data split (same split for all runs)
    train_ds, test_ds = split_data(p=PRIME, seed=SEED)

    # Extract tensors for fast training
    train_a = train_ds.a
    train_b = train_ds.b
    train_t = train_ds.targets
    test_a = test_ds.a
    test_b = test_ds.b
    test_t = test_ds.targets

    total_runs = len(optimizers) * len(learning_rates) * len(weight_decays)
    results = []
    run_idx = 0

    start_time = time.time()

    for opt_name in optimizers:
        for lr in learning_rates:
            for wd in weight_decays:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {opt_name} lr={lr} wd={wd} ...",
                      flush=True)

                result = train_run(
                    optimizer_name=opt_name,
                    lr=lr,
                    weight_decay=wd,
                    train_a=train_a,
                    train_b=train_b,
                    train_t=train_t,
                    test_a=test_a,
                    test_b=test_b,
                    test_t=test_t,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    p=PRIME,
                    seed=SEED,
                    log_interval=log_interval,
                )

                elapsed = time.time() - start_time
                print(f"        -> {result['outcome']} "
                      f"(train={result['final_train_acc']:.3f}, "
                      f"test={result['final_test_acc']:.3f}) "
                      f"[{elapsed:.0f}s elapsed]",
                      flush=True)

                results.append(result)

    elapsed_total = time.time() - start_time
    print(f"\nSweep complete: {total_runs} runs in {elapsed_total:.0f}s", flush=True)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "sweep_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "metadata": {
                "prime": PRIME,
                "seed": SEED,
                "max_epochs": max_epochs,
                "log_interval": log_interval,
                "batch_size": batch_size,
                "num_runs": total_runs,
                "total_seconds": round(elapsed_total, 1),
                "optimizers": optimizers,
                "learning_rates": learning_rates,
                "weight_decays": weight_decays,
            },
            "runs": results,
        }, f, indent=2)
    print(f"Results saved to {save_path}", flush=True)

    return results
