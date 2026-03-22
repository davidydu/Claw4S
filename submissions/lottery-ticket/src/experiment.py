"""Experiment runner for lottery ticket hypothesis tests.

Sweeps over sparsity levels, pruning strategies, tasks, and seeds.
"""

import json
import os
import time

import torch
import numpy as np

from src.model import TwoLayerMLP
from src.data import generate_modular_data, generate_regression_data
from src.pruning import magnitude_prune, random_prune, structured_prune
from src.train import train_classification, train_regression

# Experiment configuration
SPARSITY_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
PRUNING_STRATEGIES = ["magnitude", "random", "structured"]
TASKS = ["modular", "regression"]
SEEDS = [42, 123, 7]

# Model configuration
HIDDEN_DIM = 128
MODULAR_MOD = 97
REGRESSION_N = 200
REGRESSION_D = 20

# Training configuration
MAX_EPOCHS = 500
CLASSIFICATION_LR = 1e-2
REGRESSION_LR = 1e-3
PATIENCE = 100

PRUNING_FNS = {
    "magnitude": magnitude_prune,
    "random": random_prune,
    "structured": structured_prune,
}


def run_single_experiment(
    task: str, strategy: str, sparsity: float, seed: int
) -> dict:
    """Run a single training experiment with given configuration.

    Args:
        task: "modular" or "regression".
        strategy: "magnitude", "random", or "structured".
        sparsity: Fraction of weights/neurons to prune.
        seed: Random seed.

    Returns:
        Dictionary with experiment configuration and results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if task == "modular":
        input_dim = 2 * MODULAR_MOD  # one-hot(a) || one-hot(b)
        output_dim = MODULAR_MOD
        X_train, y_train, X_test, y_test = generate_modular_data(
            mod=MODULAR_MOD, seed=seed
        )
    elif task == "regression":
        input_dim = REGRESSION_D
        output_dim = 1
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_samples=REGRESSION_N, n_features=REGRESSION_D, seed=seed
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    # Create model
    model = TwoLayerMLP(input_dim, HIDDEN_DIM, output_dim)
    total_params = model.count_parameters()

    # Prune at initialization
    prune_fn = PRUNING_FNS[strategy]
    masks = prune_fn(model, sparsity, seed=seed)
    nonzero_params = model.count_nonzero_parameters()

    # Train
    lr = CLASSIFICATION_LR if task == "modular" else REGRESSION_LR

    if task == "modular":
        result = train_classification(
            model, X_train, y_train, X_test, y_test,
            masks, max_epochs=MAX_EPOCHS, lr=lr, patience=PATIENCE,
        )
    else:
        result = train_regression(
            model, X_train, y_train, X_test, y_test,
            masks, max_epochs=MAX_EPOCHS, lr=lr, patience=PATIENCE,
        )

    return {
        "task": task,
        "strategy": strategy,
        "sparsity": sparsity,
        "seed": seed,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "actual_sparsity": 1.0 - nonzero_params / total_params,
        **result,
    }


def run_all_experiments(output_dir: str = "results") -> dict:
    """Run the full experiment sweep.

    Runs all combinations of tasks x strategies x sparsities x seeds.
    For the primary analysis (magnitude pruning), we test all sparsity levels.
    For comparison strategies (random, structured), we test all sparsity levels too.

    Args:
        output_dir: Directory to save results.

    Returns:
        Dictionary with all results and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_runs = len(TASKS) * len(PRUNING_STRATEGIES) * len(SPARSITY_LEVELS) * len(SEEDS)
    run_count = 0
    start_time = time.time()

    print(f"Running {total_runs} experiments...")
    print(f"Tasks: {TASKS}")
    print(f"Strategies: {PRUNING_STRATEGIES}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print(f"Seeds: {SEEDS}")
    print()

    for task in TASKS:
        for strategy in PRUNING_STRATEGIES:
            for sparsity in SPARSITY_LEVELS:
                for seed in SEEDS:
                    run_count += 1
                    print(
                        f"[{run_count}/{total_runs}] "
                        f"task={task}, strategy={strategy}, "
                        f"sparsity={sparsity:.0%}, seed={seed}",
                        end=" ... ",
                    )

                    result = run_single_experiment(task, strategy, sparsity, seed)
                    all_results.append(result)

                    # Print key metric
                    if task == "modular":
                        print(f"test_acc={result['test_acc']:.4f}")
                    else:
                        print(f"test_r2={result['test_r2']:.4f}")

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed:.1f}s")

    # Compile metadata
    output = {
        "metadata": {
            "num_tasks": len(TASKS),
            "num_strategies": len(PRUNING_STRATEGIES),
            "num_sparsity_levels": len(SPARSITY_LEVELS),
            "num_seeds": len(SEEDS),
            "total_runs": total_runs,
            "elapsed_seconds": round(elapsed, 1),
            "hidden_dim": HIDDEN_DIM,
            "max_epochs": MAX_EPOCHS,
            "modular_mod": MODULAR_MOD,
            "regression_n": REGRESSION_N,
            "regression_d": REGRESSION_D,
        },
        "results": all_results,
    }

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {results_path}")

    return output
