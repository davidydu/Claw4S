"""Experiment runner: sweep depth vs width across parameter budgets and tasks."""

import json
import os

import torch

from src.models import FlexibleMLP, compute_width_for_budget, count_parameters
from src.tasks import make_sparse_parity_data, make_regression_data
from src.trainer import train_model


# Experiment configuration
PARAM_BUDGETS = [5_000, 20_000, 50_000]
DEPTHS = [1, 2, 4, 8]
SEED = 42

# Sparse parity config
N_BITS = 20
K_RELEVANT = 3          # k=3 is learnable in reasonable time
N_TRAIN_PARITY = 3000   # More data to learn the pattern
N_TEST_PARITY = 1000

# Per-task hyperparameters
TASK_HPARAMS = {
    "sparse_parity": {
        "max_epochs": 1500,
        "lr": 3e-3,
        "weight_decay": 1e-2,
        "patience": 250,
        "convergence_threshold": 0.85,
    },
    "smooth_regression": {
        "max_epochs": 800,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 200,
        "convergence_threshold": 0.90,
    },
}


def run_single_experiment(
    task_data: dict,
    num_hidden_layers: int,
    param_budget: int,
    hparams: dict,
) -> dict:
    """Run a single depth-width experiment.

    Args:
        task_data: Task data dict from task generators.
        num_hidden_layers: Number of hidden layers.
        param_budget: Target parameter count.
        hparams: Task-specific hyperparameters.

    Returns:
        Dict with config and results.
    """
    input_dim = task_data["input_dim"]
    output_dim = task_data["output_dim"]

    # Set seed before model init for reproducible weight initialization
    torch.manual_seed(SEED)

    width = compute_width_for_budget(
        input_dim, output_dim, num_hidden_layers, param_budget
    )
    model = FlexibleMLP(input_dim, width, output_dim, num_hidden_layers)
    actual_params = count_parameters(model)

    print(
        f"\n  Config: depth={num_hidden_layers}, width={width}, "
        f"params={actual_params:,} (budget={param_budget:,})"
    )

    results = train_model(
        model,
        task_data,
        max_epochs=hparams["max_epochs"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        convergence_threshold=hparams["convergence_threshold"],
        patience=hparams["patience"],
        seed=SEED,
    )

    return {
        "param_budget": param_budget,
        "num_hidden_layers": num_hidden_layers,
        "hidden_width": width,
        "actual_params": actual_params,
        "task_name": task_data["task_name"],
        "task_type": task_data["task_type"],
        **{k: v for k, v in results.items()
           if k not in ("epoch_losses", "epoch_test_metrics")},
        # Store sampled curves (every 50th epoch) to keep JSON small
        "loss_curve_sampled": results["epoch_losses"][::50],
        "metric_curve_sampled": results["epoch_test_metrics"][::50],
    }


def run_all_experiments() -> dict:
    """Run the full depth-vs-width sweep across budgets and tasks.

    Returns:
        Dict with all results and metadata.
    """
    torch.manual_seed(SEED)

    # Generate task data
    parity_data = make_sparse_parity_data(
        n_bits=N_BITS, k_relevant=K_RELEVANT,
        n_train=N_TRAIN_PARITY, n_test=N_TEST_PARITY, seed=SEED,
    )
    reg_data = make_regression_data(seed=SEED)

    tasks = [
        ("sparse_parity", parity_data),
        ("smooth_regression", reg_data),
    ]

    all_results = []

    for task_name, task_data in tasks:
        hparams = TASK_HPARAMS[task_name]
        for budget in PARAM_BUDGETS:
            for depth in DEPTHS:
                print(
                    f"\n{'='*60}\n"
                    f"Task: {task_name} | Budget: {budget:,} | "
                    f"Depth: {depth} layers"
                    f"\n{'='*60}"
                )
                try:
                    result = run_single_experiment(
                        task_data, depth, budget, hparams
                    )
                    all_results.append(result)
                except ValueError as e:
                    print(f"  SKIPPED: {e}")
                    all_results.append({
                        "param_budget": budget,
                        "num_hidden_layers": depth,
                        "task_name": task_name,
                        "skipped": True,
                        "error": str(e),
                    })

    metadata = {
        "seed": SEED,
        "n_bits": N_BITS,
        "k_relevant": K_RELEVANT,
        "param_budgets": PARAM_BUDGETS,
        "depths": DEPTHS,
        "task_hparams": {k: {kk: vv for kk, vv in v.items()}
                         for k, v in TASK_HPARAMS.items()},
        "num_experiments": len(all_results),
        "torch_version": torch.__version__,
    }

    return {"metadata": metadata, "results": all_results}


def save_results(results: dict, output_dir: str = "results") -> str:
    """Save results to JSON file.

    Args:
        results: Full results dict from run_all_experiments.
        output_dir: Directory to save results.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    return path
