"""Full experiment runner: sweep over hidden dims, weight decays, and seeds."""

import json
import os
import time
from typing import Any

import numpy as np

from src.data import generate_dataset
from src.train import train_model, evaluate_accuracy


# Experiment configuration — all parameters pinned for reproducibility.
HIDDEN_DIMS = [32, 64, 128]
WEIGHT_DECAYS = [0.0, 0.001, 0.01, 0.1, 1.0]
SEEDS = [42, 123, 7]
N_TRAIN = 2000
N_TEST = 1000
N_GENUINE = 10
EPOCHS = 100
LR = 0.01
BATCH_SIZE = 128


def run_single(
    hidden_dim: int,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    """Run a single training + evaluation configuration.

    Returns:
        Dictionary with config and all accuracy metrics.
    """
    data = generate_dataset(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_genuine=N_GENUINE,
        seed=seed,
    )
    input_dim = data["metadata"]["n_total_features"]

    model = train_model(
        train_dataset=data["train_dataset"],
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        weight_decay=weight_decay,
        seed=seed,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
    )

    train_acc = evaluate_accuracy(model, data["train_dataset"])
    test_acc_with = evaluate_accuracy(model, data["test_with_shortcut"])
    test_acc_without = evaluate_accuracy(model, data["test_without_shortcut"])
    shortcut_reliance = test_acc_with - test_acc_without

    return {
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "seed": seed,
        "train_acc": round(train_acc, 4),
        "test_acc_with_shortcut": round(test_acc_with, 4),
        "test_acc_without_shortcut": round(test_acc_without, 4),
        "shortcut_reliance": round(shortcut_reliance, 4),
    }


def run_experiment(results_dir: str = "results") -> dict[str, Any]:
    """Run the full sweep: 3 hidden dims x 5 weight decays x 3 seeds = 45 runs.

    Args:
        results_dir: Directory to save results.

    Returns:
        Full results dictionary.
    """
    os.makedirs(results_dir, exist_ok=True)

    total = len(HIDDEN_DIMS) * len(WEIGHT_DECAYS) * len(SEEDS)
    print(f"[1/4] Starting experiment: {total} configurations")
    print(f"       Hidden dims: {HIDDEN_DIMS}")
    print(f"       Weight decays: {WEIGHT_DECAYS}")
    print(f"       Seeds: {SEEDS}")

    results = []
    start = time.time()

    for hi, hd in enumerate(HIDDEN_DIMS):
        for wi, wd in enumerate(WEIGHT_DECAYS):
            for si, s in enumerate(SEEDS):
                idx = hi * len(WEIGHT_DECAYS) * len(SEEDS) + wi * len(SEEDS) + si + 1
                print(f"  [{idx}/{total}] hidden={hd}, wd={wd}, seed={s}", end="")
                result = run_single(hd, wd, s)
                results.append(result)
                print(f"  -> test_acc={result['test_acc_without_shortcut']:.3f}, "
                      f"reliance={result['shortcut_reliance']:.3f}")

    elapsed = time.time() - start
    print(f"[2/4] Experiment complete in {elapsed:.1f}s")

    # Aggregate: mean and std over seeds for each (hidden_dim, weight_decay)
    print("[3/4] Computing aggregate statistics")
    aggregates = []
    for hd in HIDDEN_DIMS:
        for wd in WEIGHT_DECAYS:
            runs = [r for r in results
                    if r["hidden_dim"] == hd and r["weight_decay"] == wd]
            agg = {
                "hidden_dim": hd,
                "weight_decay": wd,
                "n_seeds": len(runs),
                "train_acc_mean": round(float(np.mean([r["train_acc"] for r in runs])), 4),
                "train_acc_std": round(float(np.std([r["train_acc"] for r in runs])), 4),
                "test_acc_with_mean": round(float(np.mean([r["test_acc_with_shortcut"] for r in runs])), 4),
                "test_acc_with_std": round(float(np.std([r["test_acc_with_shortcut"] for r in runs])), 4),
                "test_acc_without_mean": round(float(np.mean([r["test_acc_without_shortcut"] for r in runs])), 4),
                "test_acc_without_std": round(float(np.std([r["test_acc_without_shortcut"] for r in runs])), 4),
                "shortcut_reliance_mean": round(float(np.mean([r["shortcut_reliance"] for r in runs])), 4),
                "shortcut_reliance_std": round(float(np.std([r["shortcut_reliance"] for r in runs])), 4),
            }
            aggregates.append(agg)

    # Key findings
    findings = _compute_findings(aggregates)

    output = {
        "metadata": {
            "n_configs": total,
            "hidden_dims": HIDDEN_DIMS,
            "weight_decays": WEIGHT_DECAYS,
            "seeds": SEEDS,
            "n_genuine_features": N_GENUINE,
            "n_total_features": N_GENUINE + 1,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "elapsed_seconds": round(elapsed, 1),
        },
        "individual_runs": results,
        "aggregates": aggregates,
        "findings": findings,
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[4/4] Saving results to {results_dir}/")

    return output


def _compute_findings(aggregates: list[dict]) -> list[str]:
    """Extract key scientific findings from aggregated results."""
    findings = []

    # Finding 1: Shortcut reliance without regularization
    no_reg = [a for a in aggregates if a["weight_decay"] == 0.0]
    mean_reliance_no_reg = np.mean([a["shortcut_reliance_mean"] for a in no_reg])
    if mean_reliance_no_reg > 0.05:
        findings.append(
            f"Without regularization (wd=0), models show mean shortcut reliance of "
            f"{mean_reliance_no_reg:.3f}, confirming preferential shortcut learning."
        )

    # Finding 2: Does weight decay reduce reliance?
    for wd in [0.01, 0.1]:
        reg = [a for a in aggregates if a["weight_decay"] == wd]
        mean_reliance_reg = np.mean([a["shortcut_reliance_mean"] for a in reg])
        reduction = mean_reliance_no_reg - mean_reliance_reg
        if reduction > 0.01:
            findings.append(
                f"Weight decay={wd} reduces shortcut reliance by {reduction:.3f} "
                f"(from {mean_reliance_no_reg:.3f} to {mean_reliance_reg:.3f})."
            )

    # Finding 3: Test accuracy without shortcut vs with shortcut
    all_with = np.mean([a["test_acc_with_mean"] for a in aggregates])
    all_without = np.mean([a["test_acc_without_mean"] for a in aggregates])
    gap = all_with - all_without
    if gap > 0.01:
        findings.append(
            f"Average test accuracy drops by {gap:.3f} when the shortcut is removed "
            f"({all_with:.3f} -> {all_without:.3f}), demonstrating shortcut dependence."
        )

    # Finding 4: Model width effect
    for hd in HIDDEN_DIMS:
        hd_no_reg = [a for a in aggregates
                     if a["hidden_dim"] == hd and a["weight_decay"] == 0.0]
        if hd_no_reg:
            rel = hd_no_reg[0]["shortcut_reliance_mean"]
            findings.append(
                f"Hidden dim={hd} with no regularization: shortcut reliance = {rel:.3f}."
            )

    # Finding 5: Best regularization (with nuance about over-regularization)
    best = min(aggregates, key=lambda a: a["shortcut_reliance_mean"])
    findings.append(
        f"Lowest shortcut reliance ({best['shortcut_reliance_mean']:.3f}) achieved with "
        f"hidden_dim={best['hidden_dim']}, weight_decay={best['weight_decay']}."
    )

    # Finding 6: Over-regularization check (wd=1.0 may kill all learning)
    strong_reg = [a for a in aggregates if a["weight_decay"] == 1.0]
    mean_train_strong = np.mean([a["train_acc_mean"] for a in strong_reg])
    if mean_train_strong < 0.55:
        findings.append(
            f"Weight decay=1.0 eliminates shortcut reliance but also prevents learning "
            f"(mean train acc={mean_train_strong:.3f}, near chance). "
            f"Shortcut reliance=0 is trivial when the model learns nothing."
        )

    # Finding 7: Moderate wd effectiveness
    moderate_wds = [0.001, 0.01]
    for wd in moderate_wds:
        reg = [a for a in aggregates if a["weight_decay"] == wd]
        mean_rel = np.mean([a["shortcut_reliance_mean"] for a in reg])
        if abs(mean_reliance_no_reg - mean_rel) < 0.02:
            findings.append(
                f"Moderate weight decay={wd} does not meaningfully reduce shortcut reliance "
                f"({mean_rel:.3f} vs {mean_reliance_no_reg:.3f} baseline), suggesting "
                f"the shortcut signal is too strong for mild regularization."
            )

    return findings
