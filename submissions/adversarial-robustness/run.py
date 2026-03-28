#!/usr/bin/env python3
"""
Adversarial Robustness Scaling -- main experiment runner.

Trains MLPs of varying widths on synthetic data, generates FGSM and PGD
adversarial examples, and measures how robust accuracy scales with model size.
Runs across two datasets (circles, moons) and three random seeds for
statistical variance.

Usage (from submissions/adversarial-robustness/):
    .venv/bin/python run.py
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Working-directory guard: must be run from the submission root
# ---------------------------------------------------------------------------
if not os.path.isfile(os.path.join(os.getcwd(), "SKILL.md")):
    print("ERROR: run.py must be executed from the submission directory")
    print("       (the folder containing SKILL.md).")
    print(f"       Current directory: {os.getcwd()}")
    sys.exit(1)

import torch
import numpy as np

from src.data import make_dataloaders
from src.models import build_model, HIDDEN_WIDTHS
from src.train import train_model, evaluate_clean
from src.attacks import fgsm_attack, pgd_attack, evaluate_robust, EPSILONS
from src.analysis import (
    compute_robustness_gaps,
    summarize_results_by_dataset,
    plot_clean_vs_robust,
    plot_robustness_gap,
    plot_param_count_scaling,
)

SEEDS = [42, 123, 7]
DATASETS = [
    {"name": "circles", "noise": 0.15},
    {"name": "moons", "noise": 0.15},
]


def run_single_experiment(dataset_name: str, noise: float, seed: int,
                          verbose: bool = False) -> list[dict]:
    """Run one experiment: train all model sizes, evaluate all epsilons.

    Args:
        dataset_name: "circles" or "moons".
        noise: Noise level for data generation.
        seed: Random seed.
        verbose: Print per-model details.

    Returns:
        List of result dicts for this dataset/seed combination.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, _, X_test, y_test = make_dataloaders(
        dataset=dataset_name, n_samples=2000, noise=noise,
        test_fraction=0.2, batch_size=256, seed=seed,
    )

    results: list[dict] = []
    for width in HIDDEN_WIDTHS:
        model = build_model(hidden_width=width, input_dim=2, seed=seed)
        n_params = model.param_count()

        train_info = train_model(
            model, train_loader, max_epochs=2000, lr=1e-3,
            patience=50, seed=seed, verbose=False,
        )
        clean_acc = evaluate_clean(model, X_test, y_test)

        if verbose:
            print(f"    Width={width:4d} ({n_params:,} params): "
                  f"{train_info['final_epoch']} epochs, clean_acc={clean_acc:.4f}")

        for eps in EPSILONS:
            fgsm_acc = evaluate_robust(model, X_test, y_test, fgsm_attack, epsilon=eps)
            pgd_acc = evaluate_robust(model, X_test, y_test, pgd_attack, epsilon=eps,
                                      n_steps=10)
            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "hidden_width": width,
                "param_count": n_params,
                "epochs": train_info["final_epoch"],
                "final_loss": train_info["final_loss"],
                "clean_acc": clean_acc,
                "epsilon": eps,
                "fgsm_acc": fgsm_acc,
                "pgd_acc": pgd_acc,
            })
    return results


def aggregate_across_seeds(all_results: list[dict]) -> list[dict]:
    """Average results across seeds per (dataset, width, epsilon).

    Args:
        all_results: Full results list with fgsm_gap and pgd_gap fields.

    Returns:
        Aggregated results (one row per dataset/width/epsilon).
    """
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["dataset"], r["hidden_width"], r["epsilon"])
        groups[key].append(r)

    aggregated = []
    for (ds, w, eps), rows in sorted(groups.items()):
        agg = {
            "dataset": ds,
            "hidden_width": w,
            "param_count": rows[0]["param_count"],
            "epsilon": eps,
            "n_seeds": len(rows),
            "clean_acc_mean": float(np.mean([r["clean_acc"] for r in rows])),
            "clean_acc_std": float(np.std([r["clean_acc"] for r in rows])),
            "fgsm_acc_mean": float(np.mean([r["fgsm_acc"] for r in rows])),
            "fgsm_acc_std": float(np.std([r["fgsm_acc"] for r in rows])),
            "pgd_acc_mean": float(np.mean([r["pgd_acc"] for r in rows])),
            "pgd_acc_std": float(np.std([r["pgd_acc"] for r in rows])),
            "fgsm_gap_mean": float(np.mean([r["fgsm_gap"] for r in rows])),
            "fgsm_gap_std": float(np.std([r["fgsm_gap"] for r in rows])),
            "pgd_gap_mean": float(np.mean([r["pgd_gap"] for r in rows])),
            "pgd_gap_std": float(np.std([r["pgd_gap"] for r in rows])),
        }
        aggregated.append(agg)
    return aggregated


def main() -> None:
    """Run the full adversarial robustness scaling experiment."""
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Adversarial Robustness Scaling Experiment")
    print("=" * 70)
    print(f"Hidden widths: {HIDDEN_WIDTHS}")
    print(f"Epsilons:      {EPSILONS}")
    print(f"Seeds:         {SEEDS}")
    print(f"Datasets:      {[d['name'] for d in DATASETS]}")
    n_total = len(HIDDEN_WIDTHS) * len(EPSILONS) * len(SEEDS) * len(DATASETS)
    print(f"Total runs:    {n_total}")
    print(f"Output dir:    {output_dir}")
    print()

    # Run all experiments
    all_results: list[dict] = []
    t_start = time.time()

    for ds_cfg in DATASETS:
        ds_name = ds_cfg["name"]
        noise = ds_cfg["noise"]
        print(f"[Dataset: {ds_name}] (noise={noise})")

        for seed in SEEDS:
            print(f"  Seed={seed}:")
            results = run_single_experiment(ds_name, noise, seed, verbose=True)
            all_results.extend(results)

        print()

    elapsed = time.time() - t_start
    print(f"Total training + evaluation time: {elapsed:.1f}s")
    print()

    # Compute robustness gaps
    print("Computing robustness gaps and summary statistics...")
    all_results = compute_robustness_gaps(all_results)

    # Aggregate across seeds
    aggregated = aggregate_across_seeds(all_results)
    dataset_summaries = summarize_results_by_dataset(all_results)

    # Per-dataset summaries
    for ds_cfg in DATASETS:
        ds_name = ds_cfg["name"]
        summary = dataset_summaries[ds_name]

        print(f"\n  [{ds_name.upper()}] Per-width summary (mean +/- std across {len(SEEDS)} seeds):")
        print(f"  {'Width':>6} {'Params':>8} {'Clean':>12} {'FGSM Gap':>16} {'PGD Gap':>16}")
        print("  " + "-" * 65)
        for w in HIDDEN_WIDTHS:
            w_agg = [a for a in aggregated
                     if a["dataset"] == ds_name and a["hidden_width"] == w]
            if w_agg:
                # Average across epsilons
                clean_mean = w_agg[0]["clean_acc_mean"]
                clean_std = w_agg[0]["clean_acc_std"]
                fgsm_gap_means = [a["fgsm_gap_mean"] for a in w_agg]
                pgd_gap_means = [a["pgd_gap_mean"] for a in w_agg]
                fgsm_gap_stds = [a["fgsm_gap_std"] for a in w_agg]
                pgd_gap_stds = [a["pgd_gap_std"] for a in w_agg]
                mean_fgsm = np.mean(fgsm_gap_means)
                mean_pgd = np.mean(pgd_gap_means)
                mean_fgsm_std = np.mean(fgsm_gap_stds)
                mean_pgd_std = np.mean(pgd_gap_stds)
                print(f"  {w:>6d} {w_agg[0]['param_count']:>8d} "
                      f"{clean_mean:>5.4f}+/-{clean_std:.4f} "
                      f"{mean_fgsm:>6.4f}+/-{mean_fgsm_std:.4f} "
                      f"{mean_pgd:>6.4f}+/-{mean_pgd_std:.4f}")

        if summary.get("corr_logparams_fgsm_gap") is not None:
            print(f"\n  Corr(log params, FGSM gap): "
                  f"{summary['corr_logparams_fgsm_gap']:.4f}")
            print(f"  Corr(log params, PGD gap):  "
                  f"{summary['corr_logparams_pgd_gap']:.4f}")

    print()

    # Generate plots using first-seed data for circles (primary dataset)
    print("Generating plots and saving results...")
    circles_first = [r for r in all_results
                     if r["dataset"] == "circles" and r["seed"] == SEEDS[0]]
    p1 = plot_clean_vs_robust(circles_first, output_dir)
    print(f"  Saved: {p1}")
    p2 = plot_robustness_gap(circles_first, output_dir)
    print(f"  Saved: {p2}")
    p3 = plot_param_count_scaling(circles_first, output_dir)
    print(f"  Saved: {p3}")

    # Save all data
    output_data = {
        "results": all_results,
        "aggregated": aggregated,
        "summary": dataset_summaries["circles"],
        "dataset_summaries": dataset_summaries,
        "config": {
            "hidden_widths": HIDDEN_WIDTHS,
            "epsilons": EPSILONS,
            "seeds": SEEDS,
            "datasets": DATASETS,
        },
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved: {json_path}")

    print()
    print("=" * 70)
    print("Experiment complete. Results saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
