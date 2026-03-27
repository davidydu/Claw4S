"""Run the full gradient norm phase transition experiment.

Trains 2-layer MLPs on modular addition and regression tasks at multiple
dataset fractions, tracking per-layer gradient norms throughout training.
Analyzes whether gradient norm phase transitions predict generalization.

Must be run from the submission directory: submissions/gradient-norms/
"""

import json
import os
import sys
import time

# Guard: must run from the correct directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: run.py must be executed from submissions/gradient-norms/")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))

from src.data import make_modular_addition_dataset, make_regression_dataset
from src.trainer import train_and_track
from src.analysis import analyze_run
from src.plotting import (
    plot_single_run,
    plot_summary_grid,
    plot_lag_barchart,
    plot_weight_norms,
)

# --- Configuration ---
FRACTIONS = [0.50, 0.70, 0.90]
HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 0.1
N_EPOCHS = 3000
SEED = 42
# Extra seeds for variance analysis on modular addition (grokking task)
VARIANCE_SEEDS = [42, 123, 7]
VARIANCE_EPOCHS = 2000  # shorter runs for variance (transitions happen by epoch ~600)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Gradient Norm Phase Transition Experiment")
print("=" * 60)
print(f"Tasks: modular_addition (mod 97), regression")
print(f"Fractions: {FRACTIONS}")
print(f"Hidden dim: {HIDDEN_DIM}, LR: {LR}, WD: {WEIGHT_DECAY}")
print(f"Epochs: {N_EPOCHS}, Seed: {SEED}")
print(f"Variance seeds (modular addition): {VARIANCE_SEEDS}")
print("=" * 60)

all_results = []
all_analyses = []
t0 = time.time()

# --- Modular addition runs (primary seed) ---
for frac in FRACTIONS:
    print(f"\n[1/3] Training modular_addition, frac={frac:.0%} ...")
    dataset = make_modular_addition_dataset(frac=frac, seed=SEED)
    result = train_and_track(
        dataset,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        n_epochs=N_EPOCHS,
        seed=SEED,
    )
    analysis = analyze_run(result)
    all_results.append(result)
    all_analyses.append(analysis)
    print(f"  Train acc: {result['train_metric'][-1]:.3f}, "
          f"Test acc: {result['test_metric'][-1]:.3f}")
    print(f"  Grad transition: epoch {analysis['gnorm_transition_epoch']}, "
          f"Metric transition: epoch {analysis['metric_transition_epoch']}, "
          f"Lag: {analysis['lag_epochs']} epochs")

# --- Regression runs (primary seed) ---
for frac in FRACTIONS:
    print(f"\n[2/3] Training regression, frac={frac:.0%} ...")
    dataset = make_regression_dataset(frac=frac, seed=SEED)
    result = train_and_track(
        dataset,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        n_epochs=N_EPOCHS,
        seed=SEED,
    )
    analysis = analyze_run(result)
    all_results.append(result)
    all_analyses.append(analysis)
    print(f"  Train R^2: {result['train_metric'][-1]:.3f}, "
          f"Test R^2: {result['test_metric'][-1]:.3f}")
    print(f"  Grad transition: epoch {analysis['gnorm_transition_epoch']}, "
          f"Metric transition: epoch {analysis['metric_transition_epoch']}, "
          f"Lag: {analysis['lag_epochs']} epochs")

# --- Multi-seed variance analysis (modular addition only) ---
print(f"\n[3/3] Multi-seed variance analysis (modular addition) ...")
import numpy as _np

variance_data: dict[float, list[int]] = {frac: [] for frac in FRACTIONS}
for seed in VARIANCE_SEEDS:
    for frac in FRACTIONS:
        print(f"  seed={seed}, frac={frac:.0%} ...", end=" ")
        dataset = make_modular_addition_dataset(frac=frac, seed=seed)
        result = train_and_track(
            dataset,
            hidden_dim=HIDDEN_DIM,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            n_epochs=VARIANCE_EPOCHS,
            seed=seed,
        )
        analysis = analyze_run(result)
        variance_data[frac].append(analysis["lag_epochs"])
        print(f"lag={analysis['lag_epochs']}")

print("\n  Multi-seed lag statistics (modular addition):")
print(f"  {'Frac':>5}  {'Mean Lag':>9}  {'Std Dev':>8}  {'Min':>6}  {'Max':>6}")
variance_summary = {}
for frac in FRACTIONS:
    lags = variance_data[frac]
    mean_lag = _np.mean(lags)
    std_lag = _np.std(lags, ddof=1) if len(lags) > 1 else 0.0
    print(f"  {frac:>4.0%}  {mean_lag:>9.1f}  {std_lag:>8.1f}  {min(lags):>6d}  {max(lags):>6d}")
    variance_summary[str(frac)] = {
        "seeds": VARIANCE_SEEDS,
        "lags": lags,
        "mean": float(mean_lag),
        "std": float(std_lag),
        "min": min(lags),
        "max": max(lags),
    }

elapsed = time.time() - t0
print(f"\n--- Training complete in {elapsed:.1f}s ---")

# --- Generate plots ---
print("\n[3/4] Generating plots ...")
for r, a in zip(all_results, all_analyses):
    path = plot_single_run(r, a, RESULTS_DIR)
    print(f"  Saved: {path}")

summary_path = plot_summary_grid(all_results, all_analyses, RESULTS_DIR)
print(f"  Saved: {summary_path}")

lag_path = plot_lag_barchart(all_analyses, RESULTS_DIR)
print(f"  Saved: {lag_path}")

weight_path = plot_weight_norms(all_results, all_analyses, RESULTS_DIR)
print(f"  Saved: {weight_path}")

# --- Save JSON results ---
print("\n[4/4] Saving results to results/ ...")
summary_data = {
    "experiment": "gradient_norm_phase_transitions",
    "config": {
        "tasks": ["modular_addition", "regression"],
        "fractions": FRACTIONS,
        "hidden_dim": HIDDEN_DIM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "n_epochs": N_EPOCHS,
        "seed": SEED,
    },
    "runs": [],
}

for r, a in zip(all_results, all_analyses):
    run_summary = {
        "task_name": a["task_name"],
        "frac": a["frac"],
        "gnorm_transition_epoch": a["gnorm_transition_epoch"],
        "gnorm_steepest_decrease_epoch": a["gnorm_steepest_decrease_epoch"],
        "metric_transition_epoch": a["metric_transition_epoch"],
        "lag_epochs": a["lag_epochs"],
        "lag_positive": a["lag_positive"],
        "xcorr_best_lag": a["xcorr_best_lag"],
        "xcorr_best_correlation": a["xcorr_best_correlation"],
        "per_layer": a["per_layer"],
        "pearson_r": a["pearson_r"],
        "pearson_p": a["pearson_p"],
        "final_train_metric": a["final_train_metric"],
        "final_test_metric": a["final_test_metric"],
    }
    summary_data["runs"].append(run_summary)

summary_data["variance_analysis"] = variance_summary

with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(summary_data, f, indent=2)
print(f"  Saved: results/results.json")

# --- Print summary ---
print("\n" + "=" * 60)
print("SUMMARY: Gradient Norm Phase Transition Lead Times")
print("=" * 60)
print(f"{'Task':<22} {'Frac':>5} {'Grad Trans':>11} {'Metric Trans':>13} {'Lag':>6} {'Leads?':>7}")
print("-" * 60)
for a in all_analyses:
    leads = "YES" if a["lag_positive"] else "NO"
    print(f"{a['task_name']:<22} {a['frac']:>4.0%} "
          f"{a['gnorm_transition_epoch']:>11d} "
          f"{a['metric_transition_epoch']:>13d} "
          f"{a['lag_epochs']:>6d} "
          f"{leads:>7}")

mod_lags = [a["lag_epochs"] for a in all_analyses if a["task_name"] == "modular_addition"]
reg_lags = [a["lag_epochs"] for a in all_analyses if a["task_name"] == "regression"]
print(f"\nModular addition avg lag: {sum(mod_lags)/len(mod_lags):.0f} epochs")
print(f"Regression avg lag:      {sum(reg_lags)/len(reg_lags):.0f} epochs")

n_positive = sum(1 for a in all_analyses if a["lag_positive"])
print(f"\nGradient norm leads metric in {n_positive}/{len(all_analyses)} runs")

# Methodological note
print("\n--- Methodological Note ---")
print("Peak-based lag (above) and cross-correlation lag (in results.json) may disagree.")
print("Peak lag measures epoch difference between gradient peak and metric transition.")
print("Cross-correlation measures derivative-level alignment, which can show negative lag")
print("when gradient norm changes are smoother/delayed relative to sharp metric transitions.")
print("Both are reported; peak-based lag is the primary metric for the 'leading indicator' thesis.")

print(f"\nTotal runtime: {elapsed:.1f}s")
