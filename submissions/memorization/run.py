"""Run the full memorization capacity scaling experiment."""

import os
import sys

# Ensure we are running from the submission directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from src.sweep import run_multi_seed_sweep
from src.analysis import analyze_results
from src.plots import plot_memorization_curves, plot_threshold_comparison
from src.report import generate_report

print("=" * 60)
print("Memorization Capacity Scaling Experiment")
print("=" * 60)

# Step 1: Run multi-seed sweep (3 seeds for variance estimation)
print("\n[1/4] Running model size sweep (3 seeds)...")
multi_results = run_multi_seed_sweep(seeds=[42, 43, 44])

# Use first seed's results as the primary sweep for analysis
# (analysis functions expect the single-seed format)
primary_sweep = {
    "metadata": {k: v for k, v in multi_results["metadata"].items()
                 if k not in ("seeds", "n_seeds")},
    "results": multi_results["per_seed_results"][0],
}
primary_sweep["metadata"]["seed"] = multi_results["metadata"]["seeds"][0]

# Step 2: Analyze results
print("\n[2/4] Analyzing results...")
analysis = analyze_results(primary_sweep)

# Attach multi-seed variance info to analysis
analysis["multi_seed"] = {
    "seeds": multi_results["metadata"]["seeds"],
    "aggregated": multi_results["aggregated"],
}

# Step 3: Generate plots
print("\n[3/4] Generating plots...")
plot_memorization_curves(analysis)
plot_threshold_comparison(analysis)

# Step 4: Generate report
print("\n[4/4] Generating report...")
report = generate_report(primary_sweep, analysis)

print("\n" + "=" * 60)
print("Experiment complete. Key results:")
for label_type in ["random", "structured"]:
    lt = analysis["label_types"][label_type]
    threshold = lt["threshold"]
    sig = lt["sigmoid_fit"]
    print(f"  {label_type} labels:")
    print(f"    Max train acc: {lt['max_train_acc']:.4f}")
    if threshold["achieved"]:
        print(f"    Interpolation threshold: {threshold['threshold_params']:,} params")
    if sig["fit_success"]:
        print(f"    Sigmoid sharpness: {sig['sharpness']:.2f} (R^2={sig['r_squared']:.4f})")

# Print multi-seed variance
print("\nMulti-seed variance (3 seeds):")
for entry in multi_results["aggregated"]:
    if entry["hidden_dim"] in [5, 20, 80, 640]:
        print(f"  {entry['label_type']:>10s} h={entry['hidden_dim']:>3d}: "
              f"train_acc={entry['train_acc_mean']:.4f} +/- {entry['train_acc_std']:.4f}, "
              f"test_acc={entry['test_acc_mean']:.4f} +/- {entry['test_acc_std']:.4f}")
print("=" * 60)
