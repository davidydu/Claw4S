"""Validate experiment results for completeness and scientific soundness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

meta = data["metadata"]
n_configs = meta["total_configs"]
n_raw = len(data["raw_results"])
n_summaries = len(data["summaries"])
n_derived = len(data["derived_metrics"])

print(f"Configurations: {n_raw} (expected {n_configs})")
print(f"Summary groups: {n_summaries}")
print(f"Derived metrics: {n_derived}")
print(f"Amplifications: {len(data['amplifications'])}")

errors = []

# Check raw count
if n_raw != n_configs:
    errors.append(f"Expected {n_configs} raw results, got {n_raw}")

# Check expected summary count: 3 honest x 3 byz x 5 frac x 3 sizes = 135
expected_summaries = 3 * 3 * 5 * 3
if n_summaries != expected_summaries:
    errors.append(f"Expected {expected_summaries} summary groups, got {n_summaries}")

# Check derived metrics: 3 honest x 3 byz x 3 sizes = 27
expected_derived = 3 * 3 * 3
if n_derived != expected_derived:
    errors.append(f"Expected {expected_derived} derived metrics, got {n_derived}")

# Sanity: no-Byzantine baseline accuracy should be > 60%
baselines = [r for r in data["raw_results"] if r["byzantine_fraction"] == 0.0]
if not baselines:
    errors.append("No baseline (f=0) results found")
else:
    avg_baseline = sum(r["accuracy"] for r in baselines) / len(baselines)
    print(f"Baseline accuracy (f=0): {avg_baseline:.3f}")
    if avg_baseline < 0.50:
        errors.append(f"Baseline accuracy {avg_baseline:.3f} unexpectedly low (< 0.50)")

# Sanity: accuracy should generally decrease with higher Byzantine fraction
for s in data["summaries"]:
    if s["mean_accuracy"] < 0.0 or s["mean_accuracy"] > 1.0:
        errors.append(f"Accuracy {s['mean_accuracy']} out of [0,1] range")

# Check Byzantine thresholds are in valid range
for d in data["derived_metrics"]:
    t = d["byzantine_threshold_50"]
    if t < 0.0 or t > 1.0:
        errors.append(f"Byzantine threshold {t} out of [0,1] range for {d}")

# Check resilience scores are in valid range
for d in data["derived_metrics"]:
    r = d["resilience_score"]
    if r < 0.0 or r > 1.0:
        errors.append(f"Resilience score {r} out of [0,1] range for {d}")

# Check reproducibility: same config + same seed = same accuracy
from collections import defaultdict
repro_groups = defaultdict(list)
for r in data["raw_results"]:
    key = (r["honest_type"], r["byzantine_type"], r["byzantine_fraction"],
           r["committee_size"], r["seed"])
    repro_groups[key].append(r["accuracy"])
for key, accs in repro_groups.items():
    if len(accs) > 1:
        errors.append(f"Duplicate config-seed combo: {key}")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
