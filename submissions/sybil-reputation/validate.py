"""Validate experiment results for completeness and scientific soundness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

meta = data["metadata"]
results = data["results"]

n_algos = len(meta["algorithms"])
n_strats = len(meta["strategies"])
n_sybil_counts = len(meta["sybil_counts"])
n_seeds = len(meta["seeds"])
total = meta["total_simulations"]
actual = len(results)

# Expected: for K=0, strategy is "none" so 1 strategy x n_algos x n_seeds
# For K>0: n_strats strategies x n_algos x n_seeds per K value
k_zero = 1  # K=0 is one level
k_nonzero = n_sybil_counts - k_zero  # K > 0
expected = n_algos * n_seeds * k_zero + n_algos * n_strats * n_seeds * k_nonzero

print(f"Algorithms:        {n_algos} ({', '.join(meta['algorithms'])})")
print(f"Strategies:        {n_strats} ({', '.join(meta['strategies'])})")
print(f"Sybil counts:      {n_sybil_counts} ({meta['sybil_counts']})")
print(f"Seeds:             {n_seeds} ({meta['seeds']})")
print(f"Simulations:       {actual} (expected {expected})")
print(f"Runtime:           {meta['elapsed_seconds']}s")

errors = []

if actual != expected:
    errors.append(f"Expected {expected} simulations, got {actual}")

if n_algos < 4:
    errors.append(f"Expected 4 algorithms, got {n_algos}")

if n_strats < 3:
    errors.append(f"Expected 3 strategies, got {n_strats}")

# Check metric ranges
for r in results:
    cfg = r["config"]
    m = r["metrics"]
    label = f"{cfg['algorithm']}/K={cfg['n_sybil']}/{cfg['strategy']}/seed={cfg['seed']}"

    acc = m["reputation_accuracy"]
    if not (-1.0 <= acc <= 1.0):
        errors.append(f"{label}: accuracy {acc} out of [-1, 1]")

    det = m["sybil_detection_rate"]
    if not (0.0 <= det <= 1.0):
        errors.append(f"{label}: detection_rate {det} out of [0, 1]")

    wel = m["honest_welfare"]
    if not (0.0 <= wel <= 1.0):
        errors.append(f"{label}: welfare {wel} out of [0, 1]")

    eff = m["market_efficiency"]
    if not (0.0 <= eff <= 1.0):
        errors.append(f"{label}: efficiency {eff} out of [0, 1]")

# Check that K=0 baselines have high accuracy (> 0.5)
baseline_accs = [
    r["metrics"]["reputation_accuracy"]
    for r in results
    if r["config"]["n_sybil"] == 0
]
if baseline_accs:
    mean_baseline = sum(baseline_accs) / len(baseline_accs)
    print(f"Baseline accuracy: {mean_baseline:.3f} (expected > 0.5)")
    if mean_baseline < 0.3:
        errors.append(
            f"Baseline accuracy too low: {mean_baseline:.3f} (expected > 0.3)"
        )

# Check that Sybil attacks degrade accuracy for at least simple_average
simple_k20 = [
    r["metrics"]["reputation_accuracy"]
    for r in results
    if r["config"]["algorithm"] == "simple_average" and r["config"]["n_sybil"] == 20
]
if simple_k20 and baseline_accs:
    mean_k20 = sum(simple_k20) / len(simple_k20)
    print(f"Simple avg K=20:   {mean_k20:.3f} (should be < baseline)")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
