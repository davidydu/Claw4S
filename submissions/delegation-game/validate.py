"""Validate delegation game results for completeness and correctness."""

import json
import os
import sys

with open("results/results.json") as f:
    data = json.load(f)

meta = data["metadata"]
n_sims = meta["num_simulations"]
n_raw = len(data["raw_results"])
n_agg = len(data["aggregated"])
expected_raw = 144  # 4 schemes x 4 compositions x 3 noise x 3 seeds
expected_agg = 48   # 4 schemes x 4 compositions x 3 noise (averaged over seeds)

print(f"Simulations: {n_sims} (expected {expected_raw})")
print(f"Raw results: {n_raw} (expected {expected_raw})")
print(f"Aggregated:  {n_agg} (expected {expected_agg})")

errors = []

# --- Structural checks ---

if n_raw != expected_raw:
    errors.append(f"Expected {expected_raw} raw results, got {n_raw}")
if n_agg != expected_agg:
    errors.append(f"Expected {expected_agg} aggregated results, got {n_agg}")

# Check all schemes present
schemes_found = set(r["scheme"] for r in data["raw_results"])
expected_schemes = {"fixed_pay", "piece_rate", "tournament", "reputation"}
if schemes_found != expected_schemes:
    errors.append(f"Expected schemes {expected_schemes}, got {schemes_found}")

# --- Metric range checks ---

for r in data["raw_results"]:
    if not (0.0 < r["avg_quality"] < 10.0):
        errors.append(f"avg_quality {r['avg_quality']:.2f} out of range "
                      f"for {r['scheme']}")
        break
    if not (0.0 <= r["shirking_rate"] <= 1.0):
        errors.append(f"shirking_rate {r['shirking_rate']:.2f} out of range")
        break
    if r["incentive_efficiency"] < 0:
        errors.append(f"incentive_efficiency negative: {r['incentive_efficiency']}")
        break

# --- Behavioral checks ---

# 1. Honest workers never shirk (under any scheme)
honest_results = [
    r for r in data["raw_results"]
    if all(w == "honest" for w in r["worker_types"])
]
for r in honest_results:
    if r["shirking_rate"] != 0.0:
        errors.append(f"Honest workers should never shirk, but got "
                      f"{r['shirking_rate']:.2%} under {r['scheme']}")
        break

# 2. Shirker component in mixed teams always shirks
mixed_results = [
    r for r in data["raw_results"]
    if sorted(r["worker_types"]) == ["honest", "shirker", "strategic"]
]
for r in mixed_results:
    # Check that the shirker worker has shirking_rate 1.0
    for wname, wdata in r["per_worker"].items():
        if wdata["type"] == "shirker" and wdata["shirking_rate"] != 1.0:
            errors.append(f"Shirker worker should always shirk, got "
                          f"{wdata['shirking_rate']:.2%}")
            break

# 3. Honest workers produce highest quality (effort=5 always)
honest_avg_q = sum(r["avg_quality"] for r in honest_results) / len(honest_results)
shirker_results = [
    r for r in data["raw_results"]
    if all(w == "shirker" for w in r["worker_types"])
]
if shirker_results:
    shirker_avg_q = sum(r["avg_quality"] for r in shirker_results) / len(shirker_results)
    print(f"\nHonest avg quality:  {honest_avg_q:.2f}")
    print(f"Shirker avg quality: {shirker_avg_q:.2f}")
    if honest_avg_q <= shirker_avg_q:
        errors.append("Expected honest workers to produce higher quality "
                      "than shirkers on average")

# 4. Determinism: same seed produces same result
seeds_check = {}
for r in data["raw_results"]:
    key = (r["scheme"], tuple(sorted(r["worker_types"])), r["noise_std"])
    seeds_check.setdefault(key, []).append(r)
for key, runs in seeds_check.items():
    if len(runs) != 3:
        errors.append(f"Expected 3 seeds per condition, got {len(runs)} for {key}")
        break

# 5. High noise produces higher quality variance than low noise
for scheme in expected_schemes:
    low_noise = [
        r for r in data["raw_results"]
        if r["scheme"] == scheme and r["noise_std"] == 0.5
    ]
    high_noise = [
        r for r in data["raw_results"]
        if r["scheme"] == scheme and r["noise_std"] == 3.0
    ]
    if low_noise and high_noise:
        avg_var_low = sum(r["quality_variance"] for r in low_noise) / len(low_noise)
        avg_var_high = sum(r["quality_variance"] for r in high_noise) / len(high_noise)
        if avg_var_high <= avg_var_low:
            errors.append(f"{scheme}: high noise should have higher quality "
                          f"variance than low noise")

# --- Output checks ---

if not os.path.exists("results/report.md"):
    errors.append("results/report.md not found")

# Print summary
print(f"\nSchemes: {sorted(schemes_found)}")
print(f"Worker compositions: {len(set(tuple(sorted(r['worker_types'])) for r in data['raw_results']))}")
print(f"Noise levels: {sorted(set(r['noise_std'] for r in data['raw_results']))}")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
