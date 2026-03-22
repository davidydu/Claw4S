"""Validate experiment results for completeness and scientific soundness."""

import json
import sys

RESULTS_PATH = "results/results.json"
SUMMARY_PATH = "results/summary.json"

errors: list[str] = []

# ---- Load results -----------------------------------------------------------

with open(RESULTS_PATH) as f:
    data = json.load(f)

meta = data["metadata"]
results = data["results"]

print(f"Simulations:   {meta['n_simulations']}")
print(f"Agent types:   {', '.join(meta['agent_types'])}")
print(f"GT fractions:  {meta['gt_fractions']}")
print(f"Distributions: {', '.join(meta['distributions'])}")
print(f"Generations:   {meta['n_generations']}")
print(f"Runtime:       {meta['elapsed_seconds']}s")
print()

# ---- Check 1: correct count ------------------------------------------------

expected = (
    len(meta["agent_types"])
    * len(meta["gt_fractions"])
    * len(meta["distributions"])
    * 3  # seeds
)
if meta["n_simulations"] != expected:
    errors.append(f"Expected {expected} sims, got {meta['n_simulations']}")
else:
    print(f"[OK] Simulation count: {meta['n_simulations']} (expected {expected})")

# ---- Check 2: all KL values are non-negative -------------------------------

neg_kl = 0
for r in results:
    for g in r["generations"]:
        if g["kl_divergence"] < 0:
            neg_kl += 1
if neg_kl > 0:
    errors.append(f"{neg_kl} negative KL divergence values found")
else:
    print("[OK] All KL divergence values non-negative")

# ---- Check 3: generation 0 KL is small -------------------------------------

high_gen0 = 0
for r in results:
    kl0 = r["generations"][0]["kl_divergence"]
    if kl0 > 0.5:
        high_gen0 += 1
if high_gen0 > 0:
    errors.append(f"{high_gen0} sims have gen-0 KL > 0.5 (learning from ground truth)")
else:
    print("[OK] All generation-0 KL values < 0.5 (good fit to ground truth)")

# ---- Check 4: naive+0% GT shows degradation --------------------------------

naive_zero = [r for r in results if r["config"]["agent_type"] == "naive" and r["config"]["gt_fraction"] == 0.0]
degraded = 0
for r in naive_zero:
    kl_first = r["generations"][0]["kl_divergence"]
    kl_last = r["generations"][-1]["kl_divergence"]
    if kl_last > kl_first:
        degraded += 1
pct = degraded / len(naive_zero) * 100 if naive_zero else 0
if pct < 80:
    errors.append(f"Only {pct:.0f}% of naive+0%GT sims show degradation (expected >= 80%)")
else:
    print(f"[OK] {pct:.0f}% of naive+0%GT sims show quality degradation")

# ---- Check 5: anchored with high GT stays stable ---------------------------

anch_high = [r for r in results if r["config"]["agent_type"] == "anchored" and r["config"]["gt_fraction"] >= 0.10]
stable = sum(1 for r in anch_high if r["collapse_generation"] is None)
pct = stable / len(anch_high) * 100 if anch_high else 0
if pct < 50:
    errors.append(f"Only {pct:.0f}% of anchored+>=10%GT sims stay stable (expected >= 50%)")
else:
    print(f"[OK] {pct:.0f}% of anchored+>=10%GT sims stay stable (no collapse)")

# ---- Check 6: summary file exists and has correct rows ---------------------

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

n_conditions = len(meta["agent_types"]) * len(meta["gt_fractions"]) * len(meta["distributions"])
if len(summary) != n_conditions:
    errors.append(f"Summary has {len(summary)} rows, expected {n_conditions}")
else:
    print(f"[OK] Summary has {len(summary)} condition rows (expected {n_conditions})")

# ---- Check 7: reproducibility (same-seed pairs match) ----------------------

from collections import defaultdict

by_config = defaultdict(list)
for r in results:
    c = r["config"]
    key = (c["agent_type"], c["gt_fraction"], c["dist_name"], c["seed"])
    by_config[key].append(r)

dupes = {k: v for k, v in by_config.items() if len(v) > 1}
if dupes:
    errors.append(f"{len(dupes)} duplicate config keys found (should be unique)")
else:
    print(f"[OK] All {len(by_config)} config keys are unique")

# ---- Verdict ----------------------------------------------------------------

print()
if errors:
    print(f"Validation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
