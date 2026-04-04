# validate.py
"""Validate ballet-sync experiment results for completeness and correctness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

num_sims = data["metadata"]["num_simulations"]
num_conditions = data["metadata"]["num_conditions"]
records = data["records"]
num_records = len(records)

print(f"Simulations:  {num_sims}")
print(f"Conditions:   {num_conditions}")
print(f"Records:      {num_records} (expected {num_sims})")

errors = []

# Check record count = 1440 (20 K × 4 topologies × 3 N × 2 sigma × 3 seeds)
EXPECTED_COUNT = 1440
if num_records != EXPECTED_COUNT:
    errors.append(
        f"Expected {EXPECTED_COUNT} records, got {num_records}"
    )

# Check all 4 topologies present
expected_topologies = {"all-to-all", "nearest-k", "hierarchical", "ring"}
found_topologies = set(r["topology"] for r in records)
if found_topologies != expected_topologies:
    missing = expected_topologies - found_topologies
    extra = found_topologies - expected_topologies
    if missing:
        errors.append(f"Missing topologies: {missing}")
    if extra:
        errors.append(f"Unexpected topologies: {extra}")

# Check evaluator scores in [0, 1]
ev_names = ["kuramoto_order", "spatial_alignment", "velocity_synchrony", "pairwise_entrainment"]
bad_score_count = 0
for r in records:
    scores = r.get("evaluator_scores", {})
    for ev in ev_names:
        score = scores.get(ev)
        if score is None:
            bad_score_count += 1
        elif not (0.0 <= score <= 1.0):
            bad_score_count += 1
            errors.append(
                f"Evaluator '{ev}' score {score:.4f} out of [0,1] "
                f"for topology={r['topology']} K={r['K']} seed={r['seed']}"
            )
            break
    if len(errors) > 5:
        break

if bad_score_count > 0 and not errors:
    errors.append(f"{bad_score_count} records have missing evaluator scores")

# Check K=0 control: mean kuramoto_order score < 0.3
k0_records = [r for r in records if abs(r["K"]) < 1e-9]
if k0_records:
    mean_k0_score = sum(
        r["evaluator_scores"].get("kuramoto_order", 0.0) for r in k0_records
    ) / len(k0_records)
    print(f"\nK=0 control (n={len(k0_records)} records):")
    print(f"  Mean kuramoto_order score at K=0: {mean_k0_score:.4f} (expected < 0.3)")
    if mean_k0_score >= 0.3:
        errors.append(
            f"K=0 control kuramoto_order mean = {mean_k0_score:.4f} >= 0.3 "
            "(evaluator may be miscalibrated)"
        )
else:
    errors.append("No K=0 records found — cannot check control condition")

# dt convergence check
print("\nRunning dt convergence check...")
try:
    from src.experiment import ExperimentConfig, run_simulation

    cfg_coarse = ExperimentConfig(
        K=1.5, topology="all-to-all", n=12, sigma=0.5, seed=0,
        total_steps=2000, dt=0.01,
    )
    cfg_fine = ExperimentConfig(
        K=1.5, topology="all-to-all", n=12, sigma=0.5, seed=0,
        total_steps=4000, dt=0.005,
    )
    res_coarse = run_simulation(cfg_coarse)
    res_fine = run_simulation(cfg_fine)

    r_coarse = res_coarse.final_r
    r_fine = res_fine.final_r

    rel_diff = abs(r_coarse - r_fine) / max(r_fine, 1e-9)
    print(f"  final_r at dt=0.01:  {r_coarse:.5f}")
    print(f"  final_r at dt=0.005: {r_fine:.5f}")
    print(f"  Relative difference: {rel_diff:.4f} (must be < 0.01)")

    if rel_diff >= 0.01:
        errors.append(
            f"dt convergence FAILED: r_coarse={r_coarse:.5f}, r_fine={r_fine:.5f}, "
            f"rel_diff={rel_diff:.4f} >= 0.01"
        )
    else:
        print("  dt convergence check passed.")

except Exception as exc:
    errors.append(f"dt convergence check raised exception: {exc}")

# Summary
print(f"\nTopologies found: {sorted(found_topologies)}")
print(f"Evaluator names:  {ev_names}")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
