"""Validate experiment results for completeness and correctness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

n_sims = data["metadata"]["n_simulations"]
n_results = len(data["results"])

print(f"Simulations: {n_sims}")
print(f"Result entries: {n_results}")

errors = []

# Check total count: 3 hackers x 3 topologies x 3 detectabilities x 4 monitors x 3 seeds = 324
if n_sims != 324:
    errors.append(f"Expected 324 simulations, got {n_sims}")
if n_results != 324:
    errors.append(f"Expected 324 result entries, got {n_results}")

# Check all parameter combinations are present
topologies = set()
detectabilities = set()
monitor_fracs = set()
hacker_counts = set()
seeds = set()

for r in data["results"]:
    p = r["params"]
    topologies.add(p["topology"])
    detectabilities.add(p["detectability"])
    monitor_fracs.add(p["monitor_fraction"])
    hacker_counts.add(p["n_initial_hackers"])
    seeds.add(p["seed"])

if topologies != {"grid", "random", "star"}:
    errors.append(f"Missing topologies: expected grid/random/star, got {topologies}")
if detectabilities != {"obvious", "subtle", "invisible"}:
    errors.append(f"Missing detectabilities: {detectabilities}")
if monitor_fracs != {0.0, 0.1, 0.25, 0.5}:
    errors.append(f"Missing monitor fractions: {monitor_fracs}")
if hacker_counts != {1, 2, 5}:
    errors.append(f"Missing hacker counts: {hacker_counts}")
if seeds != {42, 123, 7}:
    errors.append(f"Missing seeds: {seeds}")

# Sanity check: metrics are in valid ranges
for r in data["results"]:
    m = r["metrics"]
    if not (0.0 <= m["steady_state_adoption"] <= 1.0):
        errors.append(f"Invalid adoption {m['steady_state_adoption']} for {r['params']}")
    if not (0.0 <= m["final_adoption"] <= 1.0):
        errors.append(f"Invalid final_adoption {m['final_adoption']} for {r['params']}")
    if m["steady_state_divergence"] < 0:
        errors.append(f"Negative divergence {m['steady_state_divergence']} for {r['params']}")

# Check that monitors actually help (adoption with 50% monitors < without)
no_mon = [r for r in data["results"] if r["params"]["monitor_fraction"] == 0.0
          and r["params"]["detectability"] == "obvious"]
hi_mon = [r for r in data["results"] if r["params"]["monitor_fraction"] == 0.5
          and r["params"]["detectability"] == "obvious"]

avg_no = sum(r["metrics"]["steady_state_adoption"] for r in no_mon) / max(len(no_mon), 1)
avg_hi = sum(r["metrics"]["steady_state_adoption"] for r in hi_mon) / max(len(hi_mon), 1)
print(f"\nAdoption without monitors (obvious): {avg_no:.2f}")
print(f"Adoption with 50% monitors (obvious): {avg_hi:.2f}")

if avg_hi >= avg_no and len(hi_mon) > 0:
    errors.append(
        f"Monitors should reduce adoption: no_mon={avg_no:.3f} vs hi_mon={avg_hi:.3f}"
    )

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
