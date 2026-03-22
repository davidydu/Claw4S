"""Validate experiment results for completeness and correctness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

meta = data["metadata"]
results = data["results"]
n_sims = meta["n_simulations"]
n_results = len(results)
expected = (
    len(meta["compositions"])
    * len(meta["competition_levels"])
    * len(meta["complementarity_levels"])
    * len(meta["seeds"])
)

print(f"Compositions:    {len(meta['compositions'])}")
print(f"Competition lvl: {len(meta['competition_levels'])}")
print(f"Complementarity: {len(meta['complementarity_levels'])}")
print(f"Seeds:           {len(meta['seeds'])}")
print(f"Simulations:     {n_results} (expected {expected})")

errors = []

if n_results != expected:
    errors.append(f"Expected {expected} simulations, got {n_results}")

# Check each result has required fields
for i, r in enumerate(results):
    if "summary" not in r:
        errors.append(f"Result {i} missing 'summary'")
        continue
    s = r["summary"]
    if not (0.0 <= s["avg_sharing_rate"] <= 1.0):
        errors.append(f"Result {i}: avg_sharing_rate {s['avg_sharing_rate']:.3f} out of [0,1]")
    if s["avg_group_welfare"] == 0.0 and r["config"]["composition"] != ["secretive"] * 4:
        errors.append(f"Result {i}: suspicious zero group welfare")

# Check sharing behavior makes sense
open_sharing = [
    r["summary"]["tail_sharing_rate"]
    for r in results
    if r["labels"]["composition"] == "all_open"
]
secretive_sharing = [
    r["summary"]["tail_sharing_rate"]
    for r in results
    if r["labels"]["composition"] == "all_secretive"
]

if open_sharing:
    avg_open = sum(open_sharing) / len(open_sharing)
    print(f"\nAll-open tail sharing:      {avg_open:.3f} (expected ~1.0)")
    if avg_open < 0.9:
        errors.append(f"All-open sharing {avg_open:.3f} too low (expected ~1.0)")

if secretive_sharing:
    avg_sec = sum(secretive_sharing) / len(secretive_sharing)
    print(f"All-secretive tail sharing: {avg_sec:.3f} (expected ~0.0)")
    if avg_sec > 0.1:
        errors.append(f"All-secretive sharing {avg_sec:.3f} too high (expected ~0.0)")

# Check analysis file exists
try:
    with open("results/analysis.json") as f:
        analysis = json.load(f)
    print(f"\nAnalysis file: OK ({len(analysis['aggregated'])} conditions)")
    print(f"Agent rankings: {list(analysis['agent_rankings'].keys())}")
except FileNotFoundError:
    errors.append("results/analysis.json not found")

# Check report exists
try:
    with open("results/report.md") as f:
        report = f.read()
    print(f"Report file: OK ({len(report)} chars)")
except FileNotFoundError:
    errors.append("results/report.md not found")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
