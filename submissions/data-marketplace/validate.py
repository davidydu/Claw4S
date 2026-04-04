"""Validate experiment results for completeness and correctness."""

import json
import sys

with open("results/results.json") as f:
    data = json.load(f)

n_sims = data["metadata"]["n_simulations"]
runtime = data["metadata"]["runtime_seconds"]
n_raw = len(data["raw_results"])
n_findings = len(data["findings"])
n_groups = data["aggregated"]["n_groups"]

print(f"Simulations: {n_raw} (expected {n_sims})")
print(f"Unique configs: {n_groups}")
print(f"Key findings: {n_findings}")
print(f"Runtime: {runtime:.1f}s")

errors = []

# 1. Correct count
if n_raw != n_sims:
    errors.append(f"Expected {n_sims} results, got {n_raw}")

# 2. Expected 162 simulations (6 compositions x 3 sizes x 3 regimes x 3 seeds)
if n_sims != 162:
    errors.append(f"Expected 162 simulations, got {n_sims}")

# 3. All results have required fields
for i, r in enumerate(data["raw_results"]):
    for field in ["config", "metrics", "audit_scores", "buyer_welfare", "buyer_surplus", "seller_profit"]:
        if field not in r:
            errors.append(f"Result {i} missing field: {field}")
            break

# 4. Metrics are in valid ranges
for r in data["raw_results"]:
    m = r["metrics"]
    if not (0 <= m["market_efficiency"] <= 1.0 + 1e-6):
        errors.append(f"Market efficiency out of range: {m['market_efficiency']}")
    if not (0 <= m["lemons_index"] <= 1.0 + 1e-6):
        errors.append(f"Lemons index out of range: {m['lemons_index']}")

# 5. Lemons effect validation: all-predatory should have high lemons index
pred_results = [r for r in data["raw_results"] if r["config"]["composition"] == "all_predatory"]
if pred_results:
    avg_lemons = sum(r["metrics"]["lemons_index"] for r in pred_results) / len(pred_results)
    print(f"\nAll-predatory avg lemons index: {avg_lemons:.3f}")
    if avg_lemons < 0.8:
        errors.append(f"All-predatory markets should have high lemons index, got {avg_lemons:.3f}")

# 6. Honest markets should have low lemons index
honest_results = [r for r in data["raw_results"] if r["config"]["composition"] == "all_honest"]
if honest_results:
    avg_lemons = sum(r["metrics"]["lemons_index"] for r in honest_results) / len(honest_results)
    print(f"All-honest avg lemons index: {avg_lemons:.3f}")
    if avg_lemons > 0.1:
        errors.append(f"All-honest markets should have low lemons index, got {avg_lemons:.3f}")

# 7. Reproducibility: same config+seed should give same results
from collections import defaultdict
by_name = defaultdict(list)
for r in data["raw_results"]:
    c = r["config"]
    name = f"{c['composition']}__{c['market_size']}__{c['info_regime']}__seed{c['seed']}"
    by_name[name].append(r)

dupes = {k: v for k, v in by_name.items() if len(v) > 1}
if dupes:
    errors.append(f"Found {len(dupes)} duplicate config names (should be unique)")

# 8. Key findings exist
if n_findings < 3:
    errors.append(f"Expected at least 3 key findings, got {n_findings}")

# 9. Figures exist
import os
expected_figures = [
    "efficiency_by_composition.png",
    "lemons_by_composition.png",
    "buyer_surplus.png",
    "audit_heatmap.png",
]
for fig in expected_figures:
    path = os.path.join("results", fig)
    if not os.path.exists(path):
        errors.append(f"Missing figure: {fig}")
    else:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {fig}: {size_kb:.0f} KB")

# 10. Report exists
if not os.path.exists("results/report.md"):
    errors.append("Missing report.md")

print(f"\n{'='*40}")
if errors:
    print(f"VALIDATION FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("VALIDATION PASSED")
    print(f"  {n_raw} simulations, {n_findings} findings, {len(expected_figures)} figures")
    sys.exit(0)
