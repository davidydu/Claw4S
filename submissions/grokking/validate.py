"""Validate grokking phase diagram results for completeness and correctness."""

import json
import os
import sys

errors = []


def check(condition: bool, message: str) -> None:
    """Record an error if condition is False."""
    if not condition:
        errors.append(message)


# Check result files exist
print("Checking result files...")
required_files = [
    "results/sweep_results.json",
    "results/phase_diagram.json",
    "results/report.md",
    "results/phase_diagram_h16.png",
    "results/phase_diagram_h32.png",
    "results/phase_diagram_h64.png",
    "results/grokking_curves.png",
]

for f in required_files:
    exists = os.path.isfile(f)
    check(exists, f"Missing file: {f}")
    print(f"  {'OK' if exists else 'MISSING'}: {f}")

# Validate sweep results
print("\nValidating sweep results...")
try:
    with open("results/sweep_results.json") as f:
        sweep = json.load(f)

    n_runs = len(sweep)
    print(f"  Total runs: {n_runs}")
    check(n_runs == 60, f"Expected 60 runs, got {n_runs}")

    # Check all required config fields
    for r in sweep:
        check("config" in r, "Missing 'config' in result")
        check("metrics" in r, "Missing 'metrics' in result")
        check("phase" in r, "Missing 'phase' in result")

    # Check phase values
    valid_phases = {"confusion", "memorization", "grokking", "comprehension"}
    phases = [r["phase"] for r in sweep]
    for p in phases:
        check(p in valid_phases, f"Invalid phase: {p}")

    # Count phases
    from collections import Counter
    phase_counts = Counter(phases)
    print(f"  Phase distribution: {dict(phase_counts)}")

    # Check grid coverage
    weight_decays = sorted(set(r["config"]["weight_decay"] for r in sweep))
    fractions = sorted(set(r["config"]["train_fraction"] for r in sweep))
    hidden_dims = sorted(set(r["config"]["hidden_dim"] for r in sweep))
    print(f"  Weight decays: {weight_decays}")
    print(f"  Dataset fractions: {fractions}")
    print(f"  Hidden dims: {hidden_dims}")
    check(len(weight_decays) == 5, f"Expected 5 weight decay values, got {len(weight_decays)}")
    check(len(fractions) == 4, f"Expected 4 fraction values, got {len(fractions)}")
    check(len(hidden_dims) >= 3, f"Expected >= 3 hidden dim values, got {len(hidden_dims)}")

    # Check accuracy ranges
    for r in sweep:
        m = r["metrics"]
        check(
            0 <= m["final_train_acc"] <= 1.0,
            f"Train acc out of range: {m['final_train_acc']}",
        )
        check(
            0 <= m["final_test_acc"] <= 1.0,
            f"Test acc out of range: {m['final_test_acc']}",
        )

    # Check param counts under 100K
    for r in sweep:
        pc = r["config"]["param_count"]
        check(pc < 100_000, f"Param count {pc} exceeds 100K")

except FileNotFoundError:
    errors.append("Could not open results/sweep_results.json")
except json.JSONDecodeError as e:
    errors.append(f"Invalid JSON in sweep_results.json: {e}")

# Validate phase diagram summary
print("\nValidating phase summary...")
try:
    with open("results/phase_diagram.json") as f:
        summary = json.load(f)

    check("phase_counts" in summary, "Missing 'phase_counts' in summary")
    check("total_runs" in summary, "Missing 'total_runs' in summary")
    check("grokking_fraction" in summary, "Missing 'grokking_fraction' in summary")
    check(
        summary.get("total_runs") == 60,
        f"Expected 60 total_runs, got {summary.get('total_runs')}",
    )
    print(f"  Grokking fraction: {summary.get('grokking_fraction', 0):.1%}")

except FileNotFoundError:
    errors.append("Could not open results/phase_diagram.json")
except json.JSONDecodeError as e:
    errors.append(f"Invalid JSON in phase_diagram.json: {e}")

# Report files (check non-empty)
print("\nChecking file sizes...")
for f in required_files:
    if os.path.isfile(f):
        size = os.path.getsize(f)
        check(size > 0, f"File is empty: {f}")
        print(f"  {f}: {size:,} bytes")

# Final result
print()
if errors:
    print(f"Validation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
