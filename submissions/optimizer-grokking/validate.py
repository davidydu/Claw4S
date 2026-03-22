"""Validate experiment results for completeness and correctness.

Accumulates all errors before reporting, per project conventions.
Must be run from the submission directory (submissions/optimizer-grokking/).
"""

import json
import os
import sys

# Guard: must be run from submission directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: validate.py must be executed from submissions/optimizer-grokking/",
          file=sys.stderr)
    sys.exit(1)

errors = []

# Check results directory exists
if not os.path.isdir("results"):
    print("ERROR: results/ directory not found. Run run.py first.", file=sys.stderr)
    sys.exit(1)

# Check required files
required_files = [
    "results/sweep_results.json",
    "results/grokking_heatmap.png",
    "results/training_curves.png",
    "results/report.md",
]

for f in required_files:
    if not os.path.isfile(f):
        errors.append(f"Missing output file: {f}")

# Load and validate sweep results
try:
    with open("results/sweep_results.json") as f:
        data = json.load(f)
except FileNotFoundError:
    errors.append("Cannot open sweep_results.json")
    data = None
except json.JSONDecodeError as e:
    errors.append(f"Invalid JSON in sweep_results.json: {e}")
    data = None

if data is not None:
    meta = data.get("metadata", {})
    runs = data.get("runs", [])

    # Validate metadata
    num_optimizers = len(meta.get("optimizers", []))
    num_lrs = len(meta.get("learning_rates", []))
    num_wds = len(meta.get("weight_decays", []))
    expected_runs = num_optimizers * num_lrs * num_wds

    print(f"Optimizers:     {num_optimizers} ({', '.join(meta.get('optimizers', []))})")
    print(f"Learning rates: {num_lrs} ({meta.get('learning_rates', [])})")
    print(f"Weight decays:  {num_wds} ({meta.get('weight_decays', [])})")
    print(f"Total runs:     {len(runs)} (expected {expected_runs})")
    print(f"Runtime:        {meta.get('total_seconds', '?')}s")

    if num_optimizers < 4:
        errors.append(f"Expected >= 4 optimizers, got {num_optimizers}")
    if num_lrs < 3:
        errors.append(f"Expected >= 3 learning rates, got {num_lrs}")
    if num_wds < 3:
        errors.append(f"Expected >= 3 weight decays, got {num_wds}")
    if len(runs) != expected_runs:
        errors.append(f"Expected {expected_runs} runs, got {len(runs)}")

    # Validate each run
    valid_outcomes = {"grokking", "memorization", "failure"}
    outcome_counts = {"grokking": 0, "memorization": 0, "failure": 0}

    for i, run in enumerate(runs):
        outcome = run.get("outcome", "")
        if outcome not in valid_outcomes:
            errors.append(f"Run {i}: invalid outcome '{outcome}'")
        else:
            outcome_counts[outcome] += 1

        if "history" not in run or len(run["history"]) == 0:
            errors.append(f"Run {i}: empty training history")

        train_acc = run.get("final_train_acc", -1)
        test_acc = run.get("final_test_acc", -1)
        if not (0.0 <= train_acc <= 1.0):
            errors.append(f"Run {i}: train_acc={train_acc} out of [0,1]")
        if not (0.0 <= test_acc <= 1.0):
            errors.append(f"Run {i}: test_acc={test_acc} out of [0,1]")

        # Consistency check: grokking must have both memorization and grokking epochs
        if outcome == "grokking":
            if run.get("memorization_epoch") is None:
                errors.append(f"Run {i}: grokking outcome but no memorization_epoch")
            if run.get("grokking_epoch") is None:
                errors.append(f"Run {i}: grokking outcome but no grokking_epoch")

    print(f"\nOutcome distribution:")
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count}")

    # At least one grokking run should exist
    if outcome_counts["grokking"] == 0:
        errors.append("No grokking runs detected -- experiment may need tuning")

    # Seed check
    if meta.get("seed") != 42:
        errors.append(f"Expected seed=42, got {meta.get('seed')}")

    if meta.get("prime") != 97:
        errors.append(f"Expected prime=97, got {meta.get('prime')}")

# Report results
print()
if errors:
    print(f"Validation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
    sys.exit(0)
