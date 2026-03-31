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
    print(f"Completed runs: {meta.get('completed_runs', '?')}")

    if num_optimizers < 4:
        errors.append(f"Expected >= 4 optimizers, got {num_optimizers}")
    if num_lrs < 3:
        errors.append(f"Expected >= 3 learning rates, got {num_lrs}")
    if num_wds < 3:
        errors.append(f"Expected >= 3 weight decays, got {num_wds}")
    if len(runs) != expected_runs:
        errors.append(f"Expected {expected_runs} runs, got {len(runs)}")
    if meta.get("completed_runs") not in (None, len(runs)):
        errors.append(
            f"Metadata completed_runs={meta.get('completed_runs')} "
            f"does not match run count {len(runs)}"
        )

    # Reproducibility metadata should be present.
    for key in ["python_version", "torch_version", "numpy_version", "platform", "generated_utc"]:
        value = meta.get(key)
        if not value:
            errors.append(f"Missing metadata field: {key}")

    # Validate split metadata if available.
    train_examples = meta.get("train_examples")
    test_examples = meta.get("test_examples")
    if isinstance(train_examples, int) and isinstance(test_examples, int):
        if train_examples + test_examples != (meta.get("prime", 0) ** 2):
            errors.append(
                f"train_examples + test_examples should equal p^2 "
                f"(got {train_examples}+{test_examples} vs p={meta.get('prime')})"
            )

    # Validate each run
    valid_outcomes = {
        "grokking",
        "direct_generalization",
        "memorization",
        "failure",
    }
    outcome_counts = {
        "grokking": 0,
        "direct_generalization": 0,
        "memorization": 0,
        "failure": 0,
    }
    expected_configs = {
        (opt, float(lr), float(wd))
        for opt in meta.get("optimizers", [])
        for lr in meta.get("learning_rates", [])
        for wd in meta.get("weight_decays", [])
    }
    seen_configs = set()
    max_epochs = meta.get("max_epochs")

    for i, run in enumerate(runs):
        optimizer = run.get("optimizer")
        lr = run.get("lr")
        wd = run.get("weight_decay")
        try:
            config = (optimizer, float(lr), float(wd))
        except (TypeError, ValueError):
            errors.append(
                f"Run {i}: invalid optimizer/lr/wd triple "
                f"({optimizer!r}, {lr!r}, {wd!r})"
            )
            config = None

        if config is not None:
            if config in seen_configs:
                errors.append(f"Run {i}: duplicate config {config}")
            seen_configs.add(config)
            if config not in expected_configs:
                errors.append(f"Run {i}: unexpected config {config}")

        outcome = run.get("outcome", "")
        if outcome not in valid_outcomes:
            errors.append(f"Run {i}: invalid outcome '{outcome}'")
        else:
            outcome_counts[outcome] += 1

        if "history" not in run or len(run["history"]) == 0:
            errors.append(f"Run {i}: empty training history")
            history = []
        else:
            history = run["history"]

        # History integrity checks
        epochs = [entry.get("epoch") for entry in history if "epoch" in entry]
        if len(epochs) != len(history):
            errors.append(f"Run {i}: missing epoch in history entries")
        if epochs:
            if epochs != sorted(epochs):
                errors.append(f"Run {i}: history epochs are not sorted")
            if len(set(epochs)) != len(epochs):
                errors.append(f"Run {i}: duplicated epochs in history")
            if epochs[0] != 1:
                errors.append(f"Run {i}: first logged epoch should be 1 (got {epochs[0]})")
            if isinstance(max_epochs, int) and epochs[-1] != max_epochs:
                errors.append(
                    f"Run {i}: last logged epoch should be max_epochs={max_epochs} "
                    f"(got {epochs[-1]})"
                )

        train_acc = run.get("final_train_acc", -1)
        test_acc = run.get("final_test_acc", -1)
        if not (0.0 <= train_acc <= 1.0):
            errors.append(f"Run {i}: train_acc={train_acc} out of [0,1]")
        if not (0.0 <= test_acc <= 1.0):
            errors.append(f"Run {i}: test_acc={test_acc} out of [0,1]")

        if history:
            last = history[-1]
            last_train = last.get("train_acc")
            last_test = last.get("test_acc")
            if last_train is not None and abs(last_train - train_acc) > 1e-8:
                errors.append(
                    f"Run {i}: final_train_acc {train_acc} "
                    f"!= last history train_acc {last_train}"
                )
            if last_test is not None and abs(last_test - test_acc) > 1e-8:
                errors.append(
                    f"Run {i}: final_test_acc {test_acc} "
                    f"!= last history test_acc {last_test}"
                )

        generalization_epoch = run.get("generalization_epoch")
        memorization_epoch = run.get("memorization_epoch")
        grokking_epoch = run.get("grokking_epoch")

        # Consistency check: grokking must be delayed relative to memorization
        if outcome == "grokking":
            if memorization_epoch is None:
                errors.append(f"Run {i}: grokking outcome but no memorization_epoch")
            if generalization_epoch is None:
                errors.append(f"Run {i}: grokking outcome but no generalization_epoch")
            if grokking_epoch is None:
                errors.append(f"Run {i}: grokking outcome but no grokking_epoch")
            if (memorization_epoch is not None and generalization_epoch is not None
                    and generalization_epoch <= memorization_epoch):
                errors.append(
                    f"Run {i}: grokking must have generalization after memorization "
                    f"(got mem={memorization_epoch}, gen={generalization_epoch})"
                )

        if outcome == "direct_generalization":
            if memorization_epoch is None:
                errors.append(f"Run {i}: direct_generalization outcome but no memorization_epoch")
            if generalization_epoch is None:
                errors.append(
                    f"Run {i}: direct_generalization outcome but no generalization_epoch"
                )
            if grokking_epoch is not None:
                errors.append(
                    f"Run {i}: direct_generalization should not have grokking_epoch "
                    f"(got {grokking_epoch})"
                )
            if (memorization_epoch is not None and generalization_epoch is not None
                    and generalization_epoch > memorization_epoch):
                errors.append(
                    f"Run {i}: direct_generalization should not lag memorization "
                    f"(got mem={memorization_epoch}, gen={generalization_epoch})"
                )

        if outcome == "memorization":
            if memorization_epoch is None:
                errors.append(f"Run {i}: memorization outcome but no memorization_epoch")
            if generalization_epoch is not None:
                errors.append(
                    f"Run {i}: memorization outcome should not have generalization_epoch "
                    f"(got {generalization_epoch})"
                )
            if grokking_epoch is not None:
                errors.append(
                    f"Run {i}: memorization outcome should not have grokking_epoch "
                    f"(got {grokking_epoch})"
                )

        if outcome == "failure":
            if any(v is not None for v in [memorization_epoch, generalization_epoch, grokking_epoch]):
                errors.append(
                    f"Run {i}: failure outcome should not have memorization/generalization epochs"
                )

    print(f"\nOutcome distribution:")
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count}")
    print(f"Unique configs: {len(seen_configs)} (expected {expected_runs})")

    missing_configs = sorted(expected_configs - seen_configs)
    if missing_configs:
        errors.append(f"Missing {len(missing_configs)} configuration(s): {missing_configs}")

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
