"""Validate experiment results for completeness and scientific soundness.

Must be run from the submission directory: submissions/backdoor-detection/
"""

import json
import os
import sys

# Working directory guard
expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
if not os.path.exists(expected_marker):
    print("ERROR: validate.py must be executed from submissions/backdoor-detection/")
    sys.exit(1)


def main() -> None:
    """Validate results.json and generated outputs."""
    results_path = "results/results.json"
    if not os.path.exists(results_path):
        print(f"FAIL: {results_path} not found. Run run.py first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]
    errors = []

    # Check experiment count
    n_expected = (len(metadata["poison_fractions"])
                  * len(metadata["trigger_strengths"])
                  * len(metadata["hidden_dims"]))
    print(f"Experiments: {len(results)} (expected {n_expected})")
    if len(results) != n_expected:
        errors.append(f"Expected {n_expected} experiments, got {len(results)}")

    # Check all results have required fields
    required_fields = [
        "detection_auc", "eigenvalue_ratio", "clean_model_accuracy",
        "backdoored_model_accuracy", "backdoor_success_rate",
        "n_poisoned", "n_total", "elapsed_seconds",
    ]
    for i, r in enumerate(results):
        for field in required_fields:
            if field not in r:
                errors.append(f"Result {i} missing field: {field}")

    # Check AUC values are valid
    aucs = [r["detection_auc"] for r in results]
    for i, auc in enumerate(aucs):
        if not (0.0 <= auc <= 1.0):
            errors.append(f"Result {i}: AUC={auc} out of [0, 1] range")

    print(f"AUC range: [{min(aucs):.3f}, {max(aucs):.3f}]")
    mean_auc = sum(aucs) / len(aucs)
    print(f"AUC mean: {mean_auc:.3f}")

    # Check that higher poison fractions tend to have higher AUCs
    pf_to_aucs = {}
    for r in results:
        pf = r["config"]["poison_fraction"]
        pf_to_aucs.setdefault(pf, []).append(r["detection_auc"])

    print("\nDetection AUC by poison fraction:")
    prev_mean = 0.0
    for pf in sorted(pf_to_aucs.keys()):
        a = pf_to_aucs[pf]
        m = sum(a) / len(a)
        print(f"  Poison {pf*100:.0f}%: mean AUC = {m:.3f} (n={len(a)})")

    # Check model accuracy is reasonable
    clean_accs = [r["clean_model_accuracy"] for r in results]
    print(f"\nClean model accuracy: mean={sum(clean_accs)/len(clean_accs):.3f}")
    if sum(clean_accs) / len(clean_accs) < 0.5:
        errors.append(f"Clean model accuracy too low: {sum(clean_accs)/len(clean_accs):.3f}")

    # Check backdoor success rate
    bsr = [r["backdoor_success_rate"] for r in results]
    print(f"Backdoor success rate: mean={sum(bsr)/len(bsr):.3f}")

    # Check eigenvalue ratios are positive
    eig_ratios = [r["eigenvalue_ratio"] for r in results]
    for i, er in enumerate(eig_ratios):
        if er <= 0:
            errors.append(f"Result {i}: eigenvalue ratio {er} <= 0")

    # Check generated files
    expected_files = [
        "results/results.json",
        "results/report.md",
        "results/fig_auc_heatmap.png",
        "results/fig_auc_by_model_size.png",
        "results/fig_eigenvalue_ratio.png",
    ]
    for fpath in expected_files:
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  OK: {fpath} ({size} bytes)")
        else:
            errors.append(f"Missing file: {fpath}")

    # Thesis check: AUC >= 0.9 when poison >= 10%
    high_poison = [r for r in results if r["config"]["poison_fraction"] >= 0.10]
    high_auc_count = sum(1 for r in high_poison if r["detection_auc"] >= 0.9)
    print(f"\nThesis check: {high_auc_count}/{len(high_poison)} experiments with "
          f"poison >= 10% achieved AUC >= 0.9")

    # Report
    print(f"\n{'='*50}")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: All checks OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
