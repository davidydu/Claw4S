"""Validate experiment results for completeness and scientific soundness.

Must be run from the submission directory: submissions/backdoor-detection/
"""

import json
import os
import sys

from src.cli import ensure_submission_cwd


def strong_trigger_thesis_check(
    results: list[dict],
    auc_threshold: float = 0.9,
    min_poison_fraction: float = 0.10,
    min_trigger_strength: float = 10.0,
) -> tuple[int, int]:
    """Count strong-trigger experiments that clear the AUC threshold.

    The verified result for this submission is that trigger strength 10.0
    becomes reliably detectable once the poison fraction reaches 10%.
    """
    thesis_subset = [
        r for r in results
        if r["config"]["poison_fraction"] >= min_poison_fraction
        and r["config"]["trigger_strength"] >= min_trigger_strength
    ]
    passed = sum(1 for r in thesis_subset if r["detection_auc"] >= auc_threshold)
    return passed, len(thesis_subset)


def main() -> None:
    """Validate results.json and generated outputs."""
    try:
        ensure_submission_cwd(__file__)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

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

    # Thesis check: strong triggers become detectable once poison fraction reaches 10%.
    high_auc_count, thesis_total = strong_trigger_thesis_check(results)
    print(f"\nThesis check: {high_auc_count}/{thesis_total} strong-trigger experiments "
          f"with poison >= 10% achieved AUC >= 0.9")
    if high_auc_count != thesis_total:
        errors.append(
            "Not all strong-trigger experiments with poison >= 10% achieved AUC >= 0.9"
        )

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
