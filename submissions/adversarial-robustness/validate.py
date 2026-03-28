#!/usr/bin/env python3
"""
Adversarial Robustness Scaling -- results validator.

Checks that run.py produced valid, complete results with expected properties.

Usage (from submissions/adversarial-robustness/):
    .venv/bin/python validate.py
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Working-directory guard
# ---------------------------------------------------------------------------
if not os.path.isfile(os.path.join(os.getcwd(), "SKILL.md")):
    print("ERROR: validate.py must be executed from the submission directory")
    print("       (the folder containing SKILL.md).")
    print(f"       Current directory: {os.getcwd()}")
    sys.exit(1)

RESULTS_DIR = os.path.join(os.getcwd(), "results")
EXPECTED_WIDTHS = [16, 32, 64, 128, 256, 512]
EXPECTED_EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.5]
EXPECTED_SEEDS = [42, 123, 7]
EXPECTED_DATASETS = ["circles", "moons"]
EXPECTED_N_RESULTS = (len(EXPECTED_WIDTHS) * len(EXPECTED_EPSILONS)
                      * len(EXPECTED_SEEDS) * len(EXPECTED_DATASETS))  # 180


def main() -> None:
    """Validate results with error accumulation."""
    errors: list[str] = []

    # 1. Check results directory exists
    if not os.path.isdir(RESULTS_DIR):
        print("FAIL: results/ directory does not exist. Run run.py first.")
        sys.exit(1)

    # 2. Check required files
    required_files = ["results.json", "clean_vs_robust.png",
                      "robustness_gap.png", "param_scaling.png"]
    for fname in required_files:
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.isfile(fpath):
            errors.append(f"Missing file: results/{fname}")

    # 3. Load and validate JSON
    json_path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.isfile(json_path):
        errors.append("Cannot load results.json -- file missing")
        _report(errors)
        return

    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"results.json is not valid JSON: {e}")
        _report(errors)
        return

    results = data.get("results", [])
    summary = data.get("summary", {})
    dataset_summaries = data.get("dataset_summaries", {})
    aggregated = data.get("aggregated", [])

    # 4. Check result count
    if len(results) != EXPECTED_N_RESULTS:
        errors.append(f"Expected {EXPECTED_N_RESULTS} results, got {len(results)}")

    # 5. Check all widths, epsilons, seeds, datasets present
    found_widths = sorted(set(r.get("hidden_width") for r in results))
    found_epsilons = sorted(set(r.get("epsilon") for r in results))
    found_seeds = sorted(set(r.get("seed") for r in results))
    found_datasets = sorted(set(r.get("dataset") for r in results))

    if found_widths != EXPECTED_WIDTHS:
        errors.append(f"Expected widths {EXPECTED_WIDTHS}, found {found_widths}")
    if found_epsilons != EXPECTED_EPSILONS:
        errors.append(f"Expected epsilons {EXPECTED_EPSILONS}, found {found_epsilons}")
    if found_seeds != sorted(EXPECTED_SEEDS):
        errors.append(f"Expected seeds {sorted(EXPECTED_SEEDS)}, found {found_seeds}")
    if found_datasets != sorted(EXPECTED_DATASETS):
        errors.append(f"Expected datasets {sorted(EXPECTED_DATASETS)}, found {found_datasets}")

    # 6. Validate each result entry
    required_keys = ["dataset", "seed", "hidden_width", "param_count",
                     "clean_acc", "epsilon", "fgsm_acc", "pgd_acc",
                     "fgsm_gap", "pgd_gap"]
    for i, r in enumerate(results):
        for key in required_keys:
            if key not in r:
                errors.append(f"Result {i}: missing key '{key}'")

        # Accuracy bounds
        for acc_key in ["clean_acc", "fgsm_acc", "pgd_acc"]:
            val = r.get(acc_key)
            if val is not None and not (0.0 <= val <= 1.0):
                errors.append(f"Result {i}: {acc_key}={val} out of [0, 1]")

        # Gap consistency
        fgsm_gap = r.get("fgsm_gap")
        expected_fgsm_gap = r.get("clean_acc", 0) - r.get("fgsm_acc", 0)
        if fgsm_gap is not None and abs(fgsm_gap - expected_fgsm_gap) > 1e-6:
            errors.append(f"Result {i}: fgsm_gap inconsistent "
                          f"({fgsm_gap} vs clean-fgsm={expected_fgsm_gap})")

        pgd_gap = r.get("pgd_gap")
        expected_pgd_gap = r.get("clean_acc", 0) - r.get("pgd_acc", 0)
        if pgd_gap is not None and abs(pgd_gap - expected_pgd_gap) > 1e-6:
            errors.append(f"Result {i}: pgd_gap inconsistent "
                          f"({pgd_gap} vs clean-pgd={expected_pgd_gap})")

    # 7. Scientific validity checks
    # All models should achieve reasonable clean accuracy (>= 80%)
    for ds in EXPECTED_DATASETS:
        for w in EXPECTED_WIDTHS:
            w_results = [r for r in results
                         if r.get("hidden_width") == w and r.get("dataset") == ds]
            if w_results:
                clean_accs = [r["clean_acc"] for r in w_results]
                mean_clean = sum(clean_accs) / len(clean_accs)
                if mean_clean < 0.80:
                    errors.append(f"{ds}/width {w}: mean clean_acc={mean_clean:.4f} < 0.80")

    # PGD should be at least as strong as FGSM (robust_acc_pgd <= robust_acc_fgsm)
    violations = 0
    for r in results:
        fgsm_acc = r.get("fgsm_acc", 1.0)
        pgd_acc = r.get("pgd_acc", 1.0)
        if pgd_acc > fgsm_acc + 0.05:  # 5% tolerance for stochastic variation
            violations += 1
    if violations > len(results) * 0.1:  # Allow up to 10% violations
        errors.append(f"PGD weaker than FGSM in {violations}/{len(results)} cases "
                      f"(expected PGD to be at least as strong)")

    # Robust accuracy should generally decrease with epsilon
    for ds in EXPECTED_DATASETS:
        for w in EXPECTED_WIDTHS:
            for seed in EXPECTED_SEEDS:
                w_results = sorted(
                    [r for r in results
                     if r.get("hidden_width") == w
                     and r.get("dataset") == ds
                     and r.get("seed") == seed],
                    key=lambda r: r.get("epsilon", 0))
                if len(w_results) >= 2:
                    for j in range(len(w_results) - 1):
                        if (w_results[j + 1].get("fgsm_acc", 0) >
                                w_results[j].get("fgsm_acc", 1) + 0.10):
                            errors.append(
                                f"{ds}/width {w}/seed {seed}: FGSM acc increased from "
                                f"eps={w_results[j]['epsilon']} to "
                                f"eps={w_results[j+1]['epsilon']} (non-monotonic)")

    # 8. Check aggregated results
    if not aggregated:
        errors.append("Missing aggregated results (cross-seed averages)")
    else:
        expected_agg = len(EXPECTED_WIDTHS) * len(EXPECTED_EPSILONS) * len(EXPECTED_DATASETS)
        if len(aggregated) != expected_agg:
            errors.append(f"Expected {expected_agg} aggregated entries, got {len(aggregated)}")

    # 9. Check summary statistics
    if not dataset_summaries:
        errors.append("Missing dataset_summaries in results.json")
    else:
        found_summary_datasets = sorted(dataset_summaries.keys())
        if found_summary_datasets != sorted(EXPECTED_DATASETS):
            errors.append("dataset_summaries missing expected datasets "
                          f"(found {found_summary_datasets})")
        for ds_name in EXPECTED_DATASETS:
            ds_summary = dataset_summaries.get(ds_name, {})
            if "per_width" not in ds_summary:
                errors.append(f"{ds_name} summary missing 'per_width' key")
            if "corr_logparams_fgsm_gap" not in ds_summary:
                errors.append(f"{ds_name} summary missing correlation statistics")
    if summary and "per_width" not in summary:
        errors.append("Legacy summary missing 'per_width' key")

    # 10. Check plot files are non-empty
    for fname in ["clean_vs_robust.png", "robustness_gap.png", "param_scaling.png"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath) and os.path.getsize(fpath) < 1000:
            errors.append(f"Plot {fname} appears too small ({os.path.getsize(fpath)} bytes)")

    _report(errors)


def _report(errors: list[str]) -> None:
    """Print validation report and exit with appropriate code."""
    print("=" * 60)
    print("Adversarial Robustness Scaling -- Validation Report")
    print("=" * 60)

    if errors:
        print(f"\nFAILED -- {len(errors)} error(s):\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print()
        sys.exit(1)
    else:
        print("\nPASSED -- all checks passed.")
        print()
        # Print key results summary
        json_path = os.path.join(RESULTS_DIR, "results.json")
        try:
            with open(json_path) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            dataset_summaries = data.get("dataset_summaries", {})
            config = data.get("config", {})
            results = data.get("results", [])
            if dataset_summaries:
                n_datasets = len(config.get("datasets", []))
                n_seeds = len(config.get("seeds", []))
                print(f"Configuration: {n_datasets} datasets, {n_seeds} seeds, "
                      f"{len(results)} total experiments")
                if summary.get("per_width"):
                    print(f"  - Legacy summary preserved for "
                          f"{len(summary['widths'])} model sizes")
                for ds_name in EXPECTED_DATASETS:
                    ds_summary = dataset_summaries.get(ds_name)
                    if ds_summary and ds_summary.get("corr_logparams_fgsm_gap") is not None:
                        print(f"  - {ds_name}: {ds_summary['n_experiments']} dataset results, "
                              f"Corr(log params, FGSM gap) = "
                              f"{ds_summary['corr_logparams_fgsm_gap']:.4f}, "
                              f"Corr(log params, PGD gap) = "
                              f"{ds_summary['corr_logparams_pgd_gap']:.4f}")
                print()
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass  # Summary printing is best-effort
        sys.exit(0)


if __name__ == "__main__":
    main()
