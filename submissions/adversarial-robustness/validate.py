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


def infer_expected_config(data: dict) -> dict:
    """Infer expected validation dimensions from results config if present."""
    config = data.get("config", {})

    widths = config.get("hidden_widths", EXPECTED_WIDTHS)
    epsilons = config.get("epsilons", EXPECTED_EPSILONS)
    seeds = config.get("seeds", EXPECTED_SEEDS)
    raw_datasets = config.get("datasets", EXPECTED_DATASETS)

    datasets: list[str] = []
    for ds in raw_datasets:
        if isinstance(ds, dict):
            name = ds.get("name")
        else:
            name = ds
        if isinstance(name, str) and name not in datasets:
            datasets.append(name)
    if not datasets:
        datasets = list(EXPECTED_DATASETS)

    widths = sorted(set(int(w) for w in widths))
    epsilons = sorted(set(float(eps) for eps in epsilons))
    seeds = sorted(set(int(seed) for seed in seeds))

    return {
        "widths": widths,
        "epsilons": epsilons,
        "seeds": seeds,
        "datasets": datasets,
        "n_results": len(widths) * len(epsilons) * len(seeds) * len(datasets),
        "n_aggregated": len(widths) * len(epsilons) * len(datasets),
    }


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
    expected = infer_expected_config(data)

    # 4. Check result count
    if len(results) != expected["n_results"]:
        errors.append(f"Expected {expected['n_results']} results, got {len(results)}")

    # 5. Check all widths, epsilons, seeds, datasets present
    found_widths = sorted(set(r.get("hidden_width") for r in results))
    found_epsilons = sorted(set(r.get("epsilon") for r in results))
    found_seeds = sorted(set(r.get("seed") for r in results))
    found_datasets = sorted(set(r.get("dataset") for r in results))

    if found_widths != expected["widths"]:
        errors.append(f"Expected widths {expected['widths']}, found {found_widths}")
    if found_epsilons != expected["epsilons"]:
        errors.append(f"Expected epsilons {expected['epsilons']}, found {found_epsilons}")
    if found_seeds != expected["seeds"]:
        errors.append(f"Expected seeds {expected['seeds']}, found {found_seeds}")
    if found_datasets != sorted(expected["datasets"]):
        errors.append(f"Expected datasets {sorted(expected['datasets'])}, found {found_datasets}")

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
    for ds in expected["datasets"]:
        for w in expected["widths"]:
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
    for ds in expected["datasets"]:
        for w in expected["widths"]:
            for seed in expected["seeds"]:
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
        if len(aggregated) != expected["n_aggregated"]:
            errors.append(
                f"Expected {expected['n_aggregated']} aggregated entries, got {len(aggregated)}"
            )

    # 9. Check summary statistics
    if not dataset_summaries:
        errors.append("Missing dataset_summaries in results.json")
    else:
        found_summary_datasets = sorted(dataset_summaries.keys())
        if found_summary_datasets != sorted(expected["datasets"]):
            errors.append("dataset_summaries missing expected datasets "
                          f"(found {found_summary_datasets})")
        for ds_name in expected["datasets"]:
            ds_summary = dataset_summaries.get(ds_name, {})
            if "per_width" not in ds_summary:
                errors.append(f"{ds_name} summary missing 'per_width' key")
            if "corr_logparams_fgsm_gap" not in ds_summary:
                errors.append(f"{ds_name} summary missing correlation statistics")
            _validate_trend_stats(ds_name, ds_summary, errors)
    if summary and "per_width" not in summary:
        errors.append("Legacy summary missing 'per_width' key")

    # 10. Check reproducibility metadata
    environment = data.get("environment", {})
    if not isinstance(environment, dict) or not environment:
        errors.append("Missing environment metadata in results.json")
    else:
        for key in ("python", "torch", "numpy", "scipy", "platform"):
            if key not in environment:
                errors.append(f"environment metadata missing '{key}' field")

    # 11. Check plot files are non-empty
    for fname in ["clean_vs_robust.png", "robustness_gap.png", "param_scaling.png"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath) and os.path.getsize(fpath) < 1000:
            errors.append(f"Plot {fname} appears too small ({os.path.getsize(fpath)} bytes)")

    _report(errors, expected_datasets=expected["datasets"])


def _validate_trend_stats(ds_name: str, ds_summary: dict, errors: list[str]) -> None:
    """Validate trend statistics metadata in each dataset summary."""
    required = {
        "trend_fgsm_gap": "fgsm",
        "trend_pgd_gap": "pgd",
    }
    for key, label in required.items():
        trend = ds_summary.get(key)
        if not isinstance(trend, dict):
            errors.append(f"{ds_name} summary missing '{key}' trend metadata")
            continue

        ci = trend.get("pearson_r_ci95")
        if not (isinstance(ci, list) and len(ci) == 2):
            errors.append(f"{ds_name} {label} trend has malformed pearson_r_ci95")

        for p_name in ("pearson_p_value", "spearman_p_value"):
            p_val = trend.get(p_name)
            if p_val is not None and not (0.0 <= p_val <= 1.0):
                errors.append(
                    f"{ds_name} {label} trend {p_name}={p_val} outside [0, 1]"
                )


def _report(errors: list[str], expected_datasets: list[str] | None = None) -> None:
    """Print validation report and exit with appropriate code."""
    if expected_datasets is None:
        expected_datasets = list(EXPECTED_DATASETS)

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
                for ds_name in expected_datasets:
                    ds_summary = dataset_summaries.get(ds_name)
                    if ds_summary and ds_summary.get("corr_logparams_fgsm_gap") is not None:
                        fgsm_trend = ds_summary.get("trend_fgsm_gap", {})
                        pgd_trend = ds_summary.get("trend_pgd_gap", {})
                        print(f"  - {ds_name}: {ds_summary['n_experiments']} dataset results, "
                              f"Corr(log params, FGSM gap) = "
                              f"{ds_summary['corr_logparams_fgsm_gap']:.4f}, "
                              f"Corr(log params, PGD gap) = "
                              f"{ds_summary['corr_logparams_pgd_gap']:.4f}")
                        if fgsm_trend.get("pearson_p_value") is not None:
                            print(f"      trend p-values: FGSM={fgsm_trend['pearson_p_value']:.4f}, "
                                  f"PGD={pgd_trend['pearson_p_value']:.4f}")
                print()
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass  # Summary printing is best-effort
        sys.exit(0)


if __name__ == "__main__":
    main()
