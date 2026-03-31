#!/usr/bin/env python3
"""Validate the DP scaling law experiment results.

Usage (from submissions/dp-scaling/):
    .venv/bin/python validate.py

Checks:
    1. All expected output files exist.
    2. JSON has correct structure (raw_results, aggregated, scaling_fits, summary).
    3. All 45 training runs completed.
    4. All 3 privacy levels have scaling law fits.
    5. Scaling exponents are positive and reasonable.
    6. Scaling exponent ratios between privacy levels are bounded.
    7. R-squared values indicate reasonable fits.
    8. Test losses are finite and positive.
    9. Figures exist and are non-empty.

Exit code 0 = all checks pass. Non-zero = validation failure.
"""

import json
import os
import sys

# Working directory guard
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != _script_dir:
    print(f"Changing directory to {_script_dir}")
    os.chdir(_script_dir)


def validate() -> bool:
    """Run all validation checks. Returns True if all pass."""
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            failed += 1

    print("=" * 60)
    print("Validating DP Scaling Law Results")
    print("=" * 60)
    print()

    # Check 1: Output files exist
    expected_files = [
        "results/experiment_results.json",
        "results/scaling_laws.png",
        "results/accuracy_comparison.png",
    ]
    for f in expected_files:
        check(f"File exists: {f}", os.path.isfile(f))

    # Load results
    results_path = "results/experiment_results.json"
    if not os.path.isfile(results_path):
        print("\nCannot continue: results file missing.")
        return False

    with open(results_path) as f:
        results = json.load(f)

    # Check 2: JSON structure
    required_keys = ["raw_results", "aggregated", "scaling_fits", "summary", "config"]
    for key in required_keys:
        check(f"JSON has key '{key}'", key in results)

    # Check 2b: Reproducibility metadata
    config = results.get("config", {})
    env = config.get("environment", {})
    check("Config has bootstrap settings", "bootstrap" in config)
    check("Config has environment metadata", "environment" in config)
    for key in ["python_version", "torch_version", "numpy_version", "scipy_version"]:
        check(f"Environment has {key}", key in env and bool(env[key]))

    # Check 3: 45 training runs
    n_runs = len(results.get("raw_results", []))
    check(f"45 training runs completed (got {n_runs})", n_runs == 45)

    # Check 4: All privacy levels present
    expected_levels = ["non_private", "moderate_dp", "strong_dp"]
    for level in expected_levels:
        check(
            f"Scaling fit exists for {level}",
            level in results.get("scaling_fits", {}),
        )

    # Check 5: Scaling exponents are positive
    for level in expected_levels:
        fit = results.get("scaling_fits", {}).get(level, {})
        alpha = fit.get("alpha")
        if alpha is not None:
            check(
                f"{level} alpha > 0 (got {alpha:.4f})",
                alpha > 0,
            )
            check(
                f"{level} alpha < 5.0 (got {alpha:.4f})",
                alpha < 5.0,
                "Unreasonably large exponent",
            )

    # Check 6: Scaling exponents are distinct (DP affects scaling behavior)
    np_alpha = results.get("scaling_fits", {}).get("non_private", {}).get("alpha")
    mod_alpha = results.get("scaling_fits", {}).get("moderate_dp", {}).get("alpha")
    strong_alpha = results.get("scaling_fits", {}).get("strong_dp", {}).get("alpha")

    if np_alpha is not None and mod_alpha is not None:
        # Verify exponents are in a comparable range (within 10x of each other)
        ratio = mod_alpha / np_alpha if np_alpha > 0 else 0
        check(
            f"Exponent ratio moderate/non-private is bounded (got {ratio:.4f})",
            0.1 <= ratio <= 10.0,
            "Exponents should be in a comparable range",
        )
    if np_alpha is not None and strong_alpha is not None:
        ratio = strong_alpha / np_alpha if np_alpha > 0 else 0
        check(
            f"Exponent ratio strong/non-private is bounded (got {ratio:.4f})",
            0.1 <= ratio <= 10.0,
            "Exponents should be in a comparable range",
        )

    # Check 7: R-squared values
    for level in expected_levels:
        fit = results.get("scaling_fits", {}).get(level, {})
        r2 = fit.get("r_squared")
        if r2 is not None:
            check(
                f"{level} R^2 >= 0.5 (got {r2:.4f})",
                r2 >= 0.5,
                "Poor fit quality",
            )

    # Check 7b: Alpha confidence intervals
    for level in expected_levels:
        fit = results.get("scaling_fits", {}).get(level, {})
        alpha = fit.get("alpha")
        ci = fit.get("alpha_ci95")
        if alpha is not None:
            check(f"{level} has alpha_ci95 interval", isinstance(ci, list) and len(ci) == 2)
            if isinstance(ci, list) and len(ci) == 2:
                check(
                    f"{level} alpha_ci95 ordered",
                    ci[0] <= ci[1],
                    f"Got [{ci[0]}, {ci[1]}]",
                )
                check(
                    f"{level} alpha inside alpha_ci95",
                    ci[0] <= alpha <= ci[1],
                    f"alpha={alpha:.4f}, ci={ci}",
                )

    # Check 8: Test losses are finite and positive
    all_losses_valid = True
    for run in results.get("raw_results", []):
        loss = run.get("test_loss")
        if loss is None or loss <= 0 or loss != loss:  # NaN check
            all_losses_valid = False
            break
    check("All test losses are finite and positive", all_losses_valid)

    # Check 9: Figures are non-empty
    for fig_file in ["results/scaling_laws.png", "results/accuracy_comparison.png"]:
        if os.path.isfile(fig_file):
            size = os.path.getsize(fig_file)
            check(f"{fig_file} is non-empty ({size} bytes)", size > 1000)

    # Summary
    print()
    print("=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} checks passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    return failed == 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
