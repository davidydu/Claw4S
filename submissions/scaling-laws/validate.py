"""Validate analysis results for the scaling-laws submission."""

import json
import math
import os
import sys

errors = []

# Check 0: Scan all phase results for error keys
print("Check 0: Scanning for phase errors...")
# (deferred until data is loaded — see below after Check 1)

# Check 1: results/results.json exists and is valid JSON
print("Check 1: results/results.json exists and is valid JSON...")
if not os.path.exists("results/results.json"):
    errors.append("results/results.json does not exist")
    data = None
else:
    try:
        with open("results/results.json") as f:
            data = json.load(f)
        print("  PASS: results/results.json found and parsed.")
    except json.JSONDecodeError as e:
        errors.append(f"results/results.json is not valid JSON: {e}")
        data = None

# Check 0 (continued): Scan all phase results for error keys
if data is not None:
    phase_keys = ["loss_scaling", "task_scaling", "cross_metric", "extrapolation", "cross_family"]
    found_errors = False
    for phase in phase_keys:
        phase_data = data.get(phase, {})
        if isinstance(phase_data, dict) and "error" in phase_data:
            errors.append(f"Phase '{phase}' has error: {phase_data['error']}")
            found_errors = True
    if not found_errors:
        print("  PASS: No phase errors found.")

# Check 2: All 3 loss formulations present with finite params
print("Check 2: All 3 loss formulations present with finite params...")
if data is not None:
    loss_scaling = data.get("loss_scaling", {})
    required_formulations = ["kaplan", "chinchilla", "corrected"]
    for form in required_formulations:
        if form not in loss_scaling:
            errors.append(f"Loss formulation '{form}' missing from results")
        else:
            params = loss_scaling[form].get("params", {})
            if not params:
                errors.append(f"Loss formulation '{form}' has no params")
            else:
                bad_params = [
                    k for k, v in params.items()
                    if not isinstance(v, (int, float)) or not math.isfinite(v)
                ]
                if bad_params:
                    errors.append(
                        f"Loss formulation '{form}' has non-finite params: {bad_params}"
                    )
                else:
                    print(f"  PASS: '{form}' has finite params: {list(params.keys())}")

# Check 3: Loss scaling best adj-R² > 0.90
print("Check 3: Loss scaling best adj-R² > 0.90...")
if data is not None:
    loss_scaling = data.get("loss_scaling", {})
    best_r2 = max(
        (
            v.get("adj_r_squared", float("nan"))
            for v in loss_scaling.values()
            if isinstance(v, dict)
        ),
        default=float("nan"),
    )
    if math.isnan(best_r2):
        errors.append("No valid adj-R² found in loss_scaling")
    elif best_r2 <= 0.90:
        errors.append(f"Best loss scaling adj-R² = {best_r2:.4f}, expected > 0.90")
    else:
        print(f"  PASS: Best loss scaling adj-R² = {best_r2:.4f}")

# Check 3b: Bootstrap convergence rate >= 0.80 for each loss formulation
print("Check 3b: Bootstrap convergence rate >= 0.80 for each loss formulation...")
if data is not None:
    loss_scaling = data.get("loss_scaling", {})
    required_formulations = ["kaplan", "chinchilla", "corrected"]
    for form in required_formulations:
        form_data = loss_scaling.get(form, {})
        ci = form_data.get("ci", {})
        conv_rate = ci.get("convergence_rate", None)
        if conv_rate is None:
            errors.append(f"Loss formulation '{form}' has no convergence_rate in CI data")
        elif conv_rate < 0.80:
            errors.append(
                f"Loss formulation '{form}' bootstrap convergence rate = {conv_rate:.3f}, "
                f"expected >= 0.80"
            )
        else:
            print(f"  PASS: '{form}' convergence rate = {conv_rate:.3f}")

# Check 4: At least 2 tasks have adj-R² < 0.85 for bounded power-law
print("Check 4: At least 2 tasks have bounded power-law adj-R² < 0.85...")
if data is not None:
    task_scaling = data.get("task_scaling", {})
    low_r2_tasks = [
        task
        for task, task_result in task_scaling.items()
        if isinstance(task_result, dict)
        and task_result.get("bounded_power_law", {}).get("adj_r_squared", float("nan")) < 0.85
    ]
    if len(low_r2_tasks) < 2:
        errors.append(
            f"Expected at least 2 tasks with bounded power-law adj-R² < 0.85, "
            f"got {len(low_r2_tasks)}: {low_r2_tasks}"
        )
    else:
        print(
            f"  PASS: {len(low_r2_tasks)} tasks with adj-R² < 0.85 "
            f"(confirms task scaling unreliability): {low_r2_tasks}"
        )

# Check 5: All 5 figure PNGs exist in results/figures/
print("Check 5: All 5 figure PNGs exist in results/figures/...")
required_figures = [
    "loss_scaling.png",
    "task_scaling.png",
    "residuals.png",
    "model_selection.png",
    "extrapolation.png",
]
for fig in required_figures:
    path = os.path.join("results", "figures", fig)
    if not os.path.exists(path):
        errors.append(f"Missing figure: {path}")
    else:
        print(f"  PASS: {path} exists.")

# Check 5b: Extrapolation MAPE values are finite
print("Check 5b: Extrapolation MAPE values are finite...")
if data is not None:
    extrap = data.get("extrapolation", {})
    loss_mape = extrap.get("loss_mape", float("nan"))
    task_mape = extrap.get("task_mape_avg", float("nan"))
    if not math.isfinite(loss_mape):
        errors.append(f"Extrapolation loss_mape is not finite: {loss_mape}")
    elif not math.isfinite(task_mape):
        errors.append(f"Extrapolation task_mape_avg is not finite: {task_mape}")
    else:
        print(f"  PASS: loss_mape={loss_mape:.2f}%, task_mape_avg={task_mape:.2f}%")

# Check 5c: Cross-family transfer results exist
print("Check 5c: Cross-family transfer results exist...")
if data is not None:
    cf = data.get("cross_family", {})
    overlapping = cf.get("overlapping_tasks", [])
    avg_err = cf.get("avg_transfer_error", float("nan"))
    if not overlapping:
        errors.append("Cross-family transfer has no overlapping tasks")
    elif not math.isfinite(avg_err):
        errors.append(f"Cross-family avg_transfer_error is not finite: {avg_err}")
    else:
        print(f"  PASS: {len(overlapping)} overlapping tasks, avg_transfer_error={avg_err:.2f}%")

# Check 6: results/report.md exists
print("Check 6: results/report.md exists...")
if not os.path.exists("results/report.md"):
    errors.append("results/report.md does not exist")
else:
    print("  PASS: results/report.md exists.")

# Check 7: Fitted Kaplan alpha in range [0.02, 0.20]
print("Check 7: Fitted Kaplan alpha in range [0.02, 0.20]...")
if data is not None:
    kaplan_params = data.get("loss_scaling", {}).get("kaplan", {}).get("params", {})
    kaplan_alpha = kaplan_params.get("alpha", float("nan"))
    if math.isnan(kaplan_alpha):
        errors.append("Kaplan alpha is NaN — fit may have failed")
    elif not (0.02 <= kaplan_alpha <= 0.20):
        errors.append(
            f"Kaplan alpha = {kaplan_alpha:.4f} outside expected range [0.02, 0.20]"
        )
    else:
        print(f"  PASS: Kaplan alpha = {kaplan_alpha:.4f}")

# Final result
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
