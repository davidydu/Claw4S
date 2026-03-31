"""Validate analysis results for completeness and correctness."""

import json
import os
import sys

from src.config import MSI_ARTIFACT_THRESHOLD, NONLINEARITY_BOOTSTRAP_SAMPLES

if not os.path.exists("src/data.py"):
    print("ERROR: Must run from submissions/emergent-abilities/ directory")
    raise SystemExit(1)

errors = []

# ── Check results directory exists ───────────────────────────────────────────

if not os.path.isdir("results"):
    print("ERROR: results/ directory not found. Run run.py first.")
    sys.exit(1)

# ── Check results.json ───────────────────────────────────────────────────────

results_path = "results/results.json"
if not os.path.isfile(results_path):
    errors.append("results/results.json not found")
else:
    with open(results_path) as f:
        data = json.load(f)
    config = data.get("analysis_config", {})
    analysis_threshold = config.get("msi_artifact_threshold", MSI_ARTIFACT_THRESHOLD)
    analysis_bootstrap = config.get("n_bootstrap", NONLINEARITY_BOOTSTRAP_SAMPLES)

    # Check top-level keys
    required_keys = [
        "metric_comparisons",
        "nonlinearity_scores",
        "synthetic_demo",
        "mmlu_analysis",
        "analysis_config",
    ]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing key in results.json: {key}")

    # Check metric comparisons
    if "metric_comparisons" in data:
        mc = data["metric_comparisons"]
        n_tasks = len(mc)
        print(f"BIG-Bench tasks analyzed: {n_tasks}")
        if n_tasks < 4:
            errors.append(f"Expected >= 4 tasks, got {n_tasks}")

        for task_name, comparison in mc.items():
            entries = comparison.get("entries", [])
            if len(entries) < 3:
                errors.append(f"Task {task_name}: expected >= 3 entries, got {len(entries)}")
            for entry in entries:
                em = entry.get("exact_match", -1)
                pc = entry.get("partial_credit", -1)
                if not (0.0 <= em <= 1.0):
                    errors.append(f"Task {task_name}, {entry.get('model')}: exact_match {em} out of range")
                if not (0.0 <= pc <= 1.0):
                    errors.append(f"Task {task_name}, {entry.get('model')}: partial_credit {pc} out of range")
                # Partial credit should be >= exact match (continuous is higher)
                if pc < em - 0.01:
                    errors.append(
                        f"Task {task_name}, {entry.get('model')}: partial_credit {pc:.4f} < exact_match {em:.4f}"
                    )

    # Check nonlinearity scores
    if "nonlinearity_scores" in data:
        ns = data["nonlinearity_scores"]
        n_scored = len(ns)
        print(f"Tasks with nonlinearity scores: {n_scored}")

        artifact_count = sum(
            1
            for s in ns.values()
            if s.get("verdict") in ("likely_artifact", "likely_artifact_sparse")
        )
        definitional_count = sum(
            1 for s in ns.values() if s.get("verdict") == "definitional_single_token"
        )
        genuine_count = sum(1 for s in ns.values() if s.get("verdict") == "possibly_genuine")
        uncertain_count = sum(
            1 for s in ns.values() if s.get("verdict") in ("uncertain", "inconclusive_sparse")
        )
        print(f"  Likely artifacts (MSI > {analysis_threshold:.1f}): {artifact_count}")
        print(f"  Definitional (n_tokens = 1): {definitional_count}")
        print(
            f"  Possibly genuine (MSI <= {analysis_threshold:.1f}, excluding n_tokens = 1): "
            f"{genuine_count}"
        )
        print(f"  Uncertain / sparse-evidence: {uncertain_count}")

        for task_name, score in ns.items():
            if "n_tokens" not in score:
                errors.append(f"Nonlinearity score missing n_tokens: {task_name}")
            if "metric_type" not in score:
                errors.append(f"Nonlinearity score missing metric_type: {task_name}")
            for key in (
                "msi_ci_lower",
                "msi_ci_upper",
                "artifact_probability",
                "artifact_threshold",
                "n_bootstrap",
                "verdict",
            ):
                if key not in score:
                    errors.append(f"Nonlinearity score missing {key}: {task_name}")
            if not (0.0 <= score.get("artifact_probability", -1) <= 1.0):
                errors.append(f"Invalid artifact_probability for {task_name}")
            ci_low = score.get("msi_ci_lower", float("inf"))
            ci_high = score.get("msi_ci_upper", float("-inf"))
            if ci_low > ci_high:
                errors.append(f"Invalid MSI CI bounds for {task_name}: [{ci_low}, {ci_high}]")
            if score.get("artifact_threshold") != analysis_threshold:
                errors.append(
                    f"Unexpected artifact threshold for {task_name}: "
                    f"{score.get('artifact_threshold')}"
                )
            if score.get("n_bootstrap") != analysis_bootstrap:
                errors.append(
                    f"Unexpected bootstrap count for {task_name}: {score.get('n_bootstrap')}"
                )

    # Check synthetic demo
    if "synthetic_demo" in data:
        demo = data["synthetic_demo"]
        if "exact_match" not in demo or "partial_credit" not in demo:
            errors.append("Synthetic demo missing exact_match or partial_credit")
        elif len(demo["exact_match"]) < 10:
            errors.append(f"Synthetic demo too few points: {len(demo['exact_match'])}")
        else:
            print(f"Synthetic demo points: {len(demo['exact_match'])}")

    # Check MMLU analysis
    if "mmlu_analysis" in data:
        mmlu = data["mmlu_analysis"]
        n_models = mmlu.get("n_models", 0)
        print(f"MMLU models analyzed: {n_models}")
        if n_models < 5:
            errors.append(f"Expected >= 5 MMLU models, got {n_models}")

    # Check analysis config
    if data.get("seed") != config.get("seed", data.get("seed")):
        errors.append("Seed mismatch between top-level field and analysis_config.seed")
    if not isinstance(analysis_threshold, (int, float)):
        errors.append(f"Invalid msi_artifact_threshold: {analysis_threshold}")
    if not isinstance(analysis_bootstrap, int) or analysis_bootstrap <= 0:
        errors.append(f"Invalid n_bootstrap: {analysis_bootstrap}")

# ── Check report ─────────────────────────────────────────────────────────────

report_path = "results/report.md"
if not os.path.isfile(report_path):
    errors.append("results/report.md not found")
else:
    with open(report_path) as f:
        report = f.read()
    report_len = len(report)
    print(f"Report length: {report_len} characters")
    if report_len < 500:
        errors.append(f"Report too short: {report_len} characters")
    if "Schaeffer" not in report:
        errors.append("Report does not reference Schaeffer et al.")
    if "95% CI" not in report:
        errors.append("Report missing bootstrap uncertainty annotation (95% CI)")

# ── Check figures ────────────────────────────────────────────────────────────

expected_figures = [
    "results/figures/synthetic_demo.png",
    "results/figures/nonlinearity_heatmap.png",
    "results/figures/mmlu_scaling.png",
]
# Also check for at least one metric comparison figure
metric_comparison_found = False
for fname in os.listdir("results/figures") if os.path.isdir("results/figures") else []:
    if fname.startswith("metric_comparison_") and fname.endswith(".png"):
        metric_comparison_found = True
        break

if not metric_comparison_found:
    errors.append("No metric_comparison_*.png figures found")

for fig_path in expected_figures:
    if not os.path.isfile(fig_path):
        errors.append(f"Missing figure: {fig_path}")
    else:
        size = os.path.getsize(fig_path)
        if size < 1000:
            errors.append(f"Figure too small ({size} bytes): {fig_path}")

n_figures = len(os.listdir("results/figures")) if os.path.isdir("results/figures") else 0
print(f"Figures generated: {n_figures}")

# ── Summary ──────────────────────────────────────────────────────────────────

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
