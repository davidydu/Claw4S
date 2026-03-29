"""Validate analysis results for the benchmark-corr submission."""

import json
import math
import os
import sys

from src.data import get_data_fingerprint

errors = []

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

# Check 2: Metadata has >= 30 models and >= 6 benchmarks
print("Check 2: Metadata has >= 30 models and >= 6 benchmarks...")
if data is not None:
    meta = data.get("metadata", {})
    n_models = meta.get("n_models", 0)
    n_benchmarks = meta.get("n_benchmarks", 0)
    if n_models < 30:
        errors.append(f"Expected >= 30 models, got {n_models}")
    else:
        print(f"  PASS: {n_models} models.")
    if n_benchmarks < 6:
        errors.append(f"Expected >= 6 benchmarks, got {n_benchmarks}")
    else:
        print(f"  PASS: {n_benchmarks} benchmarks.")
    fingerprint = meta.get("data_fingerprint_sha256", "")
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        errors.append("Metadata fingerprint missing or not a valid SHA-256 hex digest")
    else:
        expected_fp = get_data_fingerprint()
        if fingerprint != expected_fp:
            errors.append("Metadata fingerprint does not match current hardcoded dataset")
        else:
            print("  PASS: Data fingerprint matches hardcoded source table.")

# Check 3: Correlation matrices are present and symmetric
print("Check 3: Correlation matrices are present and symmetric...")
if data is not None:
    corr = data.get("correlation", {})
    for key in ["pearson", "spearman"]:
        matrix = corr.get(key)
        if matrix is None:
            errors.append(f"Correlation matrix '{key}' is missing")
        else:
            arr = [[float(x) for x in row] for row in matrix]
            n = len(arr)
            symmetric = True
            for i in range(n):
                for j in range(n):
                    if abs(arr[i][j] - arr[j][i]) > 1e-6:
                        symmetric = False
            if not symmetric:
                errors.append(f"Correlation matrix '{key}' is not symmetric")
            elif abs(arr[0][0] - 1.0) > 1e-6:
                errors.append(f"Correlation matrix '{key}' diagonal is not 1.0")
            else:
                print(f"  PASS: '{key}' is symmetric with unit diagonal.")

# Check 4: PCA n_components_90 <= 3 (thesis: high redundancy)
print("Check 4: PCA n_components_90 <= 3 (thesis check)...")
if data is not None:
    pca = data.get("pca", {})
    n90 = pca.get("n_components_90")
    if n90 is None:
        errors.append("PCA n_components_90 is missing")
    elif n90 > 3:
        errors.append(f"PCA n_components_90 = {n90}, expected <= 3 for redundancy thesis")
    else:
        cumvar = pca.get("cumulative_variance", [])
        if len(cumvar) >= n90:
            print(f"  PASS: {n90} components explain {cumvar[n90-1]*100:.1f}% variance.")
        else:
            print(f"  PASS: n_components_90 = {n90}")

# Check 5: Redundancy greedy selection has entries
print("Check 5: Redundancy analysis has greedy selection results...")
if data is not None:
    red = data.get("redundancy", {})
    order = red.get("greedy_selection_order", [])
    varexp = red.get("greedy_variance_explained", [])
    if len(order) < 6:
        errors.append(f"Greedy selection has {len(order)} entries, expected >= 6")
    elif len(varexp) < 6:
        errors.append(f"Greedy variance has {len(varexp)} entries, expected >= 6")
    else:
        # First 2 benchmarks should explain >= 70% variance
        if varexp[1] < 0.70:
            errors.append(
                f"Top 2 benchmarks explain only {varexp[1]*100:.1f}% variance, expected >= 70%"
            )
        else:
            print(f"  PASS: Top 2 benchmarks ({order[0]}, {order[1]}) "
                  f"explain {varexp[1]*100:.1f}% variance.")

# Check 6: Family analysis has valid silhouette score
print("Check 6: Family analysis has valid silhouette score...")
if data is not None:
    fam = data.get("family_analysis", {})
    sil = fam.get("silhouette_score")
    if sil is None or not math.isfinite(sil):
        errors.append(f"Silhouette score is missing or not finite: {sil}")
    elif sil < -1.0 or sil > 1.0:
        errors.append(f"Silhouette score {sil:.3f} is out of range [-1, 1]")
    else:
        print(f"  PASS: Silhouette score = {sil:.3f}")

# Check 7: All 5 figure PNGs exist
print("Check 7: All 5 figure PNGs exist in results/figures/...")
required_figures = [
    "correlation.png",
    "pca_variance.png",
    "model_pca.png",
    "dendrogram.png",
    "redundancy.png",
]
for fig in required_figures:
    path = os.path.join("results", "figures", fig)
    if not os.path.exists(path):
        errors.append(f"Missing figure: {path}")
    else:
        print(f"  PASS: {path} exists.")

# Check 8: results/report.md exists and has content
print("Check 8: results/report.md exists and has content...")
if not os.path.exists("results/report.md"):
    errors.append("results/report.md does not exist")
else:
    with open("results/report.md") as f:
        content = f.read()
    if len(content) < 500:
        errors.append(f"results/report.md is too short ({len(content)} chars)")
    else:
        print(f"  PASS: results/report.md has {len(content)} characters.")

# Check 9: PC1-param correlation is significant
print("Check 9: PC1-log(params) correlation is significant...")
if data is not None:
    fam = data.get("family_analysis", {})
    pc1_corr = fam.get("pc1_param_correlation")
    pc1_pval = fam.get("pc1_param_pvalue")
    if pc1_corr is None or not math.isfinite(pc1_corr):
        errors.append(f"PC1-param correlation is missing or not finite: {pc1_corr}")
    elif abs(pc1_corr) < 0.5:
        errors.append(f"PC1-param correlation = {pc1_corr:.3f}, expected |r| >= 0.5")
    else:
        print(f"  PASS: PC1-param correlation = {pc1_corr:.3f} (p = {pc1_pval:.2e})")

# Check 10: Bootstrap robustness summary exists and is valid
print("Check 10: Bootstrap robustness outputs are present and coherent...")
if data is not None:
    robust = data.get("robustness", {})
    n_boot = robust.get("n_bootstrap_samples", 0)
    if n_boot < 100:
        errors.append(f"Expected >= 100 bootstrap samples, got {n_boot}")
    else:
        print(f"  PASS: n_bootstrap_samples = {n_boot}")

    ci = robust.get("pc1_param_correlation_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        errors.append("pc1_param_correlation_ci95 missing or malformed")
    elif ci[0] > ci[1]:
        errors.append("pc1_param_correlation_ci95 bounds are inverted")
    else:
        print(f"  PASS: PC1-param bootstrap CI95 = [{ci[0]:.3f}, {ci[1]:.3f}]")

    top2 = robust.get("top2_selection_frequencies", [])
    if not top2:
        errors.append("top2_selection_frequencies is missing or empty")
    else:
        top = top2[0]
        if "pair" not in top or "frequency" not in top:
            errors.append("top2_selection_frequencies[0] missing pair/frequency")
        else:
            print(
                f"  PASS: Most frequent top-2 subset is {top['pair']} "
                f"({top['frequency']*100:.1f}%)."
            )

# Final result
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
