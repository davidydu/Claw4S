"""Validate Zipf analysis results for completeness and correctness."""

import json
import os
import sys

errors = []

# --- Check results.json exists ---
if not os.path.exists("results/results.json"):
    print("FATAL: results/results.json not found. Run run.py first.")
    sys.exit(1)

with open("results/results.json") as f:
    data = json.load(f)

meta = data["metadata"]
analyses = data["analyses"]
correlation = data.get("correlation", {})

print(f"Tokenizers: {meta['num_tokenizers']}")
print(f"Corpora: {meta['num_corpora']}")
print(f"Analyses: {len(analyses)}")

# --- Check tokenizer count ---
if meta["num_tokenizers"] < 2:
    errors.append(f"Expected >= 2 tokenizers, got {meta['num_tokenizers']}")

# --- Check corpus count ---
if meta["num_corpora"] < 3:
    errors.append(f"Expected >= 3 corpora, got {meta['num_corpora']}")

# --- Check analyses count ---
if len(analyses) < 6:
    errors.append(f"Expected >= 6 analyses, got {len(analyses)}")

# --- Check each analysis ---
corpus_types_seen = set()
for a in analyses:
    label = a.get("label", "unknown")
    corpus_types_seen.add(a.get("corpus_type", "unknown"))

    # Check global fit
    gf = a.get("global_fit", {})
    alpha = gf.get("alpha", 0)
    r_squared = gf.get("r_squared", 0)

    print(f"  {label}: alpha={alpha:.3f}, R^2={r_squared:.4f}, "
          f"compression={a.get('compression_ratio', 0):.2f}")

    if not (0.1 <= alpha <= 3.0):
        errors.append(f"{label}: alpha={alpha:.3f} outside plausible range [0.1, 3.0]")

    if r_squared < 0.0 or r_squared > 1.0:
        errors.append(f"{label}: R^2={r_squared:.4f} outside [0, 1]")

    # Check piecewise fit
    pw = a.get("piecewise_fit", {})
    for region in ["head", "body", "tail"]:
        if region not in pw:
            errors.append(f"{label}: missing piecewise region '{region}'")

    # Check compression ratio
    cr = a.get("compression_ratio", 0)
    if cr <= 0:
        errors.append(f"{label}: compression_ratio={cr} <= 0")

# --- Check corpus type diversity ---
if "natural_language" not in corpus_types_seen:
    errors.append("No natural language corpora analyzed")
if "code" not in corpus_types_seen:
    errors.append("No code corpora analyzed")

# --- Check correlation ---
if correlation:
    print(f"\nCorrelation: Pearson r={correlation.get('pearson_r', 0):.4f}, "
          f"Spearman rho={correlation.get('spearman_r', 0):.4f}")
else:
    errors.append("No correlation analysis results")

# --- Check figures ---
expected_figures = [
    "results/figures/piecewise_exponents.png",
    "results/figures/correlation_alpha_compression.png",
]
for fig in expected_figures:
    if not os.path.exists(fig):
        errors.append(f"Missing figure: {fig}")
    else:
        print(f"  Figure OK: {fig}")

# Count total figures
fig_dir = "results/figures"
if os.path.isdir(fig_dir):
    fig_count = len([f for f in os.listdir(fig_dir) if f.endswith(".png")])
    print(f"\nTotal figures: {fig_count}")
    if fig_count < 3:
        errors.append(f"Expected >= 3 figures, got {fig_count}")

# --- Check report ---
if not os.path.exists("results/report.md"):
    errors.append("Missing results/report.md")
else:
    with open("results/report.md") as f:
        report = f.read()
    if len(report) < 200:
        errors.append(f"Report too short ({len(report)} chars)")
    else:
        print(f"Report: {len(report)} chars")

# --- Summary ---
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
