"""Run the full loss curve universality analysis.

Must be run from the submissions/loss-curves/ directory.
"""

import os
import sys

# Guard: must be run from the submission directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: run.py must be executed from submissions/loss-curves/")
    sys.exit(1)

from src.analysis import run_analysis, save_results
from src.plotting import (
    load_full_curves,
    plot_loss_curves_with_fits,
    plot_aic_comparison,
    plot_exponent_distributions,
    generate_report,
)

print("=" * 60)
print("Loss Curve Universality Analysis")
print("=" * 60)

# Phase 1: Train models and fit curves
print("\n[1/4] Training models and fitting curves...")
results = run_analysis()

# Phase 2: Save results
print("\n[2/4] Saving results...")
save_results(results)

# Phase 3: Generate plots
print("\n[3/4] Generating plots...")
curves = load_full_curves()
plot_loss_curves_with_fits(curves)
plot_aic_comparison(curves)
plot_exponent_distributions()

# Phase 4: Generate and print report
print("\n[4/4] Generating report...")
report = generate_report()
report_path = os.path.join("results", "report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"  Saved {report_path}")
print()
print(report)
