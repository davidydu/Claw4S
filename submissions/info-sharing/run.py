"""Run the full information-sharing experiment and generate report."""

import json
import os

from src.experiment import run_experiment
from src.analysis import run_analysis
from src.report import generate_report

if __name__ == "__main__":
    # Run experiment (multiprocessing)
    run_experiment()

    # Analyze results
    print("[4/5] Running statistical analysis...")
    analysis = run_analysis()

    # Save analysis
    os.makedirs("results", exist_ok=True)
    with open("results/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("[5/5] Generating report...")

    # Generate report
    report = generate_report(analysis)
    with open("results/report.md", "w") as f:
        f.write(report)
    print(report)
    print("\nDone. Results in results/")
