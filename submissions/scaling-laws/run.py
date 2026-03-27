"""Run the full scaling laws analysis.

Usage: .venv/bin/python run.py
"""
import os
import warnings

# Suppress overflow/divide-by-zero warnings from curve fitting restarts
warnings.filterwarnings("ignore", category=RuntimeWarning)

if not os.path.exists("src/data.py"):
    print("ERROR: Must run from submissions/scaling-laws/ directory")
    raise SystemExit(1)

os.makedirs("results/figures", exist_ok=True)

from src.analysis import run_full_analysis
from src.plots import generate_all_plots
from src.report import generate_report, save_report

results = run_full_analysis(n_bootstrap=500, seed=42)
generate_all_plots(results)
report = generate_report(results)
save_report(report)
print(report)
