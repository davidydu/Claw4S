"""Run the full benchmark correlation analysis.

Usage: .venv/bin/python run.py
"""
import os

if not os.path.exists("src/data.py"):
    print("ERROR: Must run from submissions/benchmark-corr/ directory")
    raise SystemExit(1)

os.makedirs("results/figures", exist_ok=True)

from src.analysis import run_full_analysis
from src.plots import generate_all_plots
from src.report import generate_report, save_report

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


print("[1/4] Running analysis...")
results = run_full_analysis(seed=42)

print("[2/4] Generating figures...")
generate_all_plots(results)

print("[3/4] Generating report...")
report = generate_report(results)
save_report(report)

print("[4/4] Saving results to results/")
with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(report)
