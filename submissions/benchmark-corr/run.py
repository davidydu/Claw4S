"""Run the full benchmark correlation analysis.

Usage: .venv/bin/python run.py
"""
import argparse
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


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run LLM benchmark correlation analysis with robustness checks."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic analysis (default: 42).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=400,
        help="Bootstrap samples for uncertainty estimates (default: 400, min: 100).",
    )
    return parser.parse_args()


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


args = parse_args()
if args.bootstrap_samples < 100:
    print("ERROR: --bootstrap-samples must be >= 100")
    raise SystemExit(2)

print("[1/4] Running analysis...")
results = run_full_analysis(seed=args.seed, n_bootstrap=args.bootstrap_samples)

print("[2/4] Generating figures...")
generate_all_plots(results)

print("[3/4] Generating report...")
report = generate_report(results)
save_report(report)

print("[4/4] Saving results to results/")
with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(report)
