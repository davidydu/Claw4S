"""Run the full reward-hacking propagation experiment and generate report."""

from src.experiment import run_experiment
from src.report import generate_report

if __name__ == "__main__":
    print("[1/2] Running experiment sweep (324 simulations)...")
    results = run_experiment()
    print("[2/2] Generating report...")
    report = generate_report(results)
    print(report)
