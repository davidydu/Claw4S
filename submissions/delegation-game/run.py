"""Run the full delegation game experiment and generate report."""

if __name__ == "__main__":
    from src.experiment import run_experiment
    from src.report import generate_report

    results = run_experiment()
    report = generate_report(results)
    print(report)
