# run.py
"""Run the full Chinese cosmology consistency analysis.

Pipeline:
  [1/5] Generate ~263K birth chart datetimes spanning a 60-year 甲子 cycle
  [2/5] Run BaZi, Zi Wei Dou Shu, and Wu Xing agents on each chart
  [3/5] Evaluate cross-system consistency via the evaluator panel
  [4/5] Generate report (report.md) and 5 figures (results/figures/)
  [5/5] Save results.json and statistical_tests.json to results/

Expected runtime: ~45 minutes on a single CPU (263K charts × ~0.01 s each).
All commands must be run from the submission directory.
"""

import json
import os

from src.experiment import build_chart_configs, run_chart_analysis
from src.analysis import analyze_results
from src.report import generate_report, generate_figures


def build_configs(start_year: int = 1984, end_year: int = 2044):
    """Build the full list of birth datetime configs for the experiment.

    Delegates to build_chart_configs() for easy smoke-testing:
        python -c "from run import build_configs; print(f'{len(build_configs(2000, 2001))} configs')"

    Args:
        start_year: first year to include (Jan 1)
        end_year:   first year to exclude

    Returns:
        list of datetime.datetime objects
    """
    return build_chart_configs(start_year=start_year, end_year=end_year)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate chart datetimes
    # ------------------------------------------------------------------
    print("[1/5] Generating birth chart datetimes (1984–2044)...")
    configs = build_configs(start_year=1984, end_year=2044)
    total = len(configs)
    print(f"  {total:,} datetimes generated")

    # ------------------------------------------------------------------
    # Step 2: Run all 3 agents on each chart
    # ------------------------------------------------------------------
    print(f"[2/5] Running BaZi + Zi Wei + Wu Xing agents on {total:,} charts...")
    chart_results = []
    report_every = max(1, total // 60)  # report ~60 times (once per year approx)

    for i, dt in enumerate(configs):
        chart_results.append(run_chart_analysis(dt))
        if (i + 1) % report_every == 0 or i == 0 or i == total - 1:
            year = dt.year
            pct = (i + 1) / total * 100
            print(f"  {i + 1:,}/{total:,} ({pct:.0f}%) — {year}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate cross-system consistency
    # ------------------------------------------------------------------
    print("[3/5] Evaluating cross-system consistency...")
    analysis = analyze_results(chart_results)
    stats = analysis["statistics"]
    n_domains = len(stats.get("correlation", {}))
    print(f"  {n_domains} domains evaluated")

    # ------------------------------------------------------------------
    # Step 4: Generate report and figures
    # ------------------------------------------------------------------
    print("[4/5] Generating report and figures...")
    report = generate_report(analysis)
    generate_figures(analysis, output_dir="results/figures")
    print("  5 figures saved to results/figures/")

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    print("[5/5] Saving results to results/")

    # Flatten records for JSON serialization
    def _serialize_record(r):
        """Flatten chart result to a JSON-serializable dict."""
        out = {"datetime": r.get("datetime", "")}
        for system in ["bazi", "ziwei", "wuxing"]:
            scores = r.get(f"{system}_career", None)
            for domain in ["career", "wealth", "relationships", "health", "overall"]:
                key = f"{system}_{domain}"
                if key in r:
                    out[key] = r[key]
        return out

    # Save results.json
    serializable = {
        "metadata": {
            "num_charts": total,
            "start_year": 1984,
            "end_year": 2044,
            "systems": ["bazi", "ziwei", "wuxing"],
            "domains": ["career", "wealth", "relationships", "health", "overall"],
        },
        "records": [_serialize_record(r) for r in analysis["records"]],
        "statistics_summary": {
            "n_records": stats.get("n_records", total),
            "correlation": stats.get("correlation", {}),
            "domain_agreement": stats.get("domain_agreement", {}),
        },
    }

    with open("results/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Save report.md
    with open("results/report.md", "w") as f:
        f.write(report)

    # Save statistical_tests.json
    stat_tests = {
        "correlation": stats.get("correlation", {}),
        "domain_agreement": stats.get("domain_agreement", {}),
        "mutual_information": stats.get("mutual_information", {}),
        "conditional_agreement": stats.get("conditional_agreement", {}),
        "n_records": stats.get("n_records", 0),
        "temporal_patterns": stats.get("temporal_patterns", []),
    }
    with open("results/statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2, default=str)

    print(f"\nDone. Results saved to results/")
    print(f"  results/results.json ({total:,} chart records)")
    print(f"  results/report.md")
    print(f"  results/statistical_tests.json")
    print(f"  results/figures/ (5 PNGs)")


if __name__ == "__main__":
    main()
