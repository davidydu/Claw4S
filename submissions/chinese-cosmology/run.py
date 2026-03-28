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
import argparse

from src.experiment import build_chart_configs, run_chart_analysis
from src.analysis import analyze_results
from src.report import generate_report, generate_figures


def _configure_plot_cache(output_dir: str) -> None:
    """Set writable cache directories for matplotlib/fontconfig in restricted envs."""
    cache_root = os.path.join(output_dir, ".cache")
    os.makedirs(cache_root, exist_ok=True)

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if not xdg_cache:
        os.environ["XDG_CACHE_HOME"] = cache_root
        xdg_cache = cache_root

    mpl_config = os.environ.get("MPLCONFIGDIR")
    if not mpl_config:
        mpl_config = os.path.join(xdg_cache, "matplotlib")
        os.environ["MPLCONFIGDIR"] = mpl_config

    os.makedirs(mpl_config, exist_ok=True)


def build_configs(
    start_year: int = 1984,
    end_year: int = 2044,
    max_charts: int | None = None,
):
    """Build the full list of birth datetime configs for the experiment.

    Delegates to build_chart_configs() for easy smoke-testing:
        python -c "from run import build_configs; print(f'{len(build_configs(2000, 2001))} configs')"

    Args:
        start_year: first year to include (Jan 1)
        end_year:   first year to exclude

    Returns:
        list of datetime.datetime objects
    """
    configs = build_chart_configs(start_year=start_year, end_year=end_year)

    if max_charts is None:
        return configs
    if max_charts <= 0:
        raise ValueError("max_charts must be a positive integer when provided")

    return configs[:max_charts]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Chinese cosmology consistency analysis.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1984,
        help="Start year (inclusive). Default: 1984.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2044,
        help="End year (exclusive). Default: 2044.",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=None,
        help="Optional smoke limit: analyze only first N generated charts.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for results outputs. Default: results.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation for faster smoke runs.",
    )
    return parser.parse_args()


def main(
    start_year: int = 1984,
    end_year: int = 2044,
    max_charts: int | None = None,
    output_dir: str = "results",
    skip_figures: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate chart datetimes
    # ------------------------------------------------------------------
    print(f"[1/5] Generating birth chart datetimes ({start_year}–{end_year})...")
    configs = build_configs(
        start_year=start_year,
        end_year=end_year,
        max_charts=max_charts,
    )
    total = len(configs)
    if max_charts is None:
        print(f"  {total:,} datetimes generated")
    else:
        print(f"  {total:,} datetimes selected (max_charts={max_charts:,})")

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
    if skip_figures:
        print("  Figure generation skipped (--skip-figures).")
    else:
        _configure_plot_cache(output_dir)
        generate_figures(analysis, output_dir=figures_dir)
        print(f"  5 figures saved to {figures_dir}/")

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    print(f"[5/5] Saving results to {output_dir}/")

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
            "start_year": start_year,
            "end_year": end_year,
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

    results_json_path = os.path.join(output_dir, "results.json")
    report_path = os.path.join(output_dir, "report.md")
    stats_path = os.path.join(output_dir, "statistical_tests.json")

    with open(results_json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Save report.md
    with open(report_path, "w") as f:
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
    with open(stats_path, "w") as f:
        json.dump(stat_tests, f, indent=2, default=str)

    print(f"\nDone. Results saved to {output_dir}/")
    print(f"  {results_json_path} ({total:,} chart records)")
    print(f"  {report_path}")
    print(f"  {stats_path}")
    if skip_figures:
        print(f"  {figures_dir}/ (skipped)")
    else:
        print(f"  {figures_dir}/ (5 PNGs)")


if __name__ == "__main__":
    args = parse_args()
    main(
        start_year=args.start_year,
        end_year=args.end_year,
        max_charts=args.max_charts,
        output_dir=args.output_dir,
        skip_figures=args.skip_figures,
    )
