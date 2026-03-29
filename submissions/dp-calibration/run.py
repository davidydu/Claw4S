"""Run the full DP noise calibration comparison analysis.

Computes privacy loss (epsilon) across a grid of noise multipliers,
composition steps, and failure probabilities using four accounting
methods: naive composition, advanced composition, Renyi DP, and
Gaussian DP. Saves results and generates visualizations.

Must be run from the submission directory:
    .venv/bin/python run.py
"""

import argparse
import os
import sys

# --- Working-directory guard ---
_expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "SKILL.md")
if not os.path.isfile(_expected_marker):
    print("ERROR: run.py must be executed from the dp-calibration/ "
          "submission directory.", file=sys.stderr)
    sys.exit(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --------------------------------

from src.analysis import run_analysis, save_results
from src.visualize import generate_all_figures


def _parse_csv_values(raw: str | None, cast, flag_name: str):
    """Parse comma-separated CLI values to a typed list."""
    if raw is None:
        return None
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(cast(token))
        except ValueError as exc:
            raise ValueError(f"{flag_name} must be a comma-separated list") from exc
    if not values:
        raise ValueError(f"{flag_name} must include at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configurable sweeps."""
    parser = argparse.ArgumentParser(
        description=(
            "Run DP accounting comparison across a parameter grid and save results."
        )
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for deterministic metadata")
    parser.add_argument("--t-values", type=str, default=None,
                        help="Comma-separated composition steps, e.g. 10,100,1000")
    parser.add_argument("--delta-values", type=str, default=None,
                        help="Comma-separated deltas, e.g. 1e-5,1e-6,1e-7")
    parser.add_argument("--sigma-values", type=str, default=None,
                        help="Comma-separated sigmas, e.g. 0.1,0.5,1,2,5,10")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for results JSON + figures")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation (JSON only)")
    return parser.parse_args()


def main():
    args = _parse_args()

    try:
        t_values = _parse_csv_values(args.t_values, int, "--t-values")
        delta_values = _parse_csv_values(args.delta_values, float, "--delta-values")
        sigma_values = _parse_csv_values(args.sigma_values, float, "--sigma-values")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("DP Noise Calibration Comparison")
    print("=" * 60)
    print()

    # Step 1: Run the parameter sweep
    print("[1/3] Running parameter sweep across (T, delta, sigma) grid...")
    try:
        data = run_analysis(
            seed=args.seed,
            t_values=t_values,
            delta_values=delta_values,
            sigma_values=sigma_values,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    meta = data["metadata"]
    print(f"      Grid: {meta['num_T']} T values x "
          f"{meta['num_delta']} delta values x "
          f"{meta['num_sigma']} sigma values")
    print(f"      Methods: {meta['num_methods']}")
    print(f"      Total computations: {meta['total_computations']}")
    print(f"      Elapsed: {meta['elapsed_seconds']:.3f}s")
    print()

    # Step 2: Save results
    print(f"[2/3] Saving results to {args.output_dir}/results.json...")
    path = save_results(data, output_dir=args.output_dir)
    print(f"      Saved to: {path}")
    print()

    # Step 3: Generate visualizations
    if args.skip_figures:
        print("[3/3] Skipping visualizations (--skip-figures set).")
    else:
        print("[3/3] Generating visualizations...")
        fig_paths = generate_all_figures(data, output_dir=args.output_dir)
        for p in fig_paths:
            print(f"      Saved: {p}")
    print()

    # Summary
    summary = data["summary"]
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Method Win Counts (tightest bound):")
    for method, count in summary["win_counts"].items():
        pct = 100 * count / meta["total_configs"] if meta["total_configs"] > 0 else 0
        print(f"  {method:12s}: {count:3d} / {meta['total_configs']} ({pct:.1f}%)")
    print()
    print("Average Tightness Ratio (lower = tighter, 1.0 = optimal):")
    for method in summary["avg_tightness_ratio"]:
        avg = summary["avg_tightness_ratio"][method]
        std = summary["std_tightness_ratio"][method]
        if avg == float("inf") or avg == "Infinity":
            print(f"  {method:12s}: inf")
        else:
            print(f"  {method:12s}: {avg:.4f} +/- {std:.4f}")
    print()
    print("Median / 95th-Percentile Tightness Ratio:")
    for method in summary["avg_tightness_ratio"]:
        median = summary["median_tightness_ratio"][method]
        p95 = summary["p95_tightness_ratio"][method]
        if median == float("inf") or median == "Infinity":
            print(f"  {method:12s}: inf")
        else:
            print(f"  {method:12s}: median={median:.4f}, p95={p95:.4f}")
    print()
    print("Wins by Composition Steps (T):")
    for T_str, wins in summary["wins_by_T"].items():
        winners = [f"{m}={c}" for m, c in wins.items() if c > 0]
        print(f"  T={T_str:>5s}: {', '.join(winners)}")
    print()
    print(f"Results digest (SHA256): {meta['results_digest']}")
    print("Done.")


if __name__ == "__main__":
    main()
