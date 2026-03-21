# run.py
"""Run the full pricing collusion experiment and generate report.

Uses multiprocessing to parallelize simulations across CPU cores.
Writes progress incrementally to results/progress.json.
"""

import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

from src.experiment import ExperimentConfig, run_simulation, MATCHUPS
from src.market import LogitMarket
from src.auditors import AuditorPanel
from src.analysis import compute_statistics
from src.report import generate_report, generate_figures

MEMORIES = [1, 3, 5]
PRESETS = ["e-commerce", "ride-share", "commodity"]
SEEDS = list(range(3))
SHOCK_CONDITIONS = [False, True]
# Adaptive rounds: learning matchups get 200k for stronger convergence,
# non-learning matchups get 100k (sufficient for their dynamics).
ROUNDS_BY_MATCHUP = {
    "QQ": 200_000,
    "SS": 200_000,
    "QS": 200_000,
    "PG-PG": 100_000,
    "Q-TFT": 100_000,
    "Q-Competitive": 100_000,
}


def build_configs():
    """Build the full experiment matrix with adaptive round counts."""
    configs = []
    for matchup in MATCHUPS:
        total_rounds = ROUNDS_BY_MATCHUP[matchup]
        for memory in MEMORIES:
            for preset in PRESETS:
                for shocks in SHOCK_CONDITIONS:
                    for seed in SEEDS:
                        configs.append(ExperimentConfig(
                            matchup=matchup, memory=memory, preset=preset,
                            shocks=shocks, seed=seed,
                            total_rounds=total_rounds,
                        ))
    return configs


def _run_and_audit(config):
    """Run one simulation + auditor panel. Returns a plain dict (picklable)."""
    result = run_simulation(config)
    market = LogitMarket.from_preset(config.preset)
    panel = AuditorPanel()
    audit_results = panel.audit_all(
        result.price_history, market,
        agents=result.agents,
        saved_states=result.saved_states,
    )
    return {
        "matchup": config.matchup,
        "memory": config.memory,
        "preset": config.preset,
        "shocks": config.shocks,
        "seed": config.seed,
        "final_avg_price": result.final_avg_price,
        "nash_price": result.nash_price,
        "monopoly_price": result.monopoly_price,
        "convergence_round": result.convergence_round,
        "pre_shock_price": result.pre_shock_price,
        "post_shock_price": result.post_shock_price,
        "recovery_rounds": result.recovery_rounds,
        "auditor_scores": {r.auditor_name: r.collusion_score
                           for r in audit_results},
        "panel_majority": panel.aggregate(audit_results, "majority"),
        "panel_unanimous": panel.aggregate(audit_results, "unanimous"),
        "panel_weighted": panel.aggregate(audit_results, "weighted"),
    }


def main():
    os.makedirs("results", exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    configs = build_configs(total_rounds=100_000)
    total = len(configs)
    n_workers = min(cpu_count() or 4, 8)
    print(f"[1/3] Running {total} simulations on {n_workers} workers...")

    t0 = time.time()
    records = []

    with Pool(n_workers) as pool:
        for i, record in enumerate(pool.imap_unordered(_run_and_audit, configs)):
            records.append(record)
            done = i + 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                rate = elapsed / done
                eta = rate * (total - done) / 60
                print(f"  [{done}/{total}] "
                      f"{record['matchup']}/M{record['memory']}/{record['preset']} "
                      f"| {elapsed/60:.1f}m elapsed | ~{eta:.0f}m remaining",
                      flush=True)
                # Write progress file
                with open("results/progress.json", "w") as f:
                    json.dump({"completed": done, "total": total,
                               "percent": round(100*done/total, 1),
                               "elapsed_min": round(elapsed/60, 1),
                               "est_remaining_min": round(eta, 1)}, f)

    elapsed_total = time.time() - t0
    print(f"\n  All simulations done in {elapsed_total/60:.1f} min")

    # Sort records for deterministic output regardless of worker completion order
    records.sort(key=lambda r: (r["matchup"], r["memory"], r["preset"],
                                r["shocks"], r["seed"]))

    print("[2/3] Computing statistics and generating report...")
    statistics = compute_statistics(records)
    analysis = {"records": records, "statistics": statistics}
    report = generate_report(analysis)
    generate_figures(analysis)

    print("[3/3] Saving results...")
    serializable = {
        "metadata": {
            "num_simulations": total,
            "num_conditions": len(statistics),
            "matchups": list(MATCHUPS.keys()),
            "memories": MEMORIES,
            "presets": PRESETS,
            "seeds": SEEDS,
            "rounds_by_matchup": ROUNDS_BY_MATCHUP,
        },
        "records": records,
        "statistics": statistics,
    }

    with open("results/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    with open("results/report.md", "w") as f:
        f.write(report)
    with open("results/statistical_tests.json", "w") as f:
        json.dump(statistics, f, indent=2)

    # Clean up progress file
    if os.path.exists("results/progress.json"):
        os.remove("results/progress.json")

    print(f"\nDone. Results saved to results/")
    print(f"  results/results.json ({total} records)")
    print(f"  results/report.md")
    print(f"  results/statistical_tests.json")
    print(f"  Total time: {elapsed_total/60:.1f} min ({n_workers} workers)")


if __name__ == "__main__":
    main()
