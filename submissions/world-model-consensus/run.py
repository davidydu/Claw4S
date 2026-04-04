"""
run.py — Main experiment runner for World Model Consensus.

Runs 396 simulations (4 compositions x 11 disagreement levels x
3 group sizes x 3 seeds) with multiprocessing, audits each result,
aggregates metrics, generates figures and a Markdown report.

Usage:
    .venv/bin/python run.py
"""

from __future__ import annotations

import json
import time
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

from src.experiment import (
    SimulationConfig, SimulationResult, run_simulation,
    build_experiment_matrix, COMPOSITIONS, DISAGREEMENT_LEVELS, GROUP_SIZES, SEEDS,
)
from src.auditors import run_audit_panel, AuditResult
from src.analysis import (
    aggregate_over_seeds, AggregatedMetric,
    detect_phase_transition, compute_sharpness, build_summary_table,
)
from src.report import generate_figures, generate_markdown_report


RESULTS_DIR = Path("results")


def _run_one(cfg: SimulationConfig) -> Tuple[SimulationConfig, Dict[str, float]]:
    """Run one simulation and return audit values (picklable)."""
    result = run_simulation(cfg)
    audit = run_audit_panel(result)
    return cfg, {k: v.value for k, v in audit.items()}


def main() -> None:
    configs = build_experiment_matrix()
    print(f"Running {len(configs)} simulations on {cpu_count()} cores...")
    t0 = time.time()

    # Run with multiprocessing
    with Pool() as pool:
        raw_results = pool.map(_run_one, configs)

    elapsed = time.time() - t0
    print(f"Simulations complete in {elapsed:.1f}s")

    # Group by (composition, n_agents, disagreement) -> list of audit dicts
    grouped: Dict[Tuple[str, int, float], List[Dict[str, AuditResult]]] = {}
    for cfg, audit_vals in raw_results:
        key = (cfg.composition, cfg.n_agents, cfg.disagreement)
        # Convert flat values back to AuditResult for aggregation
        ar_dict = {k: AuditResult(name=k, value=v) for k, v in audit_vals.items()}
        grouped.setdefault(key, []).append(ar_dict)

    # Aggregate over seeds
    aggregated: Dict[Tuple[str, int, float], Dict[str, AggregatedMetric]] = {}
    for key, audit_list in grouped.items():
        aggregated[key] = aggregate_over_seeds(audit_list)

    # Save raw results as JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_json = []
    for cfg, audit_vals in raw_results:
        raw_json.append({
            "composition": cfg.composition,
            "n_agents": cfg.n_agents,
            "disagreement": cfg.disagreement,
            "seed": cfg.seed,
            "n_rounds": cfg.n_rounds,
            **audit_vals,
        })
    (RESULTS_DIR / "raw_results.json").write_text(json.dumps(raw_json, indent=2))

    # Save summary table
    summary = build_summary_table(aggregated)
    (RESULTS_DIR / "summary_table.json").write_text(json.dumps(summary, indent=2))

    # Phase transition detection
    print("\n=== Phase Transition Detection ===")
    transitions = {}
    for comp in sorted(set(k[0] for k in aggregated)):
        for n in sorted(set(k[1] for k in aggregated)):
            ds, rates = [], []
            for (c, n2, d), metrics in sorted(aggregated.items()):
                if c == comp and n2 == n and "coordination_rate" in metrics:
                    ds.append(d)
                    rates.append(metrics["coordination_rate"].mean)
            if ds:
                tp = detect_phase_transition(ds, rates)
                sharp = compute_sharpness(ds, rates)
                tp_str = f"{tp:.3f}" if tp is not None else "none"
                print(f"  {comp:20s} N={n}: transition={tp_str}, sharpness={sharp:.3f}")
                transitions[f"{comp}_N{n}"] = {
                    "transition_point": tp,
                    "sharpness": sharp,
                }

    (RESULTS_DIR / "phase_transitions.json").write_text(
        json.dumps(transitions, indent=2, default=str)
    )

    # Print coordination rate table for N=4
    print("\n=== Coordination Rate (N=4, mean over seeds) ===")
    print(f"{'Composition':20s}", end="")
    for d in DISAGREEMENT_LEVELS:
        print(f"  d={d:<5.2f}", end="")
    print()

    for comp in sorted(COMPOSITIONS.keys()):
        print(f"{comp:20s}", end="")
        for d in DISAGREEMENT_LEVELS:
            key = (comp, 4, d)
            if key in aggregated and "coordination_rate" in aggregated[key]:
                val = aggregated[key]["coordination_rate"].mean
                print(f"  {val:6.3f}", end="")
            else:
                print(f"  {'—':>6s}", end="")
        print()

    # Generate figures
    print("\nGenerating figures...")
    figures = generate_figures(aggregated, RESULTS_DIR)
    for f in figures:
        print(f"  {f}")

    # Generate report
    report = generate_markdown_report(aggregated, RESULTS_DIR, figures)
    print(f"\nReport saved to {RESULTS_DIR / 'report.md'}")

    # Final summary
    print(f"\n=== DONE ===")
    print(f"Total simulations: {len(configs)}")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
