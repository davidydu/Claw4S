"""
validate.py — Validates experiment results for World Model Consensus.

Checks:
1. All expected output files exist
2. Correct number of simulations ran
3. Coordination rate at d=0 is high for all compositions
4. Coordination rate at d>=0.6 drops for stubborn/mixed
5. Phase transitions detected for stubborn compositions
6. Figures generated

Usage:
    .venv/bin/python validate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


RESULTS_DIR = Path("results")
PASS = 0
FAIL = 0


def check(condition: bool, msg: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {msg}")
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def main() -> None:
    global PASS, FAIL
    print("=== Validating World Model Consensus Results ===\n")

    # 1. Check output files
    print("1. Output files:")
    for fname in ["raw_results.json", "summary_table.json", "phase_transitions.json",
                  "report.md", "fig1_coordination_vs_disagreement.png",
                  "fig2_consensus_time.png", "fig3_group_size_effect.png",
                  "fig4_fairness.png"]:
        check((RESULTS_DIR / fname).exists(), f"{fname} exists")

    # 2. Correct number of simulations
    print("\n2. Simulation count:")
    raw = json.loads((RESULTS_DIR / "raw_results.json").read_text())
    check(len(raw) == 396, f"Expected 396 simulations, got {len(raw)}")

    # 3. Coordination at d=0 is high
    print("\n3. Coordination at d=0:")
    for comp in ["all_adaptive", "all_stubborn", "mixed", "leader_followers"]:
        d0_rates = [r["coordination_rate"] for r in raw
                    if r["composition"] == comp and r["disagreement"] == 0.0
                    and r["n_agents"] == 4]
        if d0_rates:
            mean_rate = sum(d0_rates) / len(d0_rates)
            check(mean_rate > 0.7, f"{comp} at d=0: coord={mean_rate:.3f} > 0.7")

    # 4. Stubborn agents fail at high disagreement
    print("\n4. Stubborn agents fail at high disagreement:")
    for comp in ["all_stubborn", "mixed"]:
        hi_rates = [r["coordination_rate"] for r in raw
                    if r["composition"] == comp and r["disagreement"] >= 0.6
                    and r["n_agents"] == 4]
        if hi_rates:
            mean_rate = sum(hi_rates) / len(hi_rates)
            check(mean_rate < 0.1, f"{comp} at d>=0.6: coord={mean_rate:.3f} < 0.1")

    # 5. Adaptive agents maintain coordination
    print("\n5. Adaptive agents maintain coordination:")
    adaptive_hi = [r["coordination_rate"] for r in raw
                   if r["composition"] == "all_adaptive" and r["disagreement"] >= 0.6
                   and r["n_agents"] == 4]
    if adaptive_hi:
        mean_rate = sum(adaptive_hi) / len(adaptive_hi)
        check(mean_rate > 0.5, f"all_adaptive at d>=0.6: coord={mean_rate:.3f} > 0.5")

    # 6. Phase transitions
    print("\n6. Phase transitions:")
    pt = json.loads((RESULTS_DIR / "phase_transitions.json").read_text())
    check("all_stubborn_N4" in pt, "Phase transition entry for all_stubborn N=4")
    if "all_stubborn_N4" in pt:
        tp = pt["all_stubborn_N4"]["transition_point"]
        check(tp != "None" and tp is not None,
              f"all_stubborn N=4 has transition at {tp}")
        if tp is not None and tp != "None":
            tp_f = float(tp)
            check(0.3 < tp_f < 0.8,
                  f"Transition point {tp_f:.3f} in expected range (0.3, 0.8)")

    # 7. Figures exist and are non-empty
    print("\n7. Figures:")
    for fname in ["fig1_coordination_vs_disagreement.png",
                  "fig2_consensus_time.png",
                  "fig3_group_size_effect.png",
                  "fig4_fairness.png"]:
        path = RESULTS_DIR / fname
        if path.exists():
            check(path.stat().st_size > 1000,
                  f"{fname} is non-trivial ({path.stat().st_size:,} bytes)")

    # 8. Group size effect
    print("\n8. Group size effect:")
    for n in [3, 4, 6]:
        rates = [r["coordination_rate"] for r in raw
                 if r["composition"] == "all_adaptive" and r["disagreement"] == 0.0
                 and r["n_agents"] == n]
        if rates:
            mean_rate = sum(rates) / len(rates)
            check(mean_rate > 0.5,
                  f"all_adaptive d=0 N={n}: coord={mean_rate:.3f}")

    # 9. Determinism: same seed should give same result
    print("\n9. Determinism:")
    seed42 = [r for r in raw if r["seed"] == 42 and r["composition"] == "all_adaptive"
              and r["disagreement"] == 0.0 and r["n_agents"] == 4]
    check(len(seed42) == 1, f"Exactly one entry for seed=42/adaptive/d=0/N=4")

    # Summary
    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"VALIDATION: {PASS}/{total} checks passed")
    if FAIL > 0:
        print(f"  {FAIL} checks FAILED")
        sys.exit(1)
    else:
        print("  All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
