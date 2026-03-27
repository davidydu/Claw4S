# src/analysis.py
"""Statistical analysis and aggregation of experiment results."""

import numpy as np
from scipy import stats

from src.auditors import AuditorPanel
from src.market import LogitMarket


def analyze_results(sim_results):
    """Run auditor panel on all simulation results and compute statistics.

    Args:
        sim_results: list of SimulationResult objects

    Returns:
        dict with full analysis data
    """
    panel = AuditorPanel()
    records = []

    for result in sim_results:
        market = LogitMarket.from_preset(result.config.preset)

        audit_results = panel.audit_all(
            result.price_history, market,
            agents=result.agents,
            saved_states=result.saved_states,
        )

        record = {
            "matchup": result.config.matchup,
            "memory": result.config.memory,
            "preset": result.config.preset,
            "shocks": result.config.shocks,
            "seed": result.config.seed,
            "final_avg_price": result.final_avg_price,
            "nash_price": result.nash_price,
            "monopoly_price": result.monopoly_price,
            "convergence_round": result.convergence_round,
            "pre_shock_price": result.pre_shock_price,
            "post_shock_price": result.post_shock_price,
            "recovery_rounds": result.recovery_rounds,
            "auditor_scores": {r.auditor_name: r.collusion_score
                               for r in audit_results},
            "auditor_evidence": {r.auditor_name: r.evidence
                                 for r in audit_results},
            "panel_majority": panel.aggregate(audit_results, "majority"),
            "panel_unanimous": panel.aggregate(audit_results, "unanimous"),
            "panel_weighted": panel.aggregate(audit_results, "weighted"),
        }
        records.append(record)

    return {
        "records": records,
        "statistics": compute_statistics(records),
    }


def compute_statistics(records):
    """Compute aggregate statistics across seeds for each condition."""
    from collections import defaultdict
    groups = defaultdict(list)

    for r in records:
        key = (r["matchup"], r["memory"], r["preset"], r["shocks"])
        groups[key].append(r)

    stats_out = []
    n_tests = len(groups)  # for Bonferroni correction

    for key, group in sorted(groups.items()):
        matchup, memory, preset, shocks = key
        prices = [r["final_avg_price"] for r in group]
        nash = group[0]["nash_price"]
        monopoly = group[0]["monopoly_price"]

        # Collusion index Delta: (avg_price - nash) / (monopoly - nash)
        avg_price = float(np.mean(prices))
        if monopoly > nash:
            delta = (avg_price - nash) / (monopoly - nash)
        else:
            delta = 0.0

        # One-sample t-test: are prices significantly above Nash?
        if len(prices) > 1 and np.std(prices) > 0:
            t_stat, p_value = stats.ttest_1samp(prices, nash)
            p_value_one = p_value / 2 if t_stat > 0 else 1.0
        else:
            t_stat, p_value_one = 0.0, 1.0

        # Bonferroni-corrected p-value
        p_value_corrected = min(p_value_one * n_tests, 1.0)

        # Cohen's d effect size (using Delta as primary when d is extreme)
        std = np.std(prices, ddof=1) if len(prices) > 1 else 1.0
        cohens_d = (avg_price - nash) / std if std > 1e-8 else 0.0

        # Auditor agreement
        auditor_names = list(group[0]["auditor_scores"].keys())
        agreement_count = 0
        total = len(group)
        for r in group:
            scores = list(r["auditor_scores"].values())
            verdicts = [s > 0.5 for s in scores]
            if all(verdicts) or not any(verdicts):
                agreement_count += 1

        # Collusion rates by panel method
        majority_rate = sum(1 for r in group if r["panel_majority"]) / total
        unanimous_rate = sum(1 for r in group if r["panel_unanimous"]) / total

        stats_out.append({
            "matchup": matchup,
            "memory": memory,
            "preset": preset,
            "shocks": shocks,
            "n_seeds": len(group),
            "avg_price": avg_price,
            "std_price": float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0,
            "nash_price": nash,
            "monopoly_price": monopoly,
            "collusion_index": float(delta),
            "t_statistic": float(t_stat),
            "p_value": float(p_value_one),
            "p_value_corrected": float(p_value_corrected),
            "cohens_d": float(cohens_d),
            "auditor_agreement_rate": agreement_count / total,
            "majority_collusion_rate": majority_rate,
            "unanimous_collusion_rate": unanimous_rate,
            "avg_auditor_scores": {
                name: float(np.mean([r["auditor_scores"][name] for r in group]))
                for name in auditor_names
            },
        })

    return stats_out
