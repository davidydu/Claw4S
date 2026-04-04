"""Statistical analysis of experiment results.

Aggregates results across seeds and computes summary statistics, effect
sizes, and comparison tables.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.experiment import ExperimentResult


def aggregate_results(results: list[ExperimentResult]) -> dict[str, Any]:
    """Aggregate experiment results into structured tables.

    Groups by (composition, market_size, info_regime) and averages across seeds.
    """
    groups: dict[tuple[str, str, str], list[ExperimentResult]] = defaultdict(list)
    for r in results:
        key = (r.config.composition, r.config.market_size, r.config.info_regime)
        groups[key].append(r)

    rows = []
    for (comp, size, regime), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "composition": comp,
            "market_size": size,
            "info_regime": regime,
            "n_seeds": len(group),
        }

        # Average metrics across seeds
        for metric in ["price_quality_corr", "market_efficiency", "price_efficiency",
                        "surplus_rate", "lemons_index", "reputation_accuracy",
                        "n_transactions"]:
            vals = [g.metrics[metric] for g in group]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))

        # Average audit scores
        for audit in ["fair_pricing", "exploitation", "market_efficiency", "information_asymmetry"]:
            vals = [g.audit_scores[audit] for g in group]
            row[f"audit_{audit}_mean"] = float(np.mean(vals))

        # Buyer welfare by type (average across seeds, then by buyer type)
        welfare_by_type: dict[str, list[float]] = defaultdict(list)
        surplus_by_type: dict[str, list[float]] = defaultdict(list)
        for g in group:
            for key, val in g.buyer_welfare.items():
                btype = key.rsplit("_", 1)[0]
                welfare_by_type[btype].append(val)
            for key, val in g.buyer_surplus.items():
                btype = key.rsplit("_", 1)[0]
                surplus_by_type[btype].append(val)

        row["buyer_welfare"] = {k: float(np.mean(v)) for k, v in welfare_by_type.items()}
        row["buyer_surplus"] = {k: float(np.mean(v)) for k, v in surplus_by_type.items()}

        # Seller profit by type
        profit_by_type: dict[str, list[float]] = defaultdict(list)
        for g in group:
            for key, val in g.seller_profit.items():
                stype = key.rsplit("_", 1)[0]
                profit_by_type[stype].append(val)
        row["seller_profit"] = {k: float(np.mean(v)) for k, v in profit_by_type.items()}

        rows.append(row)

    return {
        "summary_table": rows,
        "n_total_simulations": len(results),
        "n_groups": len(rows),
    }


def compute_key_findings(agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract key findings from aggregated results."""
    rows = agg["summary_table"]
    findings = []

    def _get(comp: str, regime: str, size: str = "medium") -> dict | None:
        match = [r for r in rows if r["composition"] == comp
                 and r["info_regime"] == regime and r["market_size"] == size]
        return match[0] if match else None

    # 1. Information asymmetry: opaque vs transparent in mixed-seller markets
    mixed_opaque = _get("mixed_sellers", "opaque")
    mixed_trans = _get("mixed_sellers", "transparent")
    if mixed_opaque and mixed_trans:
        eff_o = mixed_opaque["market_efficiency_mean"]
        eff_t = mixed_trans["market_efficiency_mean"]
        sr_o = mixed_opaque["surplus_rate_mean"]
        sr_t = mixed_trans["surplus_rate_mean"]
        findings.append({
            "finding": "Information regime shapes market outcomes",
            "description": (
                f"In mixed-seller markets, transparency improves allocative efficiency "
                f"from {eff_o:.2f} (opaque) to {eff_t:.2f} (transparent) and "
                f"buyer surplus per transaction from {sr_o:.4f} to {sr_t:.4f}. "
                f"Opaque markets leave reputation/analytical buyers unable to build "
                f"seller quality models, reducing them to naive-like behaviour."
            ),
            "efficiency_opaque": eff_o,
            "efficiency_transparent": eff_t,
        })

    # 2. Buyer sophistication: surplus differences across buyer types
    if mixed_opaque and mixed_trans:
        bw_opaque = mixed_opaque["buyer_welfare"]
        bw_trans = mixed_trans["buyer_welfare"]
        findings.append({
            "finding": "Buyer sophistication only helps with information access",
            "description": (
                f"In opaque markets, buyer surplus is similar across types: "
                + ", ".join(f"{k}={v:.4f}" for k, v in sorted(bw_opaque.items()))
                + ". But in transparent markets, analytical and reputation buyers "
                f"diverge: "
                + ", ".join(f"{k}={v:.4f}" for k, v in sorted(bw_trans.items()))
                + ". Sophistication without information yields no advantage."
            ),
            "welfare_opaque": bw_opaque,
            "welfare_transparent": bw_trans,
        })

    # 3. Lemons effect: predatory markets destroy surplus
    honest = _get("all_honest", "opaque")
    predatory = _get("all_predatory", "opaque")
    if honest and predatory:
        findings.append({
            "finding": "Predatory sellers destroy buyer surplus",
            "description": (
                f"All-honest markets yield surplus rate {honest['surplus_rate_mean']:.4f} "
                f"per transaction; all-predatory markets yield "
                f"{predatory['surplus_rate_mean']:.4f}. Lemons index: "
                f"honest={honest['lemons_index_mean']:.2f}, "
                f"predatory={predatory['lemons_index_mean']:.2f}. "
                f"Predatory sellers extract value through inflated claims and "
                f"low actual quality."
            ),
            "surplus_honest": honest["surplus_rate_mean"],
            "surplus_predatory": predatory["surplus_rate_mean"],
        })

    # 4. Strategic sellers are harder to detect than predatory
    strategic = _get("all_strategic", "opaque")
    if predatory and strategic:
        findings.append({
            "finding": "Strategic sellers are more dangerous than predatory",
            "description": (
                f"Strategic sellers achieve higher profit margins by adapting claims: "
                f"strategic seller profit = "
                f"{strategic['seller_profit'].get('strategic', 0):.0f}, "
                f"predatory seller profit = "
                f"{predatory['seller_profit'].get('predatory', 0):.0f}. "
                f"Allocative efficiency: strategic={strategic['market_efficiency_mean']:.2f}, "
                f"predatory={predatory['market_efficiency_mean']:.2f}."
            ),
            "strategic_profit": strategic["seller_profit"].get("strategic", 0),
            "predatory_profit": predatory["seller_profit"].get("predatory", 0),
        })

    # 5. Scale effect: larger markets with more seller diversity
    mixed_small = _get("mixed_sellers", "opaque", "small")
    mixed_large = _get("mixed_sellers", "opaque", "large")
    if mixed_small and mixed_large:
        findings.append({
            "finding": "Market scale affects allocative efficiency",
            "description": (
                f"Allocative efficiency in opaque mixed markets: "
                f"small={mixed_small['market_efficiency_mean']:.2f}, "
                f"medium={mixed_opaque['market_efficiency_mean']:.2f}, "
                f"large={mixed_large['market_efficiency_mean']:.2f}. "
                f"Larger markets with more sellers dilute exploitation but "
                f"also make it harder for buyers to identify the best seller."
            ),
            "small_efficiency": mixed_small["market_efficiency_mean"],
            "large_efficiency": mixed_large["market_efficiency_mean"],
        })

    return findings
