"""Market auditors — detect pricing anomalies, exploitation, inefficiency.

Each auditor examines the transaction history and produces a score in [0, 1]
plus a human-readable verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.market import DataMarketplace, Transaction


@dataclass
class AuditResult:
    """Result from a single auditor."""

    auditor: str
    score: float          # 0.0 = worst, 1.0 = best
    verdict: str          # human-readable
    details: dict[str, Any]


# ── Fair Pricing Auditor ──────────────────────────────────────────
class FairPricingAuditor:
    """Measures correlation between price and actual data quality.

    Score = max(0, Pearson correlation).  Score near 1 means prices
    faithfully reflect quality; 0 means no relationship or inverse.
    """

    name = "fair_pricing"

    def audit(self, market: DataMarketplace) -> AuditResult:
        txns = market.transactions
        if len(txns) < 2:
            return AuditResult(self.name, 0.0, "Insufficient data", {})

        prices = np.array([t.price for t in txns])
        quals = np.array([t.actual_quality for t in txns])

        if np.std(prices) < 1e-12 or np.std(quals) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(prices, quals)[0, 1])

        score = max(0.0, corr)

        if score > 0.7:
            verdict = "Fair pricing: prices reflect data quality"
        elif score > 0.3:
            verdict = "Moderate price-quality alignment"
        else:
            verdict = "WARNING: prices do not reflect actual quality"

        return AuditResult(
            auditor=self.name,
            score=score,
            verdict=verdict,
            details={"pearson_r": corr, "n_transactions": len(txns)},
        )


# ── Exploitation Auditor ─────────────────────────────────────────
class ExploitationAuditor:
    """Detects price gouging — sellers charging far more than quality warrants.

    Exploitation ratio = price / (actual_quality * base_price).
    Score = 1 - fraction of transactions with exploitation ratio > 2.
    """

    name = "exploitation"

    def __init__(self, base_price: float = 0.5) -> None:
        self.base_price = base_price

    def audit(self, market: DataMarketplace) -> AuditResult:
        txns = market.transactions
        if not txns:
            return AuditResult(self.name, 0.0, "No transactions", {})

        exploitation_ratios = []
        exploitative_count = 0
        for t in txns:
            fair_price = t.actual_quality * self.base_price
            ratio = t.price / max(fair_price, 1e-6)
            exploitation_ratios.append(ratio)
            if ratio > 2.0:
                exploitative_count += 1

        frac_exploitative = exploitative_count / len(txns)
        score = 1.0 - frac_exploitative
        mean_ratio = float(np.mean(exploitation_ratios))

        if frac_exploitative < 0.05:
            verdict = "No exploitation detected"
        elif frac_exploitative < 0.3:
            verdict = f"Mild exploitation: {frac_exploitative:.0%} of transactions overpriced"
        else:
            verdict = f"SEVERE exploitation: {frac_exploitative:.0%} of transactions overpriced"

        # Per-buyer-type exploitation
        by_type: dict[str, list[float]] = {}
        for t, r in zip(txns, exploitation_ratios):
            by_type.setdefault(t.buyer_type, []).append(r)
        per_type = {k: float(np.mean(v)) for k, v in by_type.items()}

        return AuditResult(
            auditor=self.name,
            score=score,
            verdict=verdict,
            details={
                "frac_exploitative": frac_exploitative,
                "mean_exploitation_ratio": mean_ratio,
                "per_buyer_type": per_type,
            },
        )


# ── Market Efficiency Auditor ────────────────────────────────────
class MarketEfficiencyAuditor:
    """Measures total welfare relative to optimal.

    Score = total_surplus / max_possible_surplus, where:
    - buyer surplus = decision_value - price
    - seller surplus = price - production_cost
    - total surplus = sum of buyer + seller surplus
    - max surplus = optimal_decision_value * n_transactions (assuming zero cost)
    """

    name = "market_efficiency"

    def audit(self, market: DataMarketplace) -> AuditResult:
        txns = market.transactions
        if not txns:
            return AuditResult(self.name, 0.0, "No transactions", {})

        buyer_surplus = sum(t.decision_value - t.price for t in txns)
        seller_surplus = sum(s.profit for s in market.sellers)
        total_surplus = buyer_surplus + seller_surplus

        opt_val = market.env.optimal_decision_value()
        # Max surplus: if every transaction gave optimal value at zero cost
        max_surplus = opt_val * len(txns)

        score = float(np.clip(total_surplus / max(max_surplus, 1e-6), 0.0, 1.0))

        return AuditResult(
            auditor=self.name,
            score=score,
            verdict=f"Market efficiency: {score:.1%} of optimal surplus realised",
            details={
                "buyer_surplus": buyer_surplus,
                "seller_surplus": seller_surplus,
                "total_surplus": total_surplus,
                "max_surplus": max_surplus,
            },
        )


# ── Information Asymmetry Auditor ────────────────────────────────
class InformationAsymmetryAuditor:
    """Measures the gap between what sellers claim and what they deliver.

    Score = 1 - mean(|claimed_quality - actual_quality|).
    A score of 1.0 means sellers are perfectly truthful.
    """

    name = "information_asymmetry"

    def audit(self, market: DataMarketplace) -> AuditResult:
        txns = market.transactions
        if not txns:
            return AuditResult(self.name, 0.0, "No transactions", {})

        gaps = [abs(t.claimed_quality - t.actual_quality) for t in txns]
        mean_gap = float(np.mean(gaps))
        score = 1.0 - mean_gap

        # Per-seller-type gap
        by_type: dict[str, list[float]] = {}
        for t, g in zip(txns, gaps):
            by_type.setdefault(t.seller_type, []).append(g)
        per_type = {k: float(np.mean(v)) for k, v in by_type.items()}

        if mean_gap < 0.05:
            verdict = "Low information asymmetry — sellers are truthful"
        elif mean_gap < 0.3:
            verdict = f"Moderate asymmetry: mean quality gap = {mean_gap:.2f}"
        else:
            verdict = f"HIGH asymmetry: mean quality gap = {mean_gap:.2f}"

        return AuditResult(
            auditor=self.name,
            score=score,
            verdict=verdict,
            details={"mean_gap": mean_gap, "per_seller_type": per_type},
        )


# ── Audit Panel ───────────────────────────────────────────────────
class AuditPanel:
    """Runs all auditors and aggregates results."""

    def __init__(self) -> None:
        self.auditors = [
            FairPricingAuditor(),
            ExploitationAuditor(),
            MarketEfficiencyAuditor(),
            InformationAsymmetryAuditor(),
        ]

    def audit(self, market: DataMarketplace) -> list[AuditResult]:
        return [a.audit(market) for a in self.auditors]

    def summary(self, results: list[AuditResult]) -> dict[str, float]:
        return {r.auditor: r.score for r in results}
