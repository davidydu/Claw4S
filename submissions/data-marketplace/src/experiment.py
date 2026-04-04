"""Experiment configuration and runner.

Defines the full experiment matrix: compositions x market sizes x
information regimes x seeds.  Each configuration is run as an
independent simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.environment import DataEnvironment
from src.sellers import create_seller, BaseSeller
from src.buyers import create_buyer, BaseBuyer
from src.market import DataMarketplace
from src.auditors import AuditPanel


# ── Market compositions ───────────────────────────────────────────
COMPOSITIONS: dict[str, dict[str, list[tuple[str, float]]]] = {
    "all_honest": {
        "sellers": [("honest", 0.9), ("honest", 0.7), ("honest", 0.5)],
        "buyers": [("naive", None), ("reputation", None), ("analytical", None)],
    },
    "all_strategic": {
        "sellers": [("strategic", 0.4), ("strategic", 0.5), ("strategic", 0.3)],
        "buyers": [("naive", None), ("reputation", None), ("analytical", None)],
    },
    "all_predatory": {
        "sellers": [("predatory", 0.1), ("predatory", 0.15), ("predatory", 0.05)],
        "buyers": [("naive", None), ("reputation", None), ("analytical", None)],
    },
    "mixed_sellers": {
        "sellers": [("honest", 0.9), ("strategic", 0.4), ("predatory", 0.1)],
        "buyers": [("naive", None), ("reputation", None), ("analytical", None)],
    },
    "naive_buyers": {
        "sellers": [("honest", 0.9), ("strategic", 0.4), ("predatory", 0.1)],
        "buyers": [("naive", None), ("naive", None), ("naive", None)],
    },
    "analytical_buyers": {
        "sellers": [("honest", 0.9), ("strategic", 0.4), ("predatory", 0.1)],
        "buyers": [("analytical", None), ("analytical", None), ("analytical", None)],
    },
}

MARKET_SIZES = {
    "small": (2, 2),   # (n_sellers, n_buyers) — use first N from composition
    "medium": (3, 3),
    "large": (5, 5),
}

INFO_REGIMES = ["transparent", "opaque", "partial"]

SEEDS = [42, 123, 456]

N_ROUNDS = 10_000
N_STATES = 5
TRUE_DIST = np.array([0.05, 0.10, 0.15, 0.30, 0.40])


# ── Experiment config ─────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """Single experiment configuration."""

    composition: str
    market_size: str
    info_regime: str
    seed: int
    n_rounds: int = N_ROUNDS

    @property
    def name(self) -> str:
        return f"{self.composition}__{self.market_size}__{self.info_regime}__seed{self.seed}"


@dataclass
class ExperimentResult:
    """Output of a single simulation run."""

    config: ExperimentConfig
    metrics: dict[str, Any]
    audit_scores: dict[str, float]
    buyer_welfare: dict[str, float]
    buyer_surplus: dict[str, float]
    seller_profit: dict[str, float]


# ── Runner ────────────────────────────────────────────────────────
def _expand_agents(comp_spec: list[tuple[str, float]], target_n: int,
                   seed_base: int) -> list[tuple[str, float, int]]:
    """Expand a composition spec to target_n agents, cycling if needed.

    Returns list of (type, quality_or_None, rng_seed).
    """
    result = []
    for i in range(target_n):
        agent_type, quality = comp_spec[i % len(comp_spec)]
        result.append((agent_type, quality, seed_base + i * 100))
    return result


def run_simulation(config: ExperimentConfig) -> ExperimentResult:
    """Run a single marketplace simulation.

    This function is designed to be called from a multiprocessing Pool.
    """
    comp = COMPOSITIONS[config.composition]
    n_sellers, n_buyers = MARKET_SIZES[config.market_size]

    rng = np.random.default_rng(config.seed)

    env = DataEnvironment(
        n_states=N_STATES,
        rng=rng,
        true_dist=TRUE_DIST.copy(),
    )

    # Create sellers
    seller_specs = _expand_agents(comp["sellers"], n_sellers, config.seed * 1000)
    sellers: list[BaseSeller] = []
    for i, (stype, quality, s_seed) in enumerate(seller_specs):
        sellers.append(create_seller(stype, seller_id=i, quality=quality,
                                     rng=np.random.default_rng(s_seed)))

    # Create buyers
    buyer_specs = _expand_agents(comp["buyers"], n_buyers, config.seed * 2000)
    buyers: list[BaseBuyer] = []
    for i, (btype, _, b_seed) in enumerate(buyer_specs):
        buyers.append(create_buyer(btype, buyer_id=i, n_states=N_STATES,
                                   rng=np.random.default_rng(b_seed)))

    # Run market
    market = DataMarketplace(env, sellers, buyers, info_regime=config.info_regime)
    market.run(config.n_rounds)

    # Audit
    panel = AuditPanel()
    audit_results = panel.audit(market)
    audit_scores = panel.summary(audit_results)

    # Buyer welfare and surplus
    buyer_welfare: dict[str, float] = {}
    buyer_surplus: dict[str, float] = {}
    for b in buyers:
        key = f"{b.buyer_type}_{b.buyer_id}"
        buyer_welfare[key] = b.welfare
        buyer_surplus[key] = b.total_value - b.total_spent

    # Seller profit
    seller_profit: dict[str, float] = {}
    for s in sellers:
        key = f"{s.seller_type}_{s.seller_id}"
        seller_profit[key] = s.profit

    metrics = {
        "price_quality_corr": market.price_quality_correlation(),
        "market_efficiency": market.market_efficiency(),
        "price_efficiency": market.price_efficiency(),
        "surplus_rate": market.surplus_rate(),
        "lemons_index": market.lemons_index(),
        "reputation_accuracy": market.reputation_accuracy(),
        "n_transactions": len(market.transactions),
    }

    return ExperimentResult(
        config=config,
        metrics=metrics,
        audit_scores=audit_scores,
        buyer_welfare=buyer_welfare,
        buyer_surplus=buyer_surplus,
        seller_profit=seller_profit,
    )


def build_experiment_matrix() -> list[ExperimentConfig]:
    """Build the full experiment matrix.

    6 compositions x 3 sizes x 3 regimes x 3 seeds = 162 configs.
    """
    configs = []
    for comp in COMPOSITIONS:
        for size in MARKET_SIZES:
            for regime in INFO_REGIMES:
                for seed in SEEDS:
                    configs.append(ExperimentConfig(
                        composition=comp,
                        market_size=size,
                        info_regime=regime,
                        seed=seed,
                    ))
    return configs
