# src/auditors.py
"""Auditor agents for detecting tacit collusion in price histories."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AuditResult:
    """Result from a single auditor's analysis."""
    auditor_name: str
    collusion_score: float  # 0.0 = competitive, 1.0 = fully collusive
    evidence: dict = field(default_factory=dict)


class BaseAuditor(ABC):
    """Base class for all auditor agents. Subclass and implement audit()."""
    name: str = "base"

    @abstractmethod
    def audit(self, price_history, market, **kwargs):
        """Analyze price history and return an AuditResult."""


class MarginAuditor(BaseAuditor):
    """Auditor 1: Price-Cost Margin Analyzer.

    Scores where the observed markup falls on the Nash -> Monopoly spectrum.
    """
    name = "margin"

    def audit(self, price_history, market):
        # Use final 20% of rounds
        n = len(price_history)
        tail = price_history[int(n * 0.8):]
        avg_prices = np.array([market.price_grid[tail[:, i]].mean()
                               for i in range(market.n_sellers)])
        avg_markup = ((avg_prices - market.costs) / market.costs).mean()

        nash = market.nash_price()
        nash_markup = (nash - market.costs[0]) / market.costs[0]
        monopoly = market.monopoly_price()
        monopoly_markup = (monopoly - market.costs[0]) / market.costs[0]

        if monopoly_markup <= nash_markup:
            score = 0.0
        else:
            score = (avg_markup - nash_markup) / (monopoly_markup - nash_markup)
            score = float(np.clip(score, 0.0, 1.0))

        return AuditResult(
            auditor_name=self.name,
            collusion_score=score,
            evidence={
                "avg_markup": float(avg_markup),
                "nash_markup": float(nash_markup),
                "monopoly_markup": float(monopoly_markup),
            },
        )


class DeviationPunishmentAuditor(BaseAuditor):
    """Auditor 2: Deviation-Punishment Detector.

    Scans for high-price -> deviation -> punishment -> recovery patterns.
    """
    name = "deviation_punishment"

    def audit(self, price_history, market, window=500, threshold=0.3):
        n = len(price_history)
        if n < window * 3:
            return AuditResult(self.name, 0.0,
                               {"reason": "insufficient history"})

        prices = np.array([market.price_grid[price_history[:, i]]
                           for i in range(market.n_sellers)]).T
        nash = market.nash_price()
        episodes = 0
        scanned = 0

        for start in range(0, n - window * 2, window):
            scanned += 1
            seg1 = prices[start:start + window].mean(axis=0)
            seg2 = prices[start + window:start + window * 2].mean(axis=0)

            # Check: was seg1 high, then seg2 dropped (deviation/punishment)?
            high_phase = (seg1 > nash * (1 + threshold)).any()
            drop = ((seg1 - seg2) / seg1 > threshold * 0.5).any()

            if high_phase and drop:
                # Check for recovery after punishment
                if start + window * 3 <= n:
                    seg3 = prices[start + window * 2:start + window * 3].mean(axis=0)
                    recovery = (seg3 > seg2).all()
                    if recovery:
                        episodes += 1

        score = min(episodes / max(scanned * 0.1, 1), 1.0) if scanned > 0 else 0.0

        return AuditResult(
            auditor_name=self.name,
            collusion_score=float(score),
            evidence={
                "episodes_detected": episodes,
                "windows_scanned": scanned,
            },
        )


class CounterfactualAuditor(BaseAuditor):
    """Auditor 3: Counterfactual Simulator.

    Replaces one agent with a Nash bot and checks if prices drop.
    """
    name = "counterfactual"

    def audit(self, price_history, market, agents=None, saved_states=None):
        """Audit using counterfactual simulation.

        If agents and saved_states are provided, runs a counterfactual simulation.
        Otherwise, uses a price-based heuristic.
        """
        n = len(price_history)
        tail_start = int(n * 0.9)
        tail = price_history[tail_start:]

        prices_original = np.array([market.price_grid[tail[:, i]]
                                    for i in range(market.n_sellers)]).T
        avg_original = prices_original.mean(axis=0)

        if agents is not None and saved_states is not None:
            # Full counterfactual: replay with Nash bot replacing agent 0
            from src.agents import CompetitiveAgent
            cf_agents = list(agents)
            for a, s in zip(cf_agents, saved_states):
                if s is not None:
                    a.load_state(s)
                a.set_learning(False)
            cf_agents[0] = CompetitiveAgent(agent_id=0, market=market)

            cf_prices_list = []
            cf_history = price_history[:tail_start].copy()
            for _ in range(len(tail)):
                actions = [a.choose_action(cf_history) for a in cf_agents]
                cf_history = np.vstack([cf_history, actions])
                cf_prices_list.append([market.price_grid[a] for a in actions])

            avg_counterfactual = np.array(cf_prices_list).mean(axis=0)
        else:
            # Heuristic: compare observed prices to Nash
            nash = market.nash_price()
            avg_counterfactual = np.full(market.n_sellers, nash)

        price_drop = (avg_original - avg_counterfactual).mean()
        nash = market.nash_price()
        monopoly = market.monopoly_price()
        max_drop = monopoly - nash

        score = float(np.clip(price_drop / max_drop, 0.0, 1.0)) if max_drop > 0 else 0.0

        return AuditResult(
            auditor_name=self.name,
            collusion_score=score,
            evidence={
                "avg_original_price": float(avg_original.mean()),
                "avg_counterfactual_price": float(avg_counterfactual.mean()),
                "price_drop": float(price_drop),
            },
        )


class WelfareAuditor(BaseAuditor):
    """Auditor 4: Welfare Analyst.

    Scores consumer welfare loss relative to Nash -> Monopoly range.
    Reports consumer surplus, producer surplus, and total welfare.
    """
    name = "welfare"

    def audit(self, price_history, market):
        n = len(price_history)
        tail = price_history[int(n * 0.8):]

        # Compute average consumer surplus, producer surplus, total welfare in tail
        cs_observed = 0.0
        ps_observed = 0.0
        for row in tail:
            prices = market.price_grid[row]
            demand = market.compute_demand(prices)
            cs_observed += (-demand * prices).sum()
            ps_observed += ((prices - market.costs) * demand).sum()
        cs_observed /= len(tail)
        ps_observed /= len(tail)
        tw_observed = cs_observed + ps_observed

        # Welfare at Nash
        nash = market.nash_price()
        nash_prices = np.full(market.n_sellers, nash)
        nash_demand = market.compute_demand(nash_prices)
        cs_nash = (-nash_demand * nash_prices).sum()
        ps_nash = ((nash_prices - market.costs) * nash_demand).sum()
        tw_nash = cs_nash + ps_nash

        # Welfare at monopoly
        monopoly = market.monopoly_price()
        mon_prices = np.full(market.n_sellers, monopoly)
        mon_demand = market.compute_demand(mon_prices)
        cs_monopoly = (-mon_demand * mon_prices).sum()
        ps_monopoly = ((mon_prices - market.costs) * mon_demand).sum()
        tw_monopoly = cs_monopoly + ps_monopoly

        if cs_nash <= cs_monopoly:
            score = 0.0
        else:
            # How much consumer welfare is lost relative to Nash baseline?
            loss = cs_nash - cs_observed
            max_loss = cs_nash - cs_monopoly
            score = float(np.clip(loss / max_loss, 0.0, 1.0))

        return AuditResult(
            auditor_name=self.name,
            collusion_score=score,
            evidence={
                "cs_observed": float(cs_observed),
                "cs_nash": float(cs_nash),
                "cs_monopoly": float(cs_monopoly),
                "ps_observed": float(ps_observed),
                "ps_nash": float(ps_nash),
                "ps_monopoly": float(ps_monopoly),
                "tw_observed": float(tw_observed),
                "tw_nash": float(tw_nash),
                "tw_monopoly": float(tw_monopoly),
            },
        )


class AuditorPanel:
    """Runs all auditors and aggregates their verdicts."""

    def __init__(self):
        self.auditors = [
            MarginAuditor(),
            DeviationPunishmentAuditor(),
            CounterfactualAuditor(),
            WelfareAuditor(),
        ]

    def audit_all(self, price_history, market, agents=None,
                  saved_states=None):
        results = []
        for auditor in self.auditors:
            if isinstance(auditor, CounterfactualAuditor):
                result = auditor.audit(price_history, market,
                                       agents=agents,
                                       saved_states=saved_states)
            else:
                result = auditor.audit(price_history, market)
            results.append(result)
        return results

    def aggregate(self, results, method="majority"):
        scores = [r.collusion_score for r in results]
        if method == "majority":
            votes = sum(1 for s in scores if s > 0.5)
            return votes >= 3
        elif method == "unanimous":
            return all(s > 0.5 for s in scores)
        elif method == "weighted":
            # Equal weights by default; can be calibrated later
            return sum(scores) / len(scores) > 0.5
        else:
            raise ValueError(f"Unknown method: {method}")
