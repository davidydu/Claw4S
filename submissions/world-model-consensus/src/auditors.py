"""
Four auditors that evaluate simulation results, plus an AuditorPanel.

1. CoordinationAuditor  — coordination rate in final 20% of rounds
2. ConsensusSpeedAuditor — rounds until first sustained consensus
3. WelfareAuditor       — average payoff in final 20%
4. FairnessAuditor      — whether consensus favours majority or minority priors
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from src.experiment import SimulationResult


# ======================================================================
# Individual auditors
# ======================================================================

@dataclass
class AuditResult:
    """Result from a single auditor."""
    name: str
    value: float
    detail: str = ""


class CoordinationAuditor:
    """Fraction of final-20% rounds where all agents chose the same action."""

    name = "coordination_rate"

    def audit(self, result: SimulationResult) -> AuditResult:
        n = len(result.coordinated)
        tail_start = int(n * 0.8)
        tail = result.coordinated[tail_start:]
        rate = float(tail.mean()) if len(tail) > 0 else 0.0
        return AuditResult(
            name=self.name,
            value=rate,
            detail=f"Coordination in rounds {tail_start}..{n}: {rate:.4f}",
        )


class ConsensusSpeedAuditor:
    """Number of rounds until first *sustained* consensus.

    Sustained = 100 consecutive rounds where all agents agree.
    Returns n_rounds if never achieved.
    """

    name = "consensus_time"
    SUSTAIN_WINDOW = 100

    def audit(self, result: SimulationResult) -> AuditResult:
        n = len(result.coordinated)
        streak = 0
        for t in range(n):
            if result.coordinated[t]:
                streak += 1
                if streak >= self.SUSTAIN_WINDOW:
                    first_t = t - self.SUSTAIN_WINDOW + 1
                    return AuditResult(
                        name=self.name,
                        value=float(first_t),
                        detail=f"Sustained consensus first at round {first_t}",
                    )
            else:
                streak = 0

        return AuditResult(
            name=self.name,
            value=float(n),  # never achieved
            detail=f"No sustained consensus within {n} rounds",
        )


class WelfareAuditor:
    """Average payoff per agent in the final 20% of rounds.

    In this all-or-nothing game, welfare = coordination_rate.
    We compute it from payoffs directly for generality.
    """

    name = "welfare"

    def audit(self, result: SimulationResult) -> AuditResult:
        n = result.payoff_history.shape[0]
        tail_start = int(n * 0.8)
        tail = result.payoff_history[tail_start:]
        welfare = float(tail.mean()) if tail.size > 0 else 0.0
        return AuditResult(
            name=self.name,
            value=welfare,
            detail=f"Mean payoff (final 20%): {welfare:.4f}",
        )


class FairnessAuditor:
    """Minority-suppression index.

    When consensus forms, how often does the group converge to the action
    that was *initially preferred by the majority* vs. the minority?

    Index:
    - 1.0 = consensus always on majority-preferred action
    - 0.0 = consensus always on a minority-preferred action
    - NaN / -1 = no consensus formed
    """

    name = "fairness"

    def audit(self, result: SimulationResult) -> AuditResult:
        prefs = result.preferred_actions
        n_agents = len(prefs)
        n = result.action_history.shape[0]
        tail_start = int(n * 0.8)

        # Find the majority-preferred action(s)
        from collections import Counter
        pref_counts = Counter(prefs)
        majority_action = pref_counts.most_common(1)[0][0]

        # Look at coordinated rounds in the tail
        tail_actions = result.action_history[tail_start:]
        tail_coord = result.coordinated[tail_start:]

        coordinated_rounds = tail_actions[tail_coord]
        if len(coordinated_rounds) == 0:
            return AuditResult(
                name=self.name,
                value=-1.0,
                detail="No coordination in tail — fairness undefined",
            )

        # In coordinated rounds all agents play the same action; take col 0
        consensus_actions = coordinated_rounds[:, 0]
        majority_frac = float((consensus_actions == majority_action).mean())

        return AuditResult(
            name=self.name,
            value=majority_frac,
            detail=(f"Majority-preferred action ({majority_action}) chosen "
                    f"in {majority_frac:.2%} of coordinated tail rounds"),
        )


# ======================================================================
# Auditor panel
# ======================================================================

ALL_AUDITORS = [
    CoordinationAuditor(),
    ConsensusSpeedAuditor(),
    WelfareAuditor(),
    FairnessAuditor(),
]


def run_audit_panel(result: SimulationResult) -> Dict[str, AuditResult]:
    """Run all auditors on a simulation result."""
    return {a.name: a.audit(result) for a in ALL_AUDITORS}
