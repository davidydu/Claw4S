"""Tests for src/auditors.py — all four auditors."""

import numpy as np
import pytest

from src.experiment import SimulationConfig, SimulationResult
from src.auditors import (
    CoordinationAuditor, ConsensusSpeedAuditor, WelfareAuditor,
    FairnessAuditor, run_audit_panel,
)


def _make_result(coordinated_pattern: np.ndarray,
                 action_col0: np.ndarray | None = None,
                 n_agents: int = 4,
                 preferred: list | None = None) -> SimulationResult:
    """Helper: build a SimulationResult from a coordinated boolean array."""
    n_rounds = len(coordinated_pattern)
    action_history = np.zeros((n_rounds, n_agents), dtype=np.int32)
    payoff_history = np.zeros((n_rounds, n_agents), dtype=np.float64)

    for t in range(n_rounds):
        if coordinated_pattern[t]:
            action = 0 if action_col0 is None else int(action_col0[t])
            action_history[t, :] = action
            payoff_history[t, :] = 1.0

    return SimulationResult(
        config=SimulationConfig(n_agents=n_agents, n_rounds=n_rounds),
        action_history=action_history,
        payoff_history=payoff_history,
        coordinated=coordinated_pattern,
        preferred_actions=preferred or [0, 1, 2, 3][:n_agents],
    )


class TestCoordinationAuditor:
    def test_perfect_coordination(self):
        result = _make_result(np.ones(1000, dtype=bool))
        ar = CoordinationAuditor().audit(result)
        assert ar.value == pytest.approx(1.0)

    def test_no_coordination(self):
        result = _make_result(np.zeros(1000, dtype=bool))
        ar = CoordinationAuditor().audit(result)
        assert ar.value == pytest.approx(0.0)

    def test_only_measures_final_20pct(self):
        # First 80% coordinated, last 20% not
        pattern = np.ones(1000, dtype=bool)
        pattern[800:] = False
        result = _make_result(pattern)
        ar = CoordinationAuditor().audit(result)
        assert ar.value == pytest.approx(0.0)


class TestConsensusSpeedAuditor:
    def test_immediate_consensus(self):
        result = _make_result(np.ones(1000, dtype=bool))
        ar = ConsensusSpeedAuditor().audit(result)
        assert ar.value == 0.0  # sustained from round 0

    def test_no_consensus(self):
        result = _make_result(np.zeros(1000, dtype=bool))
        ar = ConsensusSpeedAuditor().audit(result)
        assert ar.value == 1000.0  # never achieved

    def test_delayed_consensus(self):
        pattern = np.zeros(1000, dtype=bool)
        pattern[500:] = True  # sustained from 500
        result = _make_result(pattern)
        ar = ConsensusSpeedAuditor().audit(result)
        assert ar.value == 500.0


class TestWelfareAuditor:
    def test_perfect_welfare(self):
        result = _make_result(np.ones(1000, dtype=bool))
        ar = WelfareAuditor().audit(result)
        assert ar.value == pytest.approx(1.0)

    def test_zero_welfare(self):
        result = _make_result(np.zeros(1000, dtype=bool))
        ar = WelfareAuditor().audit(result)
        assert ar.value == pytest.approx(0.0)


class TestFairnessAuditor:
    def test_majority_always_wins(self):
        # 3 out of 4 agents prefer action 0
        pattern = np.ones(1000, dtype=bool)
        actions = np.zeros(1000, dtype=np.int32)  # always action 0
        result = _make_result(pattern, action_col0=actions,
                              preferred=[0, 0, 0, 1])
        ar = FairnessAuditor().audit(result)
        assert ar.value == pytest.approx(1.0)

    def test_minority_always_wins(self):
        # majority prefers 0, but consensus forms on action 1
        pattern = np.ones(1000, dtype=bool)
        actions = np.ones(1000, dtype=np.int32)  # always action 1
        result = _make_result(pattern, action_col0=actions,
                              preferred=[0, 0, 0, 1])
        ar = FairnessAuditor().audit(result)
        assert ar.value == pytest.approx(0.0)

    def test_no_coordination_returns_negative(self):
        result = _make_result(np.zeros(1000, dtype=bool))
        ar = FairnessAuditor().audit(result)
        assert ar.value == -1.0


class TestAuditPanel:
    def test_returns_all_four_metrics(self):
        result = _make_result(np.ones(1000, dtype=bool))
        panel = run_audit_panel(result)
        assert set(panel.keys()) == {"coordination_rate", "consensus_time", "welfare", "fairness"}

    def test_consistent_with_individual(self):
        result = _make_result(np.ones(1000, dtype=bool))
        panel = run_audit_panel(result)
        assert panel["coordination_rate"].value == CoordinationAuditor().audit(result).value
