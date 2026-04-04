"""Tests for incentive schemes."""

import pytest
from src.incentives import (
    FixedPay,
    PieceRate,
    Tournament,
    ReputationBased,
    create_scheme,
)


class TestFixedPay:
    def test_constant_wage(self):
        scheme = FixedPay(wage=3.0)
        wages = scheme.compute_wages([1.0, 5.0, 3.0], ["a", "b", "c"], 0)
        assert wages == [3.0, 3.0, 3.0]

    def test_ignores_quality(self):
        scheme = FixedPay(wage=2.0)
        w1 = scheme.compute_wages([0.0], ["a"], 0)
        w2 = scheme.compute_wages([10.0], ["a"], 0)
        assert w1 == w2


class TestPieceRate:
    def test_proportional_to_quality(self):
        scheme = PieceRate(rate=0.5, base=1.0)
        wages = scheme.compute_wages([2.0, 4.0], ["a", "b"], 0)
        assert wages[0] == pytest.approx(2.0)  # 1.0 + 0.5*2
        assert wages[1] == pytest.approx(3.0)  # 1.0 + 0.5*4

    def test_negative_quality_clamped(self):
        scheme = PieceRate(rate=0.5, base=1.0)
        wages = scheme.compute_wages([-1.0], ["a"], 0)
        assert wages[0] == pytest.approx(1.0)  # base only


class TestTournament:
    def test_winner_gets_bonus(self):
        scheme = Tournament(bonus=4.0, base=1.0)
        wages = scheme.compute_wages([1.0, 5.0, 3.0], ["a", "b", "c"], 0)
        assert wages[0] == pytest.approx(1.0)
        assert wages[1] == pytest.approx(5.0)  # 1 + 4 bonus
        assert wages[2] == pytest.approx(1.0)

    def test_tie_splits_bonus(self):
        scheme = Tournament(bonus=4.0, base=1.0)
        wages = scheme.compute_wages([5.0, 5.0, 1.0], ["a", "b", "c"], 0)
        assert wages[0] == pytest.approx(3.0)  # 1 + 4/2
        assert wages[1] == pytest.approx(3.0)
        assert wages[2] == pytest.approx(1.0)

    def test_empty_list(self):
        scheme = Tournament()
        assert scheme.compute_wages([], [], 0) == []


class TestReputationBased:
    def test_initial_reputation(self):
        scheme = ReputationBased(alpha=0.1, rep_bonus=4.0, base=1.0)
        wages = scheme.compute_wages([3.0], ["a"], 0)
        # Initial rep = 0.5, then updated with q_norm = (3-1)/4 = 0.5
        # new_rep = 0.1*0.5 + 0.9*0.5 = 0.5
        assert wages[0] == pytest.approx(1.0 + 4.0 * 0.5)

    def test_reputation_updates(self):
        scheme = ReputationBased(alpha=0.1, rep_bonus=4.0, base=1.0)
        # First round: high quality
        scheme.compute_wages([5.0], ["a"], 0)
        rep1 = scheme.get_reputations()["a"]
        # q_norm = (5-1)/4 = 1.0, new_rep = 0.1*1.0 + 0.9*0.5 = 0.55
        assert rep1 == pytest.approx(0.55)

        # Second round: low quality
        scheme.compute_wages([1.0], ["a"], 1)
        rep2 = scheme.get_reputations()["a"]
        # q_norm = 0.0, new_rep = 0.1*0.0 + 0.9*0.55 = 0.495
        assert rep2 == pytest.approx(0.495)

    def test_reset(self):
        scheme = ReputationBased()
        scheme.compute_wages([3.0], ["a"], 0)
        assert len(scheme.get_reputations()) == 1
        scheme.reset()
        assert scheme.get_reputations() == {}


class TestCreateScheme:
    def test_creates_all_types(self):
        for name in ["fixed_pay", "piece_rate", "tournament", "reputation"]:
            scheme = create_scheme(name)
            assert scheme.name == name

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown scheme"):
            create_scheme("unknown")
