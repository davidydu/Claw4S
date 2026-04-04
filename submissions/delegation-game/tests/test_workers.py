"""Tests for worker agent types."""

import numpy as np
import pytest
from src.workers import (
    HonestWorker,
    ShirkerWorker,
    StrategicWorker,
    AdaptiveWorker,
    create_worker,
)


class TestHonestWorker:
    def test_always_max_effort(self):
        w = HonestWorker("alice")
        for t in range(100):
            assert w.choose_effort(t, []) == 5

    def test_type_and_name(self):
        w = HonestWorker("alice")
        assert w.worker_type == "honest"
        assert w.name == "alice"


class TestShirkerWorker:
    def test_always_min_effort(self):
        w = ShirkerWorker("bob")
        for t in range(100):
            assert w.choose_effort(t, []) == 1

    def test_type_and_name(self):
        w = ShirkerWorker("bob")
        assert w.worker_type == "shirker"
        assert w.name == "bob"


class TestStrategicWorker:
    def test_starts_at_midpoint(self):
        w = StrategicWorker("carol")
        assert w.choose_effort(0, []) == 3

    def test_increases_effort_on_high_pay(self):
        w = StrategicWorker("carol")
        history = [{"worker": "carol", "effort": 3, "wage": 6.0, "round": 0}]
        e = w.choose_effort(1, history)
        assert e == 4  # pay/effort = 2.0 > 1.2, so increase

    def test_decreases_effort_on_low_pay(self):
        w = StrategicWorker("carol")
        history = [{"worker": "carol", "effort": 3, "wage": 1.5, "round": 0}]
        e = w.choose_effort(1, history)
        assert e == 2  # pay/effort = 0.5 < 0.8, so decrease

    def test_effort_clamped_1_5(self):
        w = StrategicWorker("carol")
        # Drive effort down repeatedly
        for i in range(10):
            history = [{"worker": "carol", "effort": 1, "wage": 0.1, "round": i}]
            e = w.choose_effort(i + 1, history)
            assert 1 <= e <= 5

    def test_reset(self):
        w = StrategicWorker("carol")
        w._effort = 5
        w.reset()
        assert w._effort == 3


class TestAdaptiveWorker:
    def test_effort_in_range(self):
        rng = np.random.default_rng(42)
        w = AdaptiveWorker("dave", rng=rng)
        for t in range(50):
            e = w.choose_effort(t, [])
            assert 1 <= e <= 5

    def test_learns_from_history(self):
        rng = np.random.default_rng(42)
        w = AdaptiveWorker("dave", rng=rng)
        # Simulate sequential rounds where effort=5 always pays well
        # and other efforts pay poorly, so the worker learns effort=5 is best
        history = []
        # Seed with diverse efforts so ema_returns has >= 3 entries
        for e in [1, 2, 3, 4, 5]:
            wage = 10.0 if e == 5 else 0.5
            history.append({
                "worker": "dave", "effort": e,
                "wage": wage, "round": len(history),
            })
            w.choose_effort(len(history), history)
        # Now run many rounds where effort=5 consistently pays well
        for i in range(200):
            history.append({
                "worker": "dave", "effort": 5, "wage": 10.0,
                "round": len(history),
            })
            w.choose_effort(len(history), history)
        # After learning, EMA for effort=5 should be highest
        assert 5 in w._ema_returns
        assert w._ema_returns[5] > w._ema_returns.get(1, 0)

    def test_reset_clears_state(self):
        w = AdaptiveWorker("dave")
        w._ema_returns = {1: 5.0, 5: 10.0}
        w.reset()
        assert w._ema_returns == {}


class TestCreateWorker:
    def test_creates_all_types(self):
        for wtype in ["honest", "shirker", "strategic", "adaptive"]:
            w = create_worker(wtype, f"test_{wtype}")
            assert w.worker_type == wtype
            assert w.name == f"test_{wtype}"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown worker type"):
            create_worker("unknown", "test")
