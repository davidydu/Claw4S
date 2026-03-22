"""Tests for simulation engine."""

import random
from src.simulation import generate_signals, detect_cascade, run_single_simulation
from src.agents import ACTION_A, ACTION_B


def test_generate_signals_correct_length():
    """generate_signals returns the right number of signals."""
    rng = random.Random(42)
    signals = generate_signals(ACTION_A, 20, 0.7, rng)
    assert len(signals) == 20


def test_generate_signals_all_correct_at_q1():
    """With q=1.0, all signals match true state."""
    rng = random.Random(42)
    signals = generate_signals(ACTION_A, 50, 1.0, rng)
    assert all(s == ACTION_A for s in signals)


def test_generate_signals_binary():
    """All signals are 0 or 1."""
    rng = random.Random(42)
    signals = generate_signals(ACTION_B, 100, 0.7, rng)
    assert all(s in (ACTION_A, ACTION_B) for s in signals)


def test_generate_signals_symmetry():
    """Signal distribution is approximately symmetric across true states.

    With q=0.7 and 1000 signals, about 70% should match true state.
    """
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    signals_a = generate_signals(ACTION_A, 1000, 0.7, rng_a)
    signals_b = generate_signals(ACTION_B, 1000, 0.7, rng_b)
    frac_correct_a = sum(1 for s in signals_a if s == ACTION_A) / 1000
    frac_correct_b = sum(1 for s in signals_b if s == ACTION_B) / 1000
    assert abs(frac_correct_a - frac_correct_b) < 0.01


def test_detect_cascade_no_cascade():
    """No cascade when all agents follow their signal."""
    actions = [0, 1, 0, 1, 0]
    signals = [0, 1, 0, 1, 0]
    result = detect_cascade(actions, signals)
    assert result["cascade_formed"] is False


def test_detect_cascade_too_short():
    """Sequences shorter than 3 cannot form a cascade."""
    result = detect_cascade([0, 0], [0, 1])
    assert result["cascade_formed"] is False


def test_detect_cascade_basic():
    """Detect a basic cascade: agents 0,1 choose A, agent 2 has signal B but chooses A."""
    actions = [0, 0, 0, 0, 0]
    signals = [0, 0, 1, 1, 0]
    # Agent 2: predecessors [A, A], signal B, chose A => cascade on A
    result = detect_cascade(actions, signals)
    assert result["cascade_formed"] is True
    assert result["cascade_start"] == 2
    assert result["cascade_action"] == ACTION_A


def test_detect_cascade_length():
    """Cascade length counts consecutive agents matching cascade action."""
    actions = [0, 0, 0, 0, 1, 0]
    signals = [0, 0, 1, 1, 1, 0]
    result = detect_cascade(actions, signals)
    assert result["cascade_formed"] is True
    assert result["cascade_length"] == 2  # agents 2,3 follow A, then agent 4 breaks


def test_run_single_simulation_structure():
    """run_single_simulation returns all expected keys."""
    result = run_single_simulation("bayesian", 10, 0.7, ACTION_A, 42)
    expected_keys = {
        "agent_type", "n_agents", "signal_quality", "true_state", "seed",
        "actions", "signals", "cascade_formed", "cascade_start",
        "cascade_action", "cascade_length", "agents_in_cascade",
        "cascade_correct", "majority_correct",
    }
    assert set(result.keys()) == expected_keys


def test_run_single_simulation_deterministic():
    """Same parameters and seed produce identical results."""
    r1 = run_single_simulation("bayesian", 20, 0.7, ACTION_A, 42)
    r2 = run_single_simulation("bayesian", 20, 0.7, ACTION_A, 42)
    assert r1["actions"] == r2["actions"]
    assert r1["signals"] == r2["signals"]


def test_run_single_simulation_different_seeds():
    """Different seeds produce different results (with high probability)."""
    r1 = run_single_simulation("bayesian", 20, 0.7, ACTION_A, 42)
    r2 = run_single_simulation("bayesian", 20, 0.7, ACTION_A, 123)
    # Very unlikely to be identical with different seeds and 20 agents
    assert r1["actions"] != r2["actions"] or r1["signals"] != r2["signals"]
