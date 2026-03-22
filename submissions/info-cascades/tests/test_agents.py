"""Tests for agent decision logic."""

import random
from src.agents import (
    ACTION_A,
    ACTION_B,
    BayesianAgent,
    HeuristicAgent,
    ContrarianAgent,
    NoisyBayesianAgent,
    make_agent,
    _log_likelihood_ratio,
)


def test_bayesian_follows_signal_no_predecessors():
    """With no predecessors, Bayesian agent follows its private signal."""
    agent = BayesianAgent()
    assert agent.choose(ACTION_A, [], 0.7) == ACTION_A
    assert agent.choose(ACTION_B, [], 0.7) == ACTION_B


def test_bayesian_follows_strong_public_evidence():
    """With overwhelming public evidence, Bayesian ignores private signal."""
    agent = BayesianAgent()
    # 5 predecessors all chose A, signal says B, q=0.6
    predecessors = [ACTION_A] * 5
    # Public LLR = 5 * log(0.6/0.4) = 5 * 0.405 = 2.03
    # Private LLR for B = -log(0.6/0.4) = -0.405
    # Total = 2.03 - 0.405 = 1.62 > 0 => choose A
    assert agent.choose(ACTION_B, predecessors, 0.6) == ACTION_A


def test_bayesian_follows_signal_when_public_is_tied():
    """When predecessors are split, Bayesian follows private signal."""
    agent = BayesianAgent()
    predecessors = [ACTION_A, ACTION_B]  # balanced
    assert agent.choose(ACTION_A, predecessors, 0.7) == ACTION_A
    assert agent.choose(ACTION_B, predecessors, 0.7) == ACTION_B


def test_bayesian_symmetry():
    """Bayesian agent is symmetric: swapping state labels gives swapped action."""
    agent = BayesianAgent()
    # Signal A with predecessors [A, A, B]
    action_a = agent.choose(ACTION_A, [ACTION_A, ACTION_A, ACTION_B], 0.7)
    # Signal B with predecessors [B, B, A] — mirror image
    action_b = agent.choose(ACTION_B, [ACTION_B, ACTION_B, ACTION_A], 0.7)
    assert action_a == ACTION_A
    assert action_b == ACTION_B


def test_heuristic_follows_majority():
    """Heuristic follows majority when fraction exceeds threshold."""
    agent = HeuristicAgent(threshold=0.6)
    # 4 out of 5 chose A => frac_A = 0.8 > 0.6
    assert agent.choose(ACTION_B, [ACTION_A] * 4 + [ACTION_B], 0.7) == ACTION_A


def test_heuristic_follows_signal_when_split():
    """Heuristic follows signal when majority is below threshold."""
    agent = HeuristicAgent(threshold=0.6)
    # 3 out of 5 chose A => frac_A = 0.6 = threshold, not exceeded
    assert agent.choose(ACTION_B, [ACTION_A] * 3 + [ACTION_B] * 2, 0.7) == ACTION_B


def test_heuristic_no_predecessors():
    """Heuristic follows signal when no predecessors exist."""
    agent = HeuristicAgent(threshold=0.6)
    assert agent.choose(ACTION_A, [], 0.7) == ACTION_A
    assert agent.choose(ACTION_B, [], 0.7) == ACTION_B


def test_contrarian_goes_against_with_p1():
    """Contrarian with p=1.0 always goes against the majority."""
    rng = random.Random(42)
    agent = ContrarianAgent(p_contrarian=1.0, rng=rng)
    assert agent.choose(ACTION_A, [ACTION_A, ACTION_A, ACTION_A], 0.7) == ACTION_B


def test_contrarian_follows_with_p0():
    """Contrarian with p=0.0 always follows the majority."""
    rng = random.Random(42)
    agent = ContrarianAgent(p_contrarian=0.0, rng=rng)
    assert agent.choose(ACTION_B, [ACTION_A, ACTION_A, ACTION_A], 0.7) == ACTION_A


def test_noisy_bayesian_with_zero_noise():
    """Noisy Bayesian with zero noise behaves like pure Bayesian."""
    rng = random.Random(42)
    agent = NoisyBayesianAgent(noise_std=0.0, rng=rng)
    bayesian = BayesianAgent()
    predecessors = [ACTION_A, ACTION_A, ACTION_B]
    for signal in [ACTION_A, ACTION_B]:
        assert agent.choose(signal, predecessors, 0.7) == bayesian.choose(
            signal, predecessors, 0.7
        )


def test_make_agent_all_types():
    """make_agent returns correct types for all valid type strings."""
    rng = random.Random(0)
    assert isinstance(make_agent("bayesian", rng), BayesianAgent)
    assert isinstance(make_agent("heuristic", rng), HeuristicAgent)
    assert isinstance(make_agent("contrarian", rng), ContrarianAgent)
    assert isinstance(make_agent("noisy_bayesian", rng), NoisyBayesianAgent)


def test_make_agent_invalid_type():
    """make_agent raises ValueError for unknown type."""
    try:
        make_agent("invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_log_likelihood_ratio_empty():
    """LLR is 0 with no predecessors."""
    assert _log_likelihood_ratio([], 0.7) == 0.0


def test_log_likelihood_ratio_balanced():
    """LLR is 0 when predecessors are balanced."""
    assert _log_likelihood_ratio([ACTION_A, ACTION_B], 0.7) == 0.0
