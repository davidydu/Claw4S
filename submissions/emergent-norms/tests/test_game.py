"""Tests for game payoff structures."""

import numpy as np

from src.game import (
    NUM_ACTIONS,
    make_asymmetric_game,
    make_dominant_game,
    make_symmetric_game,
)


def test_symmetric_coordination_payoff():
    """Matching actions yield 3.0, mismatches yield 0.0."""
    game = make_symmetric_game()
    for a in range(NUM_ACTIONS):
        pi, pj = game.payoff(a, a)
        assert pi == 3.0
        assert pj == 3.0


def test_symmetric_mismatch_payoff():
    """Mismatched actions yield 0.0 in symmetric game."""
    game = make_symmetric_game()
    pi, pj = game.payoff(0, 1)
    assert pi == 0.0
    assert pj == 0.0


def test_asymmetric_optimal_welfare():
    """Optimal welfare in asymmetric game is 4.0 (action 0)."""
    game = make_asymmetric_game()
    assert game.optimal_welfare() == 4.0


def test_dominant_game_welfare_dominant():
    """Action 0 in dominant game yields highest coordination payoff (5.0)."""
    game = make_dominant_game()
    assert game.optimal_welfare() == 5.0
    pi, pj = game.payoff(0, 0)
    assert pi == 5.0


def test_dominant_game_off_diagonal():
    """Off-diagonal payoffs in dominant game are 0.5."""
    game = make_dominant_game()
    pi, pj = game.payoff(0, 1)
    assert pi == 0.5
    assert pj == 0.5


def test_payoff_matrix_shape():
    """All game matrices are (3, 3)."""
    for make_fn in [make_symmetric_game, make_asymmetric_game, make_dominant_game]:
        game = make_fn()
        assert game.payoff_matrix.shape == (3, 3)


def test_symmetric_game_is_symmetric():
    """Symmetric game matrix equals its transpose."""
    game = make_symmetric_game()
    np.testing.assert_array_equal(game.payoff_matrix, game.payoff_matrix.T)


def test_dominant_game_is_symmetric():
    """Dominant game matrix equals its transpose."""
    game = make_dominant_game()
    np.testing.assert_array_equal(game.payoff_matrix, game.payoff_matrix.T)
