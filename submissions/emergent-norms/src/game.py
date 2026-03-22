"""Coordination game definitions with configurable payoff structures.

Each game is a 3-action coordination game where agents receive high payoff
for choosing the same action and low payoff otherwise. The three game
structures differ in whether coordination equilibria are symmetric or
welfare-ranked.

References:
    Young, H.P. (1993). "The Evolution of Conventions." Econometrica.
"""

from dataclasses import dataclass

import numpy as np


NUM_ACTIONS = 3


@dataclass(frozen=True)
class GameConfig:
    """Payoff matrix for a symmetric 3-action coordination game.

    payoff_matrix[i][j] = payoff to the row player when row plays i, col plays j.
    For symmetric games, payoff_matrix[i][j] == payoff_matrix[j][i].
    """

    name: str
    payoff_matrix: np.ndarray  # shape (3, 3)

    def payoff(self, action_i: int, action_j: int) -> tuple[float, float]:
        """Return (payoff_i, payoff_j) for a pairwise interaction."""
        return (
            float(self.payoff_matrix[action_i, action_j]),
            float(self.payoff_matrix[action_j, action_i]),
        )

    def optimal_welfare(self) -> float:
        """Maximum per-agent payoff achievable under perfect coordination."""
        # Best is when both play the same action => diagonal entries
        diag = np.diag(self.payoff_matrix)
        return float(np.max(diag))

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameConfig):
            return NotImplemented
        return self.name == other.name


def make_symmetric_game() -> GameConfig:
    """All coordination equilibria yield the same payoff.

    Payoff: 3 if both match, 0 otherwise.
    """
    matrix = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    return GameConfig(name="symmetric", payoff_matrix=matrix)


def make_asymmetric_game() -> GameConfig:
    """Coordination equilibria have different payoffs, no single dominant one.

    Action 0 match: 4, Action 1 match: 3, Action 2 match: 2.
    Off-diagonal: 0.
    """
    matrix = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    return GameConfig(name="asymmetric", payoff_matrix=matrix)


def make_dominant_game() -> GameConfig:
    """One equilibrium is welfare-dominant but others still coordinate.

    Action 0 match: 5 (welfare-dominant), Action 1 match: 2, Action 2 match: 2.
    Off-diagonal: small positive payoff (0.5) to create mild anti-coordination
    incentive that makes convergence harder.
    """
    matrix = np.array([
        [5.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 2.0],
    ])
    return GameConfig(name="dominant", payoff_matrix=matrix)


ALL_GAMES = {
    "symmetric": make_symmetric_game,
    "asymmetric": make_asymmetric_game,
    "dominant": make_dominant_game,
}
