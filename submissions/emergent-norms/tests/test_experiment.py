"""Tests for the experiment grid and runner."""

from src.experiment import COMPOSITIONS, POPULATION_SIZES, SEEDS, build_experiment_grid
from src.game import ALL_GAMES


def test_grid_size():
    """Grid should have 4 compositions * 3 games * 3 sizes * 3 seeds = 108."""
    grid = build_experiment_grid()
    assert len(grid) == 108


def test_grid_covers_all_compositions():
    """Every composition appears in the grid."""
    grid = build_experiment_grid()
    comp_names = set(g[0] for g in grid)
    assert comp_names == set(COMPOSITIONS.keys())


def test_grid_covers_all_games():
    """Every game appears in the grid."""
    grid = build_experiment_grid()
    game_names = set(g[1] for g in grid)
    assert game_names == set(ALL_GAMES.keys())


def test_grid_covers_all_sizes():
    """Every population size appears in the grid."""
    grid = build_experiment_grid()
    sizes = set(g[2] for g in grid)
    assert sizes == set(POPULATION_SIZES)


def test_grid_covers_all_seeds():
    """Every seed appears in the grid."""
    grid = build_experiment_grid()
    seeds = set(g[3] for g in grid)
    assert seeds == set(SEEDS)
