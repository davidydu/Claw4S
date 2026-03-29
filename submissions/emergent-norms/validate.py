"""Validate experiment results for completeness and correctness.

Usage: .venv/bin/python validate.py
"""

import json
import sys

from src.experiment import COMPOSITIONS, POPULATION_SIZES, SEEDS
from src.game import ALL_GAMES


def main() -> None:
    with open("results/results.json") as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]
    errors: list[str] = []

    n_sims = metadata["num_simulations"]
    print(f"Simulations:      {n_sims}")
    print(f"Games:            {metadata['games']}")
    print(f"Compositions:     {metadata['compositions']}")
    print(f"Population sizes: {metadata['population_sizes']}")
    print(f"Runtime:          {metadata['elapsed_seconds']}s")

    expected_games = set(ALL_GAMES.keys())
    expected_comps = set(COMPOSITIONS.keys())
    expected_sizes = set(POPULATION_SIZES)
    expected_seeds = set(SEEDS)

    expected_grid = {
        (comp, game, size, seed)
        for comp in expected_comps
        for game in expected_games
        for size in expected_sizes
        for seed in expected_seeds
    }
    expected = len(expected_grid)

    # Check simulation count: expected full grid coverage.
    if n_sims != expected:
        errors.append(f"Expected {expected} simulations, got {n_sims}")

    if len(results) != n_sims:
        errors.append(f"Metadata says {n_sims} sims but results has {len(results)} entries")

    # Check all games are present
    games_found = set(r["game"] for r in results)
    if games_found != expected_games:
        errors.append(f"Expected games {expected_games}, got {games_found}")

    # Check all compositions are present
    comps_found = set(r["composition_name"] for r in results)
    if comps_found != expected_comps:
        errors.append(f"Expected compositions {expected_comps}, got {comps_found}")

    # Check all sizes are present
    sizes_found = set(r["population_size"] for r in results)
    if sizes_found != expected_sizes:
        errors.append(f"Expected sizes {expected_sizes}, got {sizes_found}")

    # Check all seeds are present
    seeds_found = set(r["seed"] for r in results)
    if seeds_found != expected_seeds:
        errors.append(f"Expected seeds {expected_seeds}, got {seeds_found}")

    # Check every (composition, game, size, seed) appears exactly once.
    observed_grid = [
        (r["composition_name"], r["game"], r["population_size"], r["seed"])
        for r in results
    ]
    observed_grid_set = set(observed_grid)
    duplicates = len(observed_grid) - len(observed_grid_set)
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate grid cell(s)")

    missing = expected_grid - observed_grid_set
    if missing:
        examples = sorted(missing)[:3]
        errors.append(f"Missing {len(missing)} expected grid cell(s), e.g. {examples}")

    unexpected = observed_grid_set - expected_grid
    if unexpected:
        examples = sorted(unexpected)[:3]
        errors.append(f"Found {len(unexpected)} unexpected grid cell(s), e.g. {examples}")

    # Validate metric ranges
    for r in results:
        eff = r["efficiency"]
        if not (0.0 <= eff <= 1.0):
            errors.append(
                f"Efficiency {eff} out of [0,1] for {r['composition_name']}/{r['game']}"
            )

        div = r["diversity"]
        if div not in (0, 1, 2, 3):
            errors.append(
                f"Diversity {div} not in {{0,1,2,3}} for {r['composition_name']}/{r['game']}"
            )

        conv = r["convergence_time"]
        if conv < 0 or conv > r["total_rounds"]:
            errors.append(
                f"Convergence time {conv} out of range for {r['composition_name']}/{r['game']}"
            )

        frag = r["fragility"]
        if not (0.0 <= frag <= 1.0):
            errors.append(
                f"Fragility {frag} out of [0,1] for {r['composition_name']}/{r['game']}"
            )

    # Summary stats
    efficiencies = [r["efficiency"] for r in results]
    avg_eff = sum(efficiencies) / len(efficiencies)
    converged = sum(1 for r in results if r["convergence_time"] < r["total_rounds"])
    print(f"\nAvg efficiency:   {avg_eff:.3f}")
    print(f"Converged:        {converged}/{n_sims}")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nValidation passed.")


if __name__ == "__main__":
    main()
