"""Validate experiment results for completeness and correctness.

Usage: .venv/bin/python validate.py
"""

import json
import sys


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

    # Check simulation count: 4 compositions x 3 games x 3 sizes x 3 seeds = 108
    expected = 108
    if n_sims != expected:
        errors.append(f"Expected {expected} simulations, got {n_sims}")

    if len(results) != n_sims:
        errors.append(f"Metadata says {n_sims} sims but results has {len(results)} entries")

    # Check all games are present
    games_found = set(r["game"] for r in results)
    expected_games = {"symmetric", "asymmetric", "dominant"}
    if games_found != expected_games:
        errors.append(f"Expected games {expected_games}, got {games_found}")

    # Check all compositions are present
    comps_found = set(r["composition_name"] for r in results)
    expected_comps = {"all_adaptive", "mixed_conform", "innovator_heavy", "traditionalist_heavy"}
    if comps_found != expected_comps:
        errors.append(f"Expected compositions {expected_comps}, got {comps_found}")

    # Check all sizes are present
    sizes_found = set(r["population_size"] for r in results)
    expected_sizes = {20, 50, 100}
    if sizes_found != expected_sizes:
        errors.append(f"Expected sizes {expected_sizes}, got {sizes_found}")

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
