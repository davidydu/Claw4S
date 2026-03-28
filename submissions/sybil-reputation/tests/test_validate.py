"""Tests for result validation rules."""

from validate import validate_results


def _make_result(
    algorithm: str,
    n_sybil: int,
    strategy: str,
    seed: int,
    accuracy: float,
) -> dict:
    return {
        "config": {
            "n_honest": 20,
            "n_sybil": n_sybil,
            "algorithm": algorithm,
            "strategy": strategy,
            "n_rounds": 100,
            "seed": seed,
        },
        "metrics": {
            "reputation_accuracy": accuracy,
            "sybil_detection_rate": 0.5,
            "honest_welfare": 0.5,
            "market_efficiency": 0.5,
        },
    }


def _build_data(baseline_acc: float, simple_k20_acc: float) -> dict:
    algorithms = [
        "eigentrust",
        "pagerank_trust",
        "simple_average",
        "weighted_history",
    ]
    strategies = ["bad_mouthing", "ballot_stuffing", "whitewashing"]
    seeds = [1]
    sybil_counts = [0, 20]

    results = []
    for algo in algorithms:
        # K=0 baseline has one "none" strategy per seed.
        for seed in seeds:
            results.append(
                _make_result(
                    algorithm=algo,
                    n_sybil=0,
                    strategy="none",
                    seed=seed,
                    accuracy=baseline_acc,
                )
            )

    for algo in algorithms:
        for strat in strategies:
            for seed in seeds:
                acc = simple_k20_acc if algo == "simple_average" else baseline_acc
                results.append(
                    _make_result(
                        algorithm=algo,
                        n_sybil=20,
                        strategy=strat,
                        seed=seed,
                        accuracy=acc,
                    )
                )

    return {
        "metadata": {
            "algorithms": algorithms,
            "strategies": strategies,
            "sybil_counts": sybil_counts,
            "seeds": seeds,
            "total_simulations": len(results),
            "elapsed_seconds": 1.0,
        },
        "results": results,
    }


def test_validate_flags_low_baseline_accuracy():
    data = _build_data(baseline_acc=0.4, simple_k20_acc=0.2)
    errors = validate_results(data)
    assert any("Baseline accuracy too low" in e for e in errors)


def test_validate_flags_non_degrading_simple_average_under_attack():
    data = _build_data(baseline_acc=0.8, simple_k20_acc=0.85)
    errors = validate_results(data)
    assert any("Simple average at K=20 should be below baseline" in e for e in errors)
