"""Validation helpers for experiment outputs."""

from itertools import product
from typing import Any


EXPECTED_HIDDEN_DIMS = [32, 64, 128]
EXPECTED_WEIGHT_DECAYS = [0.0, 0.001, 0.01, 0.1, 1.0]
EXPECTED_SEEDS = [42, 123, 7]


def _format_run_key(key: tuple[int, float, int]) -> str:
    hidden_dim, weight_decay, seed = key
    return f"(hidden_dim={hidden_dim}, weight_decay={weight_decay}, seed={seed})"


def _format_aggregate_key(key: tuple[int, float]) -> str:
    hidden_dim, weight_decay = key
    return f"(hidden_dim={hidden_dim}, weight_decay={weight_decay})"


def collect_validation_errors(data: dict[str, Any]) -> list[str]:
    """Collect all deterministic validation errors for a results payload."""
    errors = []

    expected_run_keys = {
        (hidden_dim, weight_decay, seed)
        for hidden_dim, weight_decay, seed in product(
            EXPECTED_HIDDEN_DIMS, EXPECTED_WEIGHT_DECAYS, EXPECTED_SEEDS
        )
    }
    expected_aggregate_keys = {
        (hidden_dim, weight_decay)
        for hidden_dim, weight_decay in product(
            EXPECTED_HIDDEN_DIMS, EXPECTED_WEIGHT_DECAYS
        )
    }

    meta = data.get("metadata", {})
    expected_n_configs = len(expected_run_keys)
    if meta.get("n_configs") != expected_n_configs:
        errors.append(
            f"Expected metadata n_configs={expected_n_configs}, got {meta.get('n_configs')}"
        )
    if meta.get("hidden_dims") != EXPECTED_HIDDEN_DIMS:
        errors.append(
            f"Expected hidden_dims={EXPECTED_HIDDEN_DIMS}, got {meta.get('hidden_dims')}"
        )
    if meta.get("weight_decays") != EXPECTED_WEIGHT_DECAYS:
        errors.append(
            f"Expected weight_decays={EXPECTED_WEIGHT_DECAYS}, got {meta.get('weight_decays')}"
        )
    if meta.get("seeds") != EXPECTED_SEEDS:
        errors.append(f"Expected seeds={EXPECTED_SEEDS}, got {meta.get('seeds')}")

    runs = data.get("individual_runs", [])
    if len(runs) != expected_n_configs:
        errors.append(f"Expected {expected_n_configs} individual runs, got {len(runs)}")

    seen_run_keys = set()
    duplicate_run_keys = set()
    for run in runs:
        run_key = (run.get("hidden_dim"), run.get("weight_decay"), run.get("seed"))
        if run_key in seen_run_keys:
            duplicate_run_keys.add(run_key)
        seen_run_keys.add(run_key)

        for metric in ["train_acc", "test_acc_with_shortcut", "test_acc_without_shortcut"]:
            value = run.get(metric, -1.0)
            if not (0.0 <= value <= 1.0):
                errors.append(
                    "Run "
                    f"hd={run.get('hidden_dim')}, wd={run.get('weight_decay')}, "
                    f"seed={run.get('seed')}: {metric}={value} out of [0,1]"
                )

    if duplicate_run_keys:
        duplicates = ", ".join(_format_run_key(k) for k in sorted(duplicate_run_keys))
        errors.append(f"Duplicate individual run configurations: {duplicates}")

    missing_run_keys = expected_run_keys - seen_run_keys
    if missing_run_keys:
        missing = ", ".join(_format_run_key(k) for k in sorted(missing_run_keys))
        errors.append(f"Missing individual run configurations: {missing}")

    unexpected_run_keys = seen_run_keys - expected_run_keys
    if unexpected_run_keys:
        extras = ", ".join(_format_run_key(k) for k in sorted(unexpected_run_keys))
        errors.append(f"Unexpected individual run configurations: {extras}")

    no_reg_runs = [run for run in runs if run.get("weight_decay") == 0.0]
    positive_reliance = sum(
        1 for run in no_reg_runs if run.get("shortcut_reliance", 0.0) > 0.0
    )
    if no_reg_runs and positive_reliance < len(no_reg_runs) // 2:
        errors.append(
            "Expected most unregularized runs to show positive shortcut reliance, "
            f"but only {positive_reliance}/{len(no_reg_runs)} do"
        )

    if runs:
        avg_with = sum(run["test_acc_with_shortcut"] for run in runs) / len(runs)
        avg_without = sum(run["test_acc_without_shortcut"] for run in runs) / len(runs)
        if avg_with < avg_without:
            errors.append("Average test accuracy with shortcut should be >= without shortcut")

    aggregates = data.get("aggregates", [])
    expected_n_aggregates = len(expected_aggregate_keys)
    if len(aggregates) != expected_n_aggregates:
        errors.append(f"Expected {expected_n_aggregates} aggregate entries, got {len(aggregates)}")

    seen_aggregate_keys = set()
    duplicate_aggregate_keys = set()
    expected_n_seeds = len(EXPECTED_SEEDS)

    for aggregate in aggregates:
        aggregate_key = (aggregate.get("hidden_dim"), aggregate.get("weight_decay"))
        if aggregate_key in seen_aggregate_keys:
            duplicate_aggregate_keys.add(aggregate_key)
        seen_aggregate_keys.add(aggregate_key)

        if aggregate.get("n_seeds") != expected_n_seeds:
            errors.append(
                "Aggregate "
                f"hd={aggregate.get('hidden_dim')}, wd={aggregate.get('weight_decay')}: "
                f"expected n_seeds={expected_n_seeds}, got {aggregate.get('n_seeds')}"
            )

    if duplicate_aggregate_keys:
        duplicates = ", ".join(
            _format_aggregate_key(k) for k in sorted(duplicate_aggregate_keys)
        )
        errors.append(f"Duplicate aggregate configurations: {duplicates}")

    missing_aggregate_keys = expected_aggregate_keys - seen_aggregate_keys
    if missing_aggregate_keys:
        missing = ", ".join(
            _format_aggregate_key(k) for k in sorted(missing_aggregate_keys)
        )
        errors.append(f"Missing aggregate configurations: {missing}")

    unexpected_aggregate_keys = seen_aggregate_keys - expected_aggregate_keys
    if unexpected_aggregate_keys:
        extras = ", ".join(
            _format_aggregate_key(k) for k in sorted(unexpected_aggregate_keys)
        )
        errors.append(f"Unexpected aggregate configurations: {extras}")

    findings = data.get("findings", [])
    if len(findings) < 2:
        errors.append(f"Expected >= 2 findings, got {len(findings)}")

    return errors
