"""Hyperparameter sweep across optimizers, learning rates, and weight decays.

Runs all combinations and collects results for analysis.
Uses tensor-based training (no DataLoader) for speed.
Supports resume mode with per-run checkpoints for long executions.
"""

import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import torch

from data import split_data, PRIME, SEED
from train import train_run

# Sweep configuration
OPTIMIZERS = ["sgd", "sgd_momentum", "adam", "adamw"]

LEARNING_RATES = [1e-1, 3e-2, 1e-2]

WEIGHT_DECAYS = [0.0, 0.01, 0.1]

MAX_EPOCHS = 750
LOG_INTERVAL = 75
BATCH_SIZE = 512
RESULTS_FILENAME = "sweep_results.json"


def _config_key(optimizer: str, lr: float, weight_decay: float) -> tuple[str, float, float]:
    return optimizer, float(lr), float(weight_decay)


def _all_configs(
    optimizers: list[str],
    learning_rates: list[float],
    weight_decays: list[float],
) -> list[tuple[str, float, float]]:
    return [
        _config_key(opt, lr, wd)
        for opt in optimizers
        for lr in learning_rates
        for wd in weight_decays
    ]


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_cached_runs(
    save_path: str,
    valid_configs: set[tuple[str, float, float]],
) -> tuple[dict[tuple[str, float, float], dict], float]:
    """Load cached run results from disk if present."""
    if not os.path.isfile(save_path):
        return {}, 0.0

    try:
        with open(save_path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, 0.0

    cached: dict[tuple[str, float, float], dict] = {}
    for run in payload.get("runs", []):
        optimizer = run.get("optimizer")
        lr = _safe_float(run.get("lr"))
        wd = _safe_float(run.get("weight_decay"))
        if not isinstance(optimizer, str) or lr is None or wd is None:
            continue
        key = _config_key(optimizer, lr, wd)
        if key in valid_configs and key not in cached:
            cached[key] = run

    prior_total_seconds = _safe_float(payload.get("metadata", {}).get("total_seconds"))
    if prior_total_seconds is None:
        prior_total_seconds = 0.0

    return cached, prior_total_seconds


def _build_metadata(
    optimizers: list[str],
    learning_rates: list[float],
    weight_decays: list[float],
    max_epochs: int,
    log_interval: int,
    batch_size: int,
    total_runs: int,
    completed_runs: int,
    total_seconds: float,
    train_examples: int,
    test_examples: int,
    resumed_from_cache: bool,
) -> dict:
    total_examples = train_examples + test_examples
    train_fraction = train_examples / total_examples if total_examples else 0.0
    return {
        "prime": PRIME,
        "seed": SEED,
        "max_epochs": max_epochs,
        "log_interval": log_interval,
        "batch_size": batch_size,
        "num_runs": total_runs,
        "completed_runs": completed_runs,
        "total_seconds": round(total_seconds, 1),
        "optimizers": optimizers,
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "train_fraction": round(train_fraction, 4),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resumed_from_cache": resumed_from_cache,
    }


def _write_results(save_path: str, metadata: dict, runs: list[dict]) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "runs": runs}, f, indent=2)


def run_sweep(
    optimizers: list[str] | None = None,
    learning_rates: list[float] | None = None,
    weight_decays: list[float] | None = None,
    max_epochs: int = MAX_EPOCHS,
    log_interval: int = LOG_INTERVAL,
    batch_size: int = BATCH_SIZE,
    results_dir: str = "results",
    resume: bool = True,
) -> list[dict]:
    """Run the full optimizer sweep.

    Args:
        optimizers: List of optimizer names. Defaults to OPTIMIZERS.
        learning_rates: List of learning rates. Defaults to LEARNING_RATES.
        weight_decays: List of weight decay values. Defaults to WEIGHT_DECAYS.
        max_epochs: Maximum epochs per run.
        log_interval: Epoch interval for logging metrics.
        batch_size: Mini-batch size for training.
        results_dir: Directory to save results.
        resume: Reuse cached run results from results/sweep_results.json.

    Returns:
        List of result dicts, one per configuration.
    """
    if optimizers is None:
        optimizers = OPTIMIZERS
    if learning_rates is None:
        learning_rates = LEARNING_RATES
    if weight_decays is None:
        weight_decays = WEIGHT_DECAYS

    # Create shared data split (same split for all runs)
    train_ds, test_ds = split_data(p=PRIME, seed=SEED)

    # Extract tensors for fast training
    train_a = train_ds.a
    train_b = train_ds.b
    train_t = train_ds.targets
    test_a = test_ds.a
    test_b = test_ds.b
    test_t = test_ds.targets

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, RESULTS_FILENAME)

    configs = _all_configs(optimizers, learning_rates, weight_decays)
    total_runs = len(configs)
    valid_configs = set(configs)
    cached_by_key: dict[tuple[str, float, float], dict] = {}
    prior_total_seconds = 0.0
    if resume:
        cached_by_key, prior_total_seconds = _load_cached_runs(save_path, valid_configs)

    results_by_key: dict[tuple[str, float, float], dict] = {}
    for key, run in cached_by_key.items():
        results_by_key[key] = run

    cached_count = 0
    new_count = 0
    start_time = time.time()

    for run_idx, (opt_name, lr, wd) in enumerate(configs, start=1):
        print(f"[{run_idx}/{total_runs}] {opt_name} lr={lr} wd={wd} ...", flush=True)

        key = _config_key(opt_name, lr, wd)
        if key in cached_by_key:
            cached_count += 1
            cached = cached_by_key[key]
            cached_train = _safe_float(cached.get("final_train_acc"))
            cached_test = _safe_float(cached.get("final_test_acc"))
            if cached_train is None:
                cached_train = 0.0
            if cached_test is None:
                cached_test = 0.0
            print(
                f"        -> cached {cached.get('outcome', 'unknown')} "
                f"(train={cached_train:.3f}, "
                f"test={cached_test:.3f})",
                flush=True,
            )
            continue

        result = train_run(
            optimizer_name=opt_name,
            lr=lr,
            weight_decay=wd,
            train_a=train_a,
            train_b=train_b,
            train_t=train_t,
            test_a=test_a,
            test_b=test_b,
            test_t=test_t,
            max_epochs=max_epochs,
            batch_size=batch_size,
            p=PRIME,
            seed=SEED,
            log_interval=log_interval,
        )
        results_by_key[key] = result
        new_count += 1

        elapsed = time.time() - start_time
        print(
            f"        -> {result['outcome']} "
            f"(train={result['final_train_acc']:.3f}, "
            f"test={result['final_test_acc']:.3f}) "
            f"[{elapsed:.0f}s elapsed]",
            flush=True,
        )

        ordered_runs = [results_by_key[cfg] for cfg in configs if cfg in results_by_key]
        snapshot_meta = _build_metadata(
            optimizers=optimizers,
            learning_rates=learning_rates,
            weight_decays=weight_decays,
            max_epochs=max_epochs,
            log_interval=log_interval,
            batch_size=batch_size,
            total_runs=total_runs,
            completed_runs=len(ordered_runs),
            total_seconds=prior_total_seconds + elapsed,
            train_examples=train_a.size(0),
            test_examples=test_a.size(0),
            resumed_from_cache=resume and cached_count > 0,
        )
        _write_results(save_path, snapshot_meta, ordered_runs)

    elapsed_total = time.time() - start_time
    cumulative_seconds = prior_total_seconds + elapsed_total
    final_runs = [results_by_key[cfg] for cfg in configs if cfg in results_by_key]
    final_meta = _build_metadata(
        optimizers=optimizers,
        learning_rates=learning_rates,
        weight_decays=weight_decays,
        max_epochs=max_epochs,
        log_interval=log_interval,
        batch_size=batch_size,
        total_runs=total_runs,
        completed_runs=len(final_runs),
        total_seconds=cumulative_seconds,
        train_examples=train_a.size(0),
        test_examples=test_a.size(0),
        resumed_from_cache=resume and cached_count > 0,
    )
    _write_results(save_path, final_meta, final_runs)

    print(
        f"\nSweep complete: {len(final_runs)}/{total_runs} runs "
        f"({new_count} newly trained, {cached_count} cached) "
        f"in {elapsed_total:.0f}s (cumulative {cumulative_seconds:.0f}s)",
        flush=True,
    )
    print(f"Results saved to {save_path}", flush=True)
    return final_runs
