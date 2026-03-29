"""Full experiment: train model pairs and compute transfer rates."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import make_gaussian_clusters
from src.models import MLP
from src.train import train_model
from src.adversarial import compute_transfer_rate


# Experiment configuration
WIDTHS = [32, 64, 128, 256]
SEEDS = [42, 43, 44]
EPSILON = 0.3
N_SAMPLES = 500
N_FEATURES = 10
N_CLASSES = 5
TRAIN_EPOCHS = 50
TRAIN_LR = 0.01
TRAIN_BATCH_SIZE = 64


def run_same_arch_experiment(results_dir: Path) -> list[dict]:
    """Run same-architecture (2-layer) transfer experiment.

    Trains 4 widths x 3 seeds = 12 models, then evaluates all 16 pairs
    x 3 seeds = 48 transfer evaluations.

    Args:
        results_dir: Directory to save results.

    Returns:
        List of result dicts for each (source_width, target_width, seed) triple.
    """
    results = []

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        dataset = make_gaussian_clusters(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_classes=N_CLASSES,
            seed=seed,
        )
        X_all = dataset.tensors[0]
        y_all = dataset.tensors[1]

        # Train all models for this seed
        models: dict[int, MLP] = {}
        for width in WIDTHS:
            print(f"  Training 2-layer MLP width={width}...", end=" ")
            torch.manual_seed(seed)
            model = MLP(
                input_dim=N_FEATURES,
                n_classes=N_CLASSES,
                hidden_width=width,
                n_hidden_layers=2,
            )
            stats = train_model(
                model, dataset,
                lr=TRAIN_LR, epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE, seed=seed,
            )
            models[width] = model
            print(f"acc={stats['final_accuracy']:.3f}")

        # Evaluate all pairs
        for src_w in WIDTHS:
            for tgt_w in WIDTHS:
                result = compute_transfer_rate(
                    source_model=models[src_w],
                    target_model=models[tgt_w],
                    X=X_all,
                    y=y_all,
                    epsilon=EPSILON,
                )
                entry = {
                    "experiment": "same_arch",
                    "source_width": src_w,
                    "target_width": tgt_w,
                    "source_depth": 2,
                    "target_depth": 2,
                    "source_params": models[src_w].param_count(),
                    "target_params": models[tgt_w].param_count(),
                    "capacity_ratio": tgt_w / src_w,
                    "seed": seed,
                    "epsilon": EPSILON,
                    **result,
                }
                results.append(entry)
                print(
                    f"  {src_w}->{tgt_w}: transfer_rate={result['transfer_rate']:.3f} "
                    f"(n_adv={result['n_successful_source_advs']})"
                )

    return results


def run_cross_depth_experiment(results_dir: Path) -> list[dict]:
    """Run cross-depth experiment: 2-layer source, 4-layer target.

    Tests whether depth mismatch affects transferability differently
    from width mismatch alone.

    Args:
        results_dir: Directory to save results.

    Returns:
        List of result dicts.
    """
    results = []

    for seed in SEEDS:
        print(f"\n=== Cross-depth, Seed {seed} ===")
        dataset = make_gaussian_clusters(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_classes=N_CLASSES,
            seed=seed,
        )
        X_all = dataset.tensors[0]
        y_all = dataset.tensors[1]

        source_models: dict[int, MLP] = {}
        for src_w in WIDTHS:
            torch.manual_seed(seed)
            source = MLP(N_FEATURES, N_CLASSES, src_w, n_hidden_layers=2)
            train_model(
                source,
                dataset,
                lr=TRAIN_LR,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                seed=seed,
            )
            source_models[src_w] = source

        target_models: dict[int, MLP] = {}
        for tgt_w in WIDTHS:
            torch.manual_seed(seed)
            target = MLP(N_FEATURES, N_CLASSES, tgt_w, n_hidden_layers=4)
            train_model(
                target,
                dataset,
                lr=TRAIN_LR,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                seed=seed,
            )
            target_models[tgt_w] = target

        for src_w in WIDTHS:
            for tgt_w in WIDTHS:
                result = compute_transfer_rate(
                    source_model=source_models[src_w],
                    target_model=target_models[tgt_w],
                    X=X_all,
                    y=y_all,
                    epsilon=EPSILON,
                )
                entry = {
                    "experiment": "cross_depth",
                    "source_width": src_w,
                    "target_width": tgt_w,
                    "source_depth": 2,
                    "target_depth": 4,
                    "source_params": source_models[src_w].param_count(),
                    "target_params": target_models[tgt_w].param_count(),
                    "capacity_ratio": tgt_w / src_w,
                    "seed": seed,
                    "epsilon": EPSILON,
                    **result,
                }
                results.append(entry)
                print(
                    f"  2L-w{src_w} -> 4L-w{tgt_w}: "
                    f"transfer={result['transfer_rate']:.3f}"
                )

    return results


def run_full_experiment(results_dir: Path) -> dict:
    """Run both experiments and save results.

    Args:
        results_dir: Directory to save JSON results.

    Returns:
        Dict with 'same_arch' and 'cross_depth' result lists, plus summary stats.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("=" * 60)
    print("PHASE 1: Same-architecture transfer (2L vs 2L)")
    print("=" * 60)
    same_arch = run_same_arch_experiment(results_dir)

    print("\n" + "=" * 60)
    print("PHASE 2: Cross-depth transfer (2L source -> 4L target)")
    print("=" * 60)
    cross_depth = run_cross_depth_experiment(results_dir)

    elapsed = time.time() - start

    # Compute summary statistics
    summary = compute_summary(same_arch, cross_depth)
    summary["runtime_seconds"] = round(elapsed, 1)

    output = {
        "same_arch_results": same_arch,
        "cross_depth_results": cross_depth,
        "summary": summary,
        "config": {
            "widths": WIDTHS,
            "seeds": SEEDS,
            "epsilon": EPSILON,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "train_epochs": TRAIN_EPOCHS,
            "train_lr": TRAIN_LR,
            "train_batch_size": TRAIN_BATCH_SIZE,
        },
    }

    out_path = results_dir / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total runtime: {elapsed:.1f}s")

    return output


def compute_summary(same_arch: list[dict], cross_depth: list[dict]) -> dict:
    """Compute summary statistics from raw results.

    Args:
        same_arch: List of same-architecture result dicts.
        cross_depth: List of cross-depth result dicts.

    Returns:
        Summary dict with aggregated statistics.
    """
    # Same-arch: aggregate by (source_width, target_width)
    same_arch_grid = {}
    for r in same_arch:
        key = (r["source_width"], r["target_width"])
        same_arch_grid.setdefault(key, []).append(r["transfer_rate"])

    same_arch_means = {}
    same_arch_stds = {}
    for key, rates in same_arch_grid.items():
        same_arch_means[f"{key[0]}->{key[1]}"] = round(float(np.mean(rates)), 4)
        same_arch_stds[f"{key[0]}->{key[1]}"] = round(float(np.std(rates)), 4)

    # Diagonal (same-width) vs off-diagonal
    diag_rates = []
    off_diag_rates = []
    for r in same_arch:
        if r["source_width"] == r["target_width"]:
            diag_rates.append(r["transfer_rate"])
        else:
            off_diag_rates.append(r["transfer_rate"])

    # By capacity ratio
    ratio_groups: dict[float, list[float]] = {}
    for r in same_arch:
        ratio = r["capacity_ratio"]
        ratio_groups.setdefault(ratio, []).append(r["transfer_rate"])

    ratio_means = {
        str(ratio): round(float(np.mean(rates)), 4)
        for ratio, rates in sorted(ratio_groups.items())
    }

    # Cross-depth summary
    cross_depth_grid = {}
    for r in cross_depth:
        key = (r["source_width"], r["target_width"])
        cross_depth_grid.setdefault(key, []).append(r["transfer_rate"])

    cross_depth_means = {}
    for key, rates in cross_depth_grid.items():
        cross_depth_means[f"2L-w{key[0]}->4L-w{key[1]}"] = round(
            float(np.mean(rates)), 4
        )

    # Compare same-width same-depth vs same-width cross-depth
    same_width_same_depth = []
    same_width_cross_depth = []
    for r in same_arch:
        if r["source_width"] == r["target_width"]:
            same_width_same_depth.append(r["transfer_rate"])
    for r in cross_depth:
        if r["source_width"] == r["target_width"]:
            same_width_cross_depth.append(r["transfer_rate"])

    return {
        "same_arch_transfer_means": same_arch_means,
        "same_arch_transfer_stds": same_arch_stds,
        "diagonal_mean_transfer": round(float(np.mean(diag_rates)), 4) if diag_rates else None,
        "diagonal_std_transfer": round(float(np.std(diag_rates)), 4) if diag_rates else None,
        "off_diagonal_mean_transfer": round(float(np.mean(off_diag_rates)), 4) if off_diag_rates else None,
        "off_diagonal_std_transfer": round(float(np.std(off_diag_rates)), 4) if off_diag_rates else None,
        "transfer_by_capacity_ratio": ratio_means,
        "cross_depth_transfer_means": cross_depth_means,
        "same_width_same_depth_mean": round(float(np.mean(same_width_same_depth)), 4) if same_width_same_depth else None,
        "same_width_cross_depth_mean": round(float(np.mean(same_width_cross_depth)), 4) if same_width_cross_depth else None,
        "n_same_arch_runs": len(same_arch),
        "n_cross_depth_runs": len(cross_depth),
    }
