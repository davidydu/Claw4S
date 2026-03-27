"""Experiment runner: sweep noise levels x architectures x seeds."""

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from src.data import build_datasets
from src.models import (
    ARCH_CONFIGS,
    WIDTH_SWEEP_WIDTHS,
    WIDTH_SWEEP_DEPTH,
    build_model,
    build_width_model,
    count_parameters,
)
from src.train import train_model, evaluate


NOISE_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
SEEDS = [42, 43, 44]
N_EPOCHS = 100
LR = 0.01
BATCH_SIZE = 64
N_SAMPLES = 500
N_FEATURES = 10
N_CLASSES = 5


@dataclass
class RunResult:
    """Result of a single training run."""
    arch: str
    depth: int
    width: int
    n_params: int
    noise_frac: float
    seed: int
    train_acc: float
    test_acc: float
    gen_gap: float  # train_acc - test_acc
    wall_seconds: float


def run_single(
    arch_name: str,
    depth: int,
    width: int,
    noise_frac: float,
    seed: int,
    model_builder: callable,
) -> RunResult:
    """Execute one training run and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, test_ds, _ = build_datasets(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        noise_frac=noise_frac,
        seed=seed,
    )

    model = model_builder()
    n_params = count_parameters(model)

    t0 = time.time()
    train_model(
        model, train_ds,
        n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=seed,
    )
    wall = time.time() - t0

    train_acc = evaluate(model, train_ds)
    test_acc = evaluate(model, test_ds)

    return RunResult(
        arch=arch_name,
        depth=depth,
        width=width,
        n_params=n_params,
        noise_frac=noise_frac,
        seed=seed,
        train_acc=round(train_acc, 4),
        test_acc=round(test_acc, 4),
        gen_gap=round(train_acc - test_acc, 4),
        wall_seconds=round(wall, 3),
    )


def run_architecture_sweep() -> list[RunResult]:
    """Sweep noise x architecture x seed (core experiment)."""
    results: list[RunResult] = []
    total = len(NOISE_FRACS) * len(ARCH_CONFIGS) * len(SEEDS)
    done = 0

    for noise in NOISE_FRACS:
        for arch_name, (depth, width, _) in ARCH_CONFIGS.items():
            for seed in SEEDS:
                r = run_single(
                    arch_name=arch_name,
                    depth=depth,
                    width=width,
                    noise_frac=noise,
                    seed=seed,
                    model_builder=lambda d=depth, w=width: build_model(
                        # Need to look up by depth/width, but build_model
                        # uses arch_name.  Build directly instead.
                        arch_name=arch_name,
                    ),
                )
                results.append(r)
                done += 1
                print(
                    f"  [{done}/{total}] arch={arch_name:14s} "
                    f"noise={noise:.0%}  seed={seed}  "
                    f"test_acc={r.test_acc:.3f}  gap={r.gen_gap:+.3f}  "
                    f"({r.wall_seconds:.1f}s)"
                )

    return results


def run_width_sweep() -> list[RunResult]:
    """Sweep noise x width (depth fixed at 2) x seed."""
    results: list[RunResult] = []
    total = len(NOISE_FRACS) * len(WIDTH_SWEEP_WIDTHS) * len(SEEDS)
    done = 0

    for noise in NOISE_FRACS:
        for width in WIDTH_SWEEP_WIDTHS:
            for seed in SEEDS:
                r = run_single(
                    arch_name=f"d{WIDTH_SWEEP_DEPTH}_w{width}",
                    depth=WIDTH_SWEEP_DEPTH,
                    width=width,
                    noise_frac=noise,
                    seed=seed,
                    model_builder=lambda w=width: build_width_model(w),
                )
                results.append(r)
                done += 1
                print(
                    f"  [{done}/{total}] width={width:4d}  "
                    f"noise={noise:.0%}  seed={seed}  "
                    f"test_acc={r.test_acc:.3f}  gap={r.gen_gap:+.3f}  "
                    f"({r.wall_seconds:.1f}s)"
                )

    return results


def run_all(results_dir: str = "results") -> dict:
    """Run both sweeps, save results, return summary dict."""
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Architecture sweep (shallow-wide / medium / deep-narrow)")
    print("=" * 60)
    arch_results = run_architecture_sweep()

    print()
    print("=" * 60)
    print("PHASE 2: Width sweep (depth=2, widths 16-256)")
    print("=" * 60)
    width_results = run_width_sweep()

    all_results = arch_results + width_results

    # Save raw results
    raw_path = os.path.join(results_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved {len(all_results)} results to {raw_path}")

    # Compute summary statistics
    summary = compute_summary(arch_results, width_results)
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    return summary


def compute_summary(
    arch_results: list[RunResult],
    width_results: list[RunResult],
) -> dict:
    """Aggregate results into mean +/- std across seeds."""
    summary: dict = {"architecture_sweep": {}, "width_sweep": {}, "findings": []}

    # --- Architecture sweep aggregation ---
    for arch_name in ARCH_CONFIGS:
        summary["architecture_sweep"][arch_name] = {}
        for noise in NOISE_FRACS:
            matching = [
                r for r in arch_results
                if r.arch == arch_name and r.noise_frac == noise
            ]
            test_accs = [r.test_acc for r in matching]
            train_accs = [r.train_acc for r in matching]
            gaps = [r.gen_gap for r in matching]
            noise_key = f"{noise:.0%}"
            summary["architecture_sweep"][arch_name][noise_key] = {
                "test_acc_mean": round(float(np.mean(test_accs)), 4),
                "test_acc_std": round(float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0, 4),
                "train_acc_mean": round(float(np.mean(train_accs)), 4),
                "train_acc_std": round(float(np.std(train_accs, ddof=1)) if len(train_accs) > 1 else 0.0, 4),
                "gen_gap_mean": round(float(np.mean(gaps)), 4),
                "gen_gap_std": round(float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0, 4),
                "n_runs": len(matching),
            }

    # --- Width sweep aggregation ---
    for width in WIDTH_SWEEP_WIDTHS:
        wname = f"d{WIDTH_SWEEP_DEPTH}_w{width}"
        summary["width_sweep"][wname] = {}
        for noise in NOISE_FRACS:
            matching = [
                r for r in width_results
                if r.width == width and r.noise_frac == noise
            ]
            test_accs = [r.test_acc for r in matching]
            train_accs = [r.train_acc for r in matching]
            gaps = [r.gen_gap for r in matching]
            noise_key = f"{noise:.0%}"
            summary["width_sweep"][wname][noise_key] = {
                "test_acc_mean": round(float(np.mean(test_accs)), 4),
                "test_acc_std": round(float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0, 4),
                "train_acc_mean": round(float(np.mean(train_accs)), 4),
                "train_acc_std": round(float(np.std(train_accs, ddof=1)) if len(train_accs) > 1 else 0.0, 4),
                "gen_gap_mean": round(float(np.mean(gaps)), 4),
                "gen_gap_std": round(float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0, 4),
                "n_runs": len(matching),
            }

    # --- Derive findings ---
    findings = derive_findings(summary)
    summary["findings"] = findings

    return summary


def derive_findings(summary: dict) -> list[str]:
    """Automatically derive key findings from summary statistics."""
    findings: list[str] = []
    arch_data = summary["architecture_sweep"]
    width_data = summary["width_sweep"]

    # Finding 1: Compare architectures at 0% and 50% noise
    for noise_key in ["0%", "50%"]:
        accs = {}
        for arch in arch_data:
            if noise_key in arch_data[arch]:
                accs[arch] = arch_data[arch][noise_key]["test_acc_mean"]
        if accs:
            best = max(accs, key=accs.get)
            worst = min(accs, key=accs.get)
            findings.append(
                f"At {noise_key} noise: best={best} ({accs[best]:.3f}), "
                f"worst={worst} ({accs[worst]:.3f}), "
                f"delta={accs[best] - accs[worst]:.3f}"
            )

    # Finding 2: Noise robustness (accuracy drop from 0% to 50%)
    for arch in arch_data:
        if "0%" in arch_data[arch] and "50%" in arch_data[arch]:
            clean = arch_data[arch]["0%"]["test_acc_mean"]
            noisy = arch_data[arch]["50%"]["test_acc_mean"]
            drop = clean - noisy
            findings.append(
                f"{arch}: accuracy drop 0%->50% noise = {drop:.3f} "
                f"({clean:.3f} -> {noisy:.3f})"
            )

    # Finding 3: Width effect on noise tolerance
    clean_accs = {}
    noisy_accs = {}
    for wname in width_data:
        if "0%" in width_data[wname] and "50%" in width_data[wname]:
            clean_accs[wname] = width_data[wname]["0%"]["test_acc_mean"]
            noisy_accs[wname] = width_data[wname]["50%"]["test_acc_mean"]
    if clean_accs:
        drops = {w: clean_accs[w] - noisy_accs[w] for w in clean_accs}
        most_robust = min(drops, key=drops.get)
        least_robust = max(drops, key=drops.get)
        findings.append(
            f"Width sweep: most noise-robust={most_robust} (drop={drops[most_robust]:.3f}), "
            f"least robust={least_robust} (drop={drops[least_robust]:.3f})"
        )

    # Finding 4: Generalization gap at high noise
    for arch in arch_data:
        if "50%" in arch_data[arch]:
            gap = arch_data[arch]["50%"]["gen_gap_mean"]
            findings.append(f"{arch} gen gap at 50% noise: {gap:+.3f}")

    return findings
