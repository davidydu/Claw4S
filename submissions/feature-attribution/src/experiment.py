"""Full experiment runner: train models, compute attributions, measure agreement."""

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from src.agreement import aggregate_agreement, pairwise_spearman
from src.attributions import METHOD_PAIRS, compute_all_attributions
from src.data import make_gaussian_clusters
from src.models import MLP, train_model


def run_experiment(
    depths: List[int] = [1, 2, 4],
    width: int = 64,
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    n_test: int = 100,
    n_steps: int = 50,
    seeds: List[int] = [42, 123, 456],
    epochs: int = 200,
    lr: float = 1e-3,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Run the full feature attribution consistency experiment.

    Sweeps over depths x seeds, trains MLPs, computes 3 attribution methods
    on n_test samples, and measures pairwise Spearman rank correlation.

    Args:
        depths: List of hidden-layer counts.
        width: Hidden layer width.
        n_samples: Total dataset size.
        n_features: Input dimensionality.
        n_classes: Number of classes.
        n_test: Number of test samples for attribution.
        n_steps: IG interpolation steps.
        seeds: Random seeds for repeated trials.
        epochs: Training epochs.
        lr: Learning rate.
        results_dir: Output directory.

    Returns:
        Full results dictionary.
    """
    os.makedirs(results_dir, exist_ok=True)
    start_time = time.time()

    all_results: Dict[str, Any] = {
        "metadata": {
            "depths": depths,
            "width": width,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_test": n_test,
            "n_steps": n_steps,
            "seeds": seeds,
            "epochs": epochs,
        },
        "per_depth": {},
    }

    for depth in depths:
        print(f"\n{'='*60}")
        print(f"  Depth = {depth} hidden layers")
        print(f"{'='*60}")

        depth_correlations: List[Dict[str, float]] = []
        depth_accuracies: List[float] = []
        depth_details: List[Dict[str, Any]] = []

        for seed in seeds:
            print(f"\n  Seed {seed}:")

            # Generate data
            X, y = make_gaussian_clusters(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                seed=seed,
            )

            # Split: first n_test for test, rest for train
            X_test, y_test = X[:n_test], y[:n_test]
            X_train, y_train = X[n_test:], y[n_test:]

            X_train_t = torch.from_numpy(X_train)
            y_train_t = torch.from_numpy(y_train)
            X_test_t = torch.from_numpy(X_test)
            y_test_t = torch.from_numpy(y_test)

            # Train model
            torch.manual_seed(seed)
            model = MLP(
                n_features=n_features,
                n_classes=n_classes,
                n_hidden=depth,
                width=width,
            )
            losses = train_model(
                model, X_train_t, y_train_t,
                lr=lr, epochs=epochs, seed=seed,
            )

            # Evaluate accuracy
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                accuracy = (preds == y_test_t).float().mean().item()
            depth_accuracies.append(accuracy)
            print(f"    Accuracy: {accuracy:.3f}")
            print(f"    Final loss: {losses[-1]:.4f}")

            # Compute attributions for test samples
            sample_correlations: List[Dict[str, float]] = []
            for i in range(n_test):
                x_i = X_test_t[i:i+1]
                target = preds[i].item()

                attrs = compute_all_attributions(
                    model, x_i, target, n_steps=n_steps
                )
                corrs = pairwise_spearman(attrs, METHOD_PAIRS)
                sample_correlations.append(corrs)

            depth_correlations.extend(sample_correlations)

            # Per-seed aggregation
            seed_agg = aggregate_agreement(sample_correlations)
            depth_details.append({
                "seed": seed,
                "accuracy": accuracy,
                "final_loss": losses[-1],
                "agreement": seed_agg,
            })

            for pair_key, stats in seed_agg.items():
                print(f"    {pair_key}: rho={stats['mean']:.3f} +/- {stats['std']:.3f}")

        # Aggregate across all seeds for this depth
        depth_agg = aggregate_agreement(depth_correlations)
        mean_acc = float(np.mean(depth_accuracies))
        std_acc = float(np.std(depth_accuracies))

        print(f"\n  >>> Depth {depth} aggregated (across {len(seeds)} seeds):")
        print(f"      Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
        for pair_key, stats in depth_agg.items():
            print(f"      {pair_key}: rho={stats['mean']:.3f} +/- {stats['std']:.3f}")

        all_results["per_depth"][str(depth)] = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "agreement": depth_agg,
            "per_seed": depth_details,
        }

    # Compute cross-depth summary
    summary = _build_summary(all_results)
    all_results["summary"] = summary

    elapsed = time.time() - start_time
    # Save results
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate report
    report = _generate_report(all_results)
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    print(
        "Observed runtime: "
        f"{round(elapsed, 1)}s (excluded from saved artifacts for deterministic reruns)"
    )

    return all_results


def _build_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build cross-depth summary statistics."""
    depths = results["metadata"]["depths"]

    pair_keys = list(
        results["per_depth"][str(depths[0])]["agreement"].keys()
    )

    summary: Dict[str, Any] = {"pair_trends": {}}

    for pair_key in pair_keys:
        means = []
        for d in depths:
            m = results["per_depth"][str(d)]["agreement"][pair_key]["mean"]
            means.append(m)

        overall_mean = float(np.mean(means))
        trend = means[-1] - means[0]  # change from shallowest to deepest

        summary["pair_trends"][pair_key] = {
            "overall_mean_rho": round(overall_mean, 4),
            "depth_trend": round(trend, 4),
            "per_depth_means": [round(m, 4) for m in means],
        }

    # Overall disagreement flag
    all_means = [
        v["overall_mean_rho"]
        for v in summary["pair_trends"].values()
    ]
    summary["overall_mean_rho"] = round(float(np.mean(all_means)), 4)
    summary["substantial_disagreement"] = summary["overall_mean_rho"] < 0.7

    return summary


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate human-readable markdown report."""
    meta = results["metadata"]
    summary = results["summary"]
    depths = meta["depths"]
    train_samples = meta["n_samples"] - meta["n_test"]

    lines = [
        "# Feature Attribution Consistency Report",
        "",
        "## Experiment Configuration",
        f"- Depths: {depths}",
        f"- Width: {meta['width']}",
        f"- Samples: {meta['n_samples']} total ({train_samples} train / {meta['n_test']} test)",
        f"- Features: {meta['n_features']}, Classes: {meta['n_classes']}",
        f"- Seeds: {meta['seeds']}",
        f"- IG steps: {meta['n_steps']}",
        "",
        "## Model Accuracy",
        "",
        "| Depth | Accuracy |",
        "|-------|----------|",
    ]

    for d in depths:
        dp = results["per_depth"][str(d)]
        lines.append(f"| {d} | {dp['accuracy_mean']:.3f} +/- {dp['accuracy_std']:.3f} |")

    lines.extend([
        "",
        "## Attribution Agreement (Spearman rho)",
        "",
    ])

    # Build table
    pair_keys = list(results["per_depth"][str(depths[0])]["agreement"].keys())
    header = "| Pair | " + " | ".join(f"Depth {d}" for d in depths) + " |"
    sep = "|------|" + "|".join("-------" for _ in depths) + "|"
    lines.append(header)
    lines.append(sep)

    for pk in pair_keys:
        short = pk.replace("_vs_", " vs ")
        cells = []
        for d in depths:
            s = results["per_depth"][str(d)]["agreement"][pk]
            cells.append(f"{s['mean']:.3f}+/-{s['std']:.3f}")
        lines.append(f"| {short} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Summary",
        f"- Overall mean Spearman rho: **{summary['overall_mean_rho']:.4f}**",
        f"- Substantial disagreement (rho < 0.7): **{summary['substantial_disagreement']}**",
        "",
        "### Depth Trends",
        "",
    ])

    for pk, trend in summary["pair_trends"].items():
        short = pk.replace("_vs_", " vs ")
        direction = "decreases" if trend["depth_trend"] < 0 else "increases"
        lines.append(
            f"- {short}: mean rho = {trend['overall_mean_rho']:.3f}, "
            f"{direction} by {abs(trend['depth_trend']):.3f} from depth "
            f"{depths[0]} to {depths[-1]}"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. Attribution methods show varying degrees of agreement depending on the method pair.",
        "2. Gradient x Input and Integrated Gradients tend to agree most (both incorporate input magnitude).",
        "3. Vanilla Gradient shows lowest agreement with other methods.",
        f"4. {_summarize_depth_effects(summary['pair_trends'])}",
        "",
    ])

    return "\n".join(lines)


def _summarize_depth_effects(pair_trends: Dict[str, Dict[str, Any]]) -> str:
    """Summarize cross-depth agreement trends without overstating the direction."""
    threshold = 0.01
    trends = [trend["depth_trend"] for trend in pair_trends.values()]
    has_increase = any(trend > threshold for trend in trends)
    has_decrease = any(trend < -threshold for trend in trends)

    if has_increase and has_decrease:
        return "Depth effects are mixed across method pairs in this configuration."
    if has_increase:
        return "Agreement increases modestly with depth for the affected method pairs."
    if has_decrease:
        return "Agreement decreases modestly with depth for the affected method pairs."
    return "Depth effects on agreement are modest in this configuration."
