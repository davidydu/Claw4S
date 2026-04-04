"""Generate summary report and figures from experiment results.

Produces:
- results/summary.json  -- all aggregated metrics
- results/figures/*.png  -- key visualisation figures
- results/tables/*.csv   -- summary tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import (
    aggregate_results,
    build_summary_table,
    compute_manipulation_speed,
    compute_resilience_ranking,
)
from src.experiment import SimResult


def save_summary_json(
    aggregated: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Write aggregated metrics to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)


def save_summary_table_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write a summary table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")


def plot_belief_error_heatmap(
    aggregated: dict[str, dict[str, Any]],
    regime: str,
    noise: float,
    output_path: Path,
) -> None:
    """Plot a heatmap of final belief error for all learner-adversary pairs."""
    learners = ["NL", "SL", "AL"]
    adversaries = ["RA", "SA", "PA"]

    data = np.full((len(learners), len(adversaries)), np.nan)
    for i, lc in enumerate(learners):
        for j, ac in enumerate(adversaries):
            key = f"{lc}-vs-{ac}_{regime}_noise{noise}"
            if key in aggregated:
                data[i, j] = aggregated[key].get(
                    "distortion.final_belief_error.mean", np.nan
                )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(adversaries)))
    ax.set_xticklabels(adversaries)
    ax.set_yticks(range(len(learners)))
    ax.set_yticklabels(learners)
    ax.set_xlabel("Adversary")
    ax.set_ylabel("Learner")
    ax.set_title(f"Final Belief Error ({regime}, noise={noise})")

    # Add text annotations.
    for i in range(len(learners)):
        for j in range(len(adversaries)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)

    plt.colorbar(im, ax=ax, label="Belief Error")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_belief_error_timeseries(
    results: list[SimResult],
    learner_code: str,
    regime: str,
    noise: float,
    output_path: Path,
) -> None:
    """Plot belief error over time for one learner against all adversaries."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"RA": "#2196F3", "SA": "#F44336", "PA": "#FF9800"}

    for ac in ["RA", "SA", "PA"]:
        matching = [
            r for r in results
            if r.config.learner_code == learner_code
            and r.config.adversary_code == ac
            and r.config.drift_regime == regime
            and r.config.noise_level == noise
        ]
        if not matching:
            continue

        # Average across seeds.
        min_len = min(len(r.belief_error_timeseries) for r in matching)
        stacked = np.array([r.belief_error_timeseries[:min_len] for r in matching])
        mean_ts = np.mean(stacked, axis=0)
        interval = matching[0].config.belief_sample_interval
        x = np.arange(min_len) * interval

        ax.plot(x, mean_ts, label=ac, color=colors[ac], linewidth=1.5)

        if len(matching) > 1:
            std_ts = np.std(stacked, axis=0)
            ax.fill_between(x, mean_ts - std_ts, mean_ts + std_ts,
                          color=colors[ac], alpha=0.2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Belief Error")
    ax.set_title(f"Belief Error: {learner_code} vs Adversaries ({regime}, noise={noise})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_manipulation_speed(
    speeds: dict[str, dict[str, float]],
    regime: str,
    noise: float,
    output_path: Path,
) -> None:
    """Bar chart of manipulation speed (rounds to reach threshold)."""
    learners = ["NL", "SL", "AL"]
    adversaries = ["RA", "SA", "PA"]
    colors = {"RA": "#2196F3", "SA": "#F44336", "PA": "#FF9800"}

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(learners))
    width = 0.25

    for i, ac in enumerate(adversaries):
        means = []
        stds = []
        for lc in learners:
            key = f"{lc}-vs-{ac}_{regime}_noise{noise}"
            if key in speeds:
                means.append(speeds[key]["mean_rounds"])
                stds.append(speeds[key]["std_rounds"])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=ac,
               color=colors[ac], capsize=3)

    ax.set_xlabel("Learner")
    ax.set_ylabel("Rounds to Threshold")
    ax.set_title(f"Manipulation Speed ({regime}, noise={noise})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(learners)
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(
    aggregated: dict[str, dict[str, Any]],
    noise: float,
    output_path: Path,
) -> None:
    """Grouped bar chart of accuracy across regimes."""
    learners = ["NL", "SL", "AL"]
    adversaries = ["SA"]  # Focus on strategic adversary.
    regimes = ["stable", "slow_drift", "volatile"]
    colors = {"stable": "#4CAF50", "slow_drift": "#FF9800", "volatile": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for idx, lc in enumerate(learners):
        ax = axes[idx]
        x = np.arange(len(regimes))
        means = []
        stds = []
        for regime in regimes:
            key = f"{lc}-vs-SA_{regime}_noise{noise}"
            if key in aggregated:
                means.append(aggregated[key].get("decision_quality.accuracy.mean", 0))
                stds.append(aggregated[key].get("decision_quality.accuracy.std", 0))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x, means, yerr=stds, color=[colors[r] for r in regimes], capsize=3)
        ax.set_title(f"{lc} vs SA")
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=30)
        ax.set_ylabel("Accuracy" if idx == 0 else "")
        ax.set_ylim(0, 0.5)

    plt.suptitle(f"Decision Accuracy vs Strategic Adversary (noise={noise})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_full_report(
    results: list[SimResult],
    output_dir: Path,
) -> dict[str, Any]:
    """Generate the complete report: JSON, CSVs, and figures.

    Returns the aggregated metrics dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    # 1. Aggregate.
    aggregated = aggregate_results(results)
    save_summary_json(aggregated, output_dir / "summary.json")

    # 2. Summary tables.
    for metric in [
        "distortion.mean_belief_error",
        "distortion.final_belief_error",
        "decision_quality.accuracy",
        "credibility.exploitation_gap",
    ]:
        rows = build_summary_table(aggregated, metric)
        safe_name = metric.replace(".", "_")
        save_summary_table_csv(rows, table_dir / f"{safe_name}.csv")

    # 3. Manipulation speed.
    speeds = compute_manipulation_speed(results)
    save_summary_json(speeds, output_dir / "manipulation_speed.json")

    # 4. Resilience ranking.
    for noise in [0.0, 0.1]:
        ranking = compute_resilience_ranking(aggregated, "SA", noise)
        save_summary_table_csv(
            ranking, table_dir / f"resilience_sa_noise{noise}.csv"
        )

    # 5. Figures.
    for regime in ["stable", "slow_drift", "volatile"]:
        for noise in [0.0, 0.1]:
            plot_belief_error_heatmap(
                aggregated, regime, noise,
                fig_dir / f"heatmap_{regime}_noise{noise}.png",
            )
            plot_manipulation_speed(
                speeds, regime, noise,
                fig_dir / f"speed_{regime}_noise{noise}.png",
            )
            for lc in ["NL", "SL", "AL"]:
                plot_belief_error_timeseries(
                    results, lc, regime, noise,
                    fig_dir / f"timeseries_{lc}_{regime}_noise{noise}.png",
                )

    for noise in [0.0, 0.1]:
        plot_accuracy_comparison(
            aggregated, noise,
            fig_dir / f"accuracy_comparison_noise{noise}.png",
        )

    print(f"Report saved to {output_dir}")
    print(f"  - summary.json")
    print(f"  - {len(list(table_dir.glob('*.csv')))} CSV tables")
    print(f"  - {len(list(fig_dir.glob('*.png')))} figures")

    return aggregated
