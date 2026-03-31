"""Visualization: heatmaps of grokking landscape and training curves.

Generates:
1. Heatmap: optimizer x (lr, wd) showing outcome
   (grokking/direct_generalization/memorization/failure)
2. Selected training curves showing grokking dynamics
3. Summary report in Markdown
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


OUTCOME_COLORS = {
    "grokking": "#2ecc71",      # green
    "direct_generalization": "#f1c40f",  # gold
    "memorization": "#e74c3c",  # red
    "failure": "#95a5a6",       # gray
}

OUTCOME_VALUES = {
    "grokking": 3,
    "direct_generalization": 2,
    "memorization": 1,
    "failure": 0,
}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a binomial rate."""
    if total <= 0:
        return 0.0, 0.0

    phat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (phat + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_results(results_dir: str = "results") -> dict:
    """Load sweep results from JSON.

    Args:
        results_dir: Directory containing sweep_results.json.

    Returns:
        Parsed JSON dict with metadata and runs.

    Raises:
        FileNotFoundError: If sweep_results.json does not exist.
    """
    path = os.path.join(results_dir, "sweep_results.json")
    with open(path) as f:
        return json.load(f)


def plot_heatmap(data: dict, results_dir: str = "results") -> str:
    """Create a heatmap of optimizer x hyperparameter showing outcomes.

    Args:
        data: Sweep results dict.
        results_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    runs = data["runs"]
    optimizers = data["metadata"]["optimizers"]
    learning_rates = data["metadata"]["learning_rates"]
    weight_decays = data["metadata"]["weight_decays"]

    n_opts = len(optimizers)
    n_cols = len(learning_rates) * len(weight_decays)

    # Build grid: rows=optimizers, cols=(lr, wd) combinations
    grid = np.zeros((n_opts, n_cols))
    col_labels = []
    for j, lr in enumerate(learning_rates):
        for k, wd in enumerate(weight_decays):
            col_labels.append(f"lr={lr}\nwd={wd}")

    for run in runs:
        row = optimizers.index(run["optimizer"])
        lr_idx = learning_rates.index(run["lr"])
        wd_idx = weight_decays.index(run["weight_decay"])
        col = lr_idx * len(weight_decays) + wd_idx
        grid[row, col] = OUTCOME_VALUES[run["outcome"]]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Custom colormap: gray=0, red=1, gold=2, green=3
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#95a5a6", "#e74c3c", "#f1c40f", "#2ecc71"])

    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_opts))
    ax.set_yticklabels(optimizers, fontsize=10)

    ax.set_xlabel("Learning Rate / Weight Decay", fontsize=11)
    ax.set_ylabel("Optimizer", fontsize=11)
    ax.set_title("Optimizer Grokking Landscape (mod 97 addition)", fontsize=13)

    # Add text annotations with final test accuracy
    for run in runs:
        row = optimizers.index(run["optimizer"])
        lr_idx = learning_rates.index(run["lr"])
        wd_idx = weight_decays.index(run["weight_decay"])
        col = lr_idx * len(weight_decays) + wd_idx
        test_acc = run["final_test_acc"]
        text_color = "white" if run["outcome"] == "failure" else "black"
        ax.text(col, row, f"{test_acc:.0%}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=text_color)

    # Legend
    patches = [
        mpatches.Patch(color=OUTCOME_COLORS["grokking"], label="Grokking"),
        mpatches.Patch(
            color=OUTCOME_COLORS["direct_generalization"],
            label="Direct Generalization",
        ),
        mpatches.Patch(color=OUTCOME_COLORS["memorization"], label="Memorization"),
        mpatches.Patch(color=OUTCOME_COLORS["failure"], label="Failure"),
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "grokking_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {save_path}", flush=True)
    return save_path


def plot_training_curves(data: dict, results_dir: str = "results") -> str:
    """Plot training curves for representative runs.

    Selects one grokking, one memorization, and one failure example.

    Args:
        data: Sweep results dict.
        results_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    runs = data["runs"]

    # Select representative runs
    examples = {}
    for outcome in ["grokking", "direct_generalization", "memorization", "failure"]:
        candidates = [r for r in runs if r["outcome"] == outcome]
        if candidates:
            # Pick the one with most interesting dynamics
            if outcome == "grokking":
                # Pick the one with largest gap between memorization and grokking
                candidates.sort(
                    key=lambda r: (r["grokking_epoch"] or 0) - (r["memorization_epoch"] or 0),
                    reverse=True,
                )
            examples[outcome] = candidates[0]

    n_plots = len(examples)
    if n_plots == 0:
        print("No runs to plot.", flush=True)
        return ""

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)

    for idx, (outcome, run) in enumerate(examples.items()):
        ax = axes[0, idx]
        history = run["history"]
        epochs = [h["epoch"] for h in history]
        train_accs = [h["train_acc"] for h in history]
        test_accs = [h["test_acc"] for h in history]

        ax.plot(epochs, train_accs, label="Train Acc", color="#3498db", linewidth=1.5)
        ax.plot(epochs, test_accs, label="Test Acc", color="#e74c3c", linewidth=1.5)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        title = (f"{outcome.upper()}\n"
                 f"{run['optimizer']} lr={run['lr']} wd={run['weight_decay']}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}", flush=True)
    return save_path


def generate_report(data: dict, results_dir: str = "results") -> str:
    """Generate a Markdown summary report.

    Args:
        data: Sweep results dict.
        results_dir: Directory to save the report.

    Returns:
        The report as a string.
    """
    runs = data["runs"]
    meta = data["metadata"]

    # Count outcomes per optimizer
    outcome_counts = {}
    for opt in meta["optimizers"]:
        opt_runs = [r for r in runs if r["optimizer"] == opt]
        counts = {
            "grokking": 0,
            "direct_generalization": 0,
            "memorization": 0,
            "failure": 0,
        }
        for r in opt_runs:
            counts[r["outcome"]] += 1
        outcome_counts[opt] = counts

    # Grokking delay statistics
    grok_delays = {}
    for opt in meta["optimizers"]:
        delays = []
        for r in runs:
            if r["optimizer"] == opt and r["outcome"] == "grokking":
                delay = (r["generalization_epoch"] or 0) - (r["memorization_epoch"] or 0)
                delays.append(delay)
        grok_delays[opt] = delays

    lines = [
        "# Optimizer Grokking Landscape: Summary Report",
        "",
        "## Experimental Setup",
        f"- **Task:** (a + b) mod {meta['prime']}",
        f"- **Model:** 2-layer MLP (embed=32, hidden=64)",
        f"- **Train/test split:** 70/30 (seed={meta['seed']})",
        f"- **Max epochs:** {meta['max_epochs']}",
        f"- **Batch size:** {meta['batch_size']}",
        f"- **Total runs:** {meta['num_runs']}",
        f"- **Runtime:** {meta['total_seconds']:.0f}s",
        "",
    ]

    if any(
        key in meta
        for key in ["python_version", "torch_version", "numpy_version", "platform", "generated_utc"]
    ):
        lines.extend([
            "## Reproducibility Provenance",
            f"- **Python:** {meta.get('python_version', 'unknown')}",
            f"- **PyTorch:** {meta.get('torch_version', 'unknown')}",
            f"- **NumPy:** {meta.get('numpy_version', 'unknown')}",
            f"- **Platform:** {meta.get('platform', 'unknown')}",
            f"- **Generated (UTC):** {meta.get('generated_utc', 'unknown')}",
            "",
        ])

    lines.extend([
        "## Outcome Summary",
        "",
        "| Optimizer | Grokking | Direct Gen. | Memorization | Failure |",
        "|-----------|----------|-------------|--------------|---------|",
    ])

    for opt in meta["optimizers"]:
        c = outcome_counts[opt]
        lines.append(
            f"| {opt} | {c['grokking']} | {c['direct_generalization']} | "
            f"{c['memorization']} | {c['failure']} |"
        )

    lines.extend([
        "",
        "## Grokking Delay (logged epochs from memorization to delayed generalization)",
        "",
        "| Optimizer | Min Delay | Max Delay | Mean Delay | Runs |",
        "|-----------|-----------|-----------|------------|------|",
    ])

    for opt in meta["optimizers"]:
        delays = grok_delays[opt]
        if delays:
            lines.append(
                f"| {opt} | {min(delays)} | {max(delays)} | {np.mean(delays):.0f} | {len(delays)} |"
            )
        else:
            lines.append(f"| {opt} | -- | -- | -- | 0 |")

    lines.extend([
        "",
        "## Statistical Uncertainty",
        "",
        "Wilson 95% CI for delayed-grokking rate per optimizer:",
        "",
        "| Optimizer | Delayed Grokking | Rate | Wilson 95% CI |",
        "|-----------|------------------|------|---------------|",
    ])

    for opt in meta["optimizers"]:
        opt_runs = [r for r in runs if r["optimizer"] == opt]
        total_opt = len(opt_runs)
        delayed = sum(1 for r in opt_runs if r["outcome"] == "grokking")
        rate = delayed / total_opt if total_opt else 0.0
        lower, upper = wilson_interval(delayed, total_opt)
        lines.append(
            f"| {opt} | {delayed}/{total_opt} | {rate * 100:.1f}% | "
            f"[{lower * 100:.1f}%, {upper * 100:.1f}%] |"
        )

    # Detailed per-run table
    lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Optimizer | LR | WD | Outcome | Train Acc | Test Acc | Mem Epoch | Gen Epoch | Grok Epoch |",
        "|-----------|------|------|---------|-----------|----------|-----------|-----------|------------|",
    ])

    for r in runs:
        mem = r["memorization_epoch"] if r["memorization_epoch"] is not None else "--"
        gen = r["generalization_epoch"] if r["generalization_epoch"] is not None else "--"
        grok = r["grokking_epoch"] if r["grokking_epoch"] is not None else "--"
        lines.append(
            f"| {r['optimizer']} | {r['lr']} | {r['weight_decay']} | "
            f"{r['outcome']} | {r['final_train_acc']:.3f} | {r['final_test_acc']:.3f} | "
            f"{mem} | {gen} | {grok} |"
        )

    # Key findings
    grok_count = sum(1 for r in runs if r["outcome"] == "grokking")
    direct_count = sum(1 for r in runs if r["outcome"] == "direct_generalization")
    mem_count = sum(1 for r in runs if r["outcome"] == "memorization")
    fail_count = sum(1 for r in runs if r["outcome"] == "failure")

    lines.extend([
        "",
        "## Key Findings",
        "",
        f"1. **Overall:** {grok_count}/{meta['num_runs']} runs showed delayed grokking, "
        f"{direct_count} reached direct generalization, {mem_count} memorized, "
        f"and {fail_count} failed to converge.",
    ])

    # Find best optimizer for grokking
    best_opt = max(meta["optimizers"], key=lambda o: outcome_counts[o]["grokking"])
    lines.append(
        f"2. **Most reliable grokker:** {best_opt} "
        f"({outcome_counts[best_opt]['grokking']}/{len(meta['learning_rates']) * len(meta['weight_decays'])} configs grokked)."
    )

    # Weight decay effect
    wd_zero_grok = sum(1 for r in runs if r["weight_decay"] == 0.0 and r["outcome"] == "grokking")
    wd_nonzero_grok = sum(1 for r in runs if r["weight_decay"] > 0.0 and r["outcome"] == "grokking")
    wd_nonzero_direct = sum(
        1
        for r in runs
        if r["weight_decay"] > 0.0 and r["outcome"] == "direct_generalization"
    )
    lines.append(
        f"3. **Weight decay effect:** {wd_zero_grok} delayed-grokking runs with wd=0 vs "
        f"{wd_nonzero_grok} delayed-grokking and {wd_nonzero_direct} direct-generalization "
        f"runs with wd>0."
    )

    lines.append("")

    report = "\n".join(lines)

    save_path = os.path.join(results_dir, "report.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {save_path}", flush=True)

    return report
