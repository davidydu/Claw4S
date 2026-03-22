"""Generate publication-quality plots for the sparsity analysis."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sparsity_evolution(experiments: list, output_dir: str) -> str:
    """Plot dead neuron fraction and zero fraction over training epochs.

    Creates one subplot per task (modular addition, regression),
    with one line per hidden width. Shows both dead neuron fraction
    (strict) and zero fraction (soft sparsity).

    Parameters
    ----------
    experiments : list of dict
        Experiment results.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to saved figure.
    """
    tasks = {}
    for exp in experiments:
        task = exp["task"]
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(exp)

    fig, axes = plt.subplots(2, len(tasks), figsize=(6 * len(tasks), 8), squeeze=False)
    fig.suptitle("Activation Sparsity During Training", fontsize=14, y=1.02)

    for idx, (task_name, exps) in enumerate(sorted(tasks.items())):
        ax_dead = axes[0][idx]
        ax_zero = axes[1][idx]

        for exp in sorted(exps, key=lambda e: e["hidden_dim"]):
            h = exp["history"]
            label = f"h={exp['hidden_dim']}"

            ax_dead.plot(h["epochs"], h["dead_neuron_fraction"],
                         label=label, marker="o", markersize=1.5)
            ax_zero.plot(h["epochs"], h["zero_fraction"],
                         label=label, marker="o", markersize=1.5)

        short_name = task_name.replace("_", " ").title()
        ax_dead.set_title(f"{short_name} - Dead Neuron Fraction")
        ax_dead.set_ylabel("Dead Neuron Fraction")
        ax_dead.legend(fontsize=8)
        ax_dead.set_ylim(-0.05, 1.05)
        ax_dead.grid(True, alpha=0.3)

        ax_zero.set_title(f"{short_name} - Zero Activation Fraction")
        ax_zero.set_xlabel("Epoch")
        ax_zero.set_ylabel("Zero Fraction")
        ax_zero.legend(fontsize=8)
        ax_zero.set_ylim(0, 1.0)
        ax_zero.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sparsity_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_and_sparsity(experiments: list, output_dir: str) -> str:
    """Plot test accuracy and sparsity on dual y-axes for modular addition.

    Shows whether grokking transitions coincide with sparsity transitions.

    Parameters
    ----------
    experiments : list of dict
        Experiment results.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to saved figure.
    """
    mod_exps = [e for e in experiments if "modular_addition" in e["task"]]
    if not mod_exps:
        return ""

    n = len(mod_exps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle("Grokking vs Sparsity (Modular Addition)", fontsize=14)

    for idx, exp in enumerate(sorted(mod_exps, key=lambda e: e["hidden_dim"])):
        ax1 = axes[0][idx]
        h = exp["history"]

        color_acc = "tab:blue"
        color_sp = "tab:red"

        ax1.plot(h["epochs"], h["test_acc"], color=color_acc, label="Test Acc")
        ax1.plot(h["epochs"], h["train_acc"], color=color_acc, linestyle="--",
                 alpha=0.5, label="Train Acc")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy", color=color_acc)
        ax1.tick_params(axis="y", labelcolor=color_acc)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = ax1.twinx()
        ax2.plot(h["epochs"], h["zero_fraction"], color=color_sp,
                 label="Zero Fraction")
        ax2.plot(h["epochs"], h["dead_neuron_fraction"], color="tab:orange",
                 linestyle=":", label="Dead Fraction")
        ax2.set_ylabel("Sparsity", color=color_sp)
        ax2.tick_params(axis="y", labelcolor=color_sp)
        ax2.set_ylim(-0.05, 1.05)

        ax1.set_title(f"h={exp['hidden_dim']}")
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "grokking_vs_sparsity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_width_vs_sparsity(experiment_summaries: list, output_dir: str) -> str:
    """Plot final sparsity metrics vs hidden width, colored by task.

    Parameters
    ----------
    experiment_summaries : list of dict
        Per-experiment summary statistics.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    tasks = {}
    for s in experiment_summaries:
        task = s["task"]
        if task not in tasks:
            tasks[task] = {"widths": [], "dead_fracs": [], "zero_fracs": [],
                           "test_accs": [], "zero_changes": []}
        tasks[task]["widths"].append(s["hidden_dim"])
        tasks[task]["dead_fracs"].append(s["final_dead_frac"])
        tasks[task]["zero_fracs"].append(s["final_zero_frac"])
        tasks[task]["test_accs"].append(s["final_test_acc"])
        tasks[task]["zero_changes"].append(s["zero_frac_change"])

    for task_name, vals in sorted(tasks.items()):
        label = task_name.replace("_", " ").replace("mod97", "").title().strip()
        ax1.plot(vals["widths"], vals["zero_fracs"], "o-", label=label, markersize=8)
        ax2.plot(vals["widths"], vals["zero_changes"], "s-", label=label, markersize=8)

    ax1.set_xlabel("Hidden Width")
    ax1.set_ylabel("Final Zero Fraction")
    ax1.set_title("Final Sparsity vs Model Width")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Hidden Width")
    ax2.set_ylabel("Zero Fraction Change (final - initial)")
    ax2.set_title("Sparsity Change During Training vs Width")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "width_vs_sparsity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(results: dict, output_dir: str) -> list:
    """Generate all analysis plots.

    Parameters
    ----------
    results : dict
        Full results from run_all_experiments.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    list of str
        Paths to generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    plots = []
    plots.append(plot_sparsity_evolution(results["experiments"], output_dir))
    plots.append(plot_accuracy_and_sparsity(results["experiments"], output_dir))
    plots.append(plot_width_vs_sparsity(results["experiment_summaries"], output_dir))

    return [p for p in plots if p]
