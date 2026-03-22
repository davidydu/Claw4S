"""Generate publication-quality plots of loss curves and fits."""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.curve_fitting import FUNCTIONAL_FORMS


TASK_LABELS = {
    "mod_add": "Modular Addition (mod 97)",
    "mod_mul": "Modular Multiplication (mod 97)",
    "regression": "Regression (Random Features)",
    "classification": "Classification (Random Features)",
}

FORM_COLORS = {
    "power_law": "#e74c3c",
    "exponential": "#3498db",
    "stretched_exp": "#2ecc71",
    "log_power": "#9b59b6",
}

FORM_LABELS = {
    "power_law": "Power Law",
    "exponential": "Exponential",
    "stretched_exp": "Stretched Exp.",
    "log_power": "Log-Power",
}


def load_full_curves(results_dir: str = "results") -> list[dict]:
    """Load full curve data from results/full_curves.json."""
    path = os.path.join(results_dir, "full_curves.json")
    with open(path) as f:
        return json.load(f)


def plot_loss_curves_with_fits(
    curves: list[dict],
    output_dir: str = "results",
    skip_epochs: int = 10,
) -> str:
    """Plot a 4x3 grid of loss curves with fitted functions overlaid.

    Returns the path to the saved figure.
    """
    tasks = ["mod_add", "mod_mul", "regression", "classification"]
    hidden_sizes = [32, 64, 128]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    fig.suptitle(
        "Training Loss Curves with Fitted Functional Forms",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Index curves by (task, hidden_size)
    curve_map = {}
    for c in curves:
        curve_map[(c["task"], c["hidden_size"])] = c

    for row, task in enumerate(tasks):
        for col, hs in enumerate(hidden_sizes):
            ax = axes[row, col]
            key = (task, hs)
            if key not in curve_map:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            c = curve_map[key]
            epochs = np.array(c["epochs"])
            losses = np.array(c["losses"])

            # Plot raw data
            ax.plot(epochs, losses, "k.", markersize=0.5, alpha=0.3, label="Data")

            # Plot fits (only on the fitted range)
            t_fit = epochs[skip_epochs:]
            for fit in c["fits"]:
                if not fit["converged"]:
                    continue
                form_name = fit["form"]
                func = FUNCTIONAL_FORMS[form_name]["func"]
                params = [fit["params"][pn] for pn in fit["param_names"]]
                y_pred = func(t_fit, *params)
                color = FORM_COLORS.get(form_name, "gray")
                label = FORM_LABELS.get(form_name, form_name)
                linewidth = 2.0 if fit == c["fits"][0] else 1.0
                linestyle = "-" if fit == c["fits"][0] else "--"
                ax.plot(
                    t_fit,
                    y_pred,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=label,
                    alpha=0.8,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if row == 0:
                ax.set_title(f"Hidden={hs}")
            if col == 0:
                ax.set_ylabel(f"{TASK_LABELS[task]}\nLoss", fontsize=9)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "loss_curves_with_fits.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")
    return out_path


def plot_aic_comparison(
    curves: list[dict],
    output_dir: str = "results",
) -> str:
    """Plot AIC comparison bar chart for all runs.

    Returns path to saved figure.
    """
    tasks = ["mod_add", "mod_mul", "regression", "classification"]
    hidden_sizes = [32, 64, 128]
    forms = ["power_law", "exponential", "stretched_exp", "log_power"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("AIC Comparison by Task", fontsize=13, fontweight="bold")

    curve_map = {}
    for c in curves:
        curve_map[(c["task"], c["hidden_size"])] = c

    for ti, task in enumerate(tasks):
        ax = axes[ti]
        x = np.arange(len(forms))
        width = 0.25

        for hi, hs in enumerate(hidden_sizes):
            key = (task, hs)
            if key not in curve_map:
                continue
            c = curve_map[key]
            # Get AIC for each form
            aic_map = {f["form"]: f["aic"] for f in c["fits"]}
            aic_vals = [aic_map.get(fn, np.nan) for fn in forms]

            # Normalize relative to minimum AIC for readability
            min_aic = min(v for v in aic_vals if np.isfinite(v))
            delta_aic = [v - min_aic if np.isfinite(v) else 0 for v in aic_vals]

            colors = [FORM_COLORS[fn] for fn in forms]
            ax.bar(x + hi * width, delta_aic, width, label=f"h={hs}", alpha=0.7)

        ax.set_title(TASK_LABELS[task], fontsize=9)
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [FORM_LABELS[fn] for fn in forms], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylabel("Delta AIC" if ti == 0 else "")
        if ti == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "aic_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")
    return out_path


def plot_exponent_distributions(
    results_json_path: str = "results/results.json",
    output_dir: str = "results",
) -> str:
    """Plot distributions of key exponents grouped by task."""
    with open(results_json_path) as f:
        data = json.load(f)

    universality = data["universality"]
    exponents = universality["exponents_by_task"]

    tasks = list(exponents.keys())
    forms_present = set()
    for task_data in exponents.values():
        forms_present.update(task_data.keys())

    fig, axes = plt.subplots(1, len(forms_present), figsize=(4 * len(forms_present), 4))
    if len(forms_present) == 1:
        axes = [axes]

    fig.suptitle("Exponent Distributions by Task", fontsize=13, fontweight="bold")

    for fi, form in enumerate(sorted(forms_present)):
        ax = axes[fi]
        positions = []
        labels = []
        data_points = []

        for ti, task in enumerate(tasks):
            vals = exponents.get(task, {}).get(form, [])
            if vals:
                positions.append(ti)
                labels.append(task.replace("_", "\n"))
                data_points.append(vals)

        if data_points:
            bp = ax.boxplot(data_points, positions=range(len(data_points)), widths=0.5)
            # Also scatter the individual points
            for i, vals in enumerate(data_points):
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
                ax.scatter(
                    [i + j for j in jitter],
                    vals,
                    alpha=0.6,
                    s=30,
                    color=FORM_COLORS.get(form, "gray"),
                    zorder=3,
                )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{FORM_LABELS.get(form, form)}", fontsize=10)
        ax.set_ylabel("Exponent value")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "exponent_distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")
    return out_path


def generate_report(results_json_path: str = "results/results.json") -> str:
    """Generate a text report summarizing the analysis."""
    with open(results_json_path) as f:
        data = json.load(f)

    lines = []
    lines.append("=" * 70)
    lines.append("LOSS CURVE UNIVERSALITY ANALYSIS -- SUMMARY REPORT")
    lines.append("=" * 70)

    meta = data["metadata"]
    lines.append(f"\nConfiguration:")
    lines.append(f"  Tasks: {', '.join(meta['tasks'])}")
    lines.append(f"  Hidden sizes: {meta['hidden_sizes']}")
    lines.append(f"  Epochs: {meta['n_epochs']} (skip first {meta['skip_epochs']})")
    lines.append(f"  Total runs: {meta['total_runs']}")
    lines.append(f"  Runtime: {meta['elapsed_seconds']}s")

    uni = data["universality"]
    lines.append(f"\n--- Universality Summary ---")
    lines.append(f"  Majority best-fit form: {uni['majority_form']}")
    lines.append(
        f"  Fraction of runs with majority form: "
        f"{uni['majority_fraction']:.1%}"
    )
    lines.append(f"\n  Best-fit form counts:")
    for form, count in sorted(
        uni["form_counts"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"    {form}: {count}/{meta['total_runs']}")

    lines.append(f"\n  Best form by task:")
    for task, form in uni["best_form_by_task"].items():
        lines.append(f"    {task}: {form}")

    lines.append(f"\n--- Per-Run Results ---")
    lines.append(
        f"{'Task':<18} {'Hidden':>6} {'Params':>7} {'Final Loss':>11} "
        f"{'Best Form':<15} {'AIC':>10} {'BIC':>10}"
    )
    lines.append("-" * 80)

    for run in data["runs_summary"]:
        best = run["fits"][0] if run["fits"] else {}
        aic_str = f"{best.get('aic', float('inf')):.1f}"
        bic_str = f"{best.get('bic', float('inf')):.1f}"
        lines.append(
            f"{run['task']:<18} {run['hidden_size']:>6} {run['n_params']:>7} "
            f"{run['final_loss']:>11.6f} {run['best_form']:<15} "
            f"{aic_str:>10} {bic_str:>10}"
        )

    # Exponent summary
    lines.append(f"\n--- Key Exponents (by form) ---")
    for form, params in uni["exponents_by_form"].items():
        for pname, vals in params.items():
            arr = np.array(vals)
            lines.append(
                f"  {form}.{pname}: "
                f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
                f"min={arr.min():.4f}, max={arr.max():.4f}"
            )

    lines.append(f"\n{'=' * 70}")
    report = "\n".join(lines)
    return report
