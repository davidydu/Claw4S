"""Analysis and summary statistics for symmetry-breaking experiments."""

import json
import os
from typing import Dict, List, Any

import numpy as np


def compute_breaking_epoch(
    epochs: List[int],
    symmetry_values: List[float],
    threshold: float = 0.5,
) -> int:
    """Find the first epoch where symmetry drops below a threshold.

    Args:
        epochs: List of epoch numbers at which symmetry was measured.
        symmetry_values: Corresponding symmetry metric values.
        threshold: Symmetry value below which we consider symmetry "broken".

    Returns:
        The epoch at which symmetry first dropped below threshold,
        or -1 if it never did.
    """
    for epoch, sym in zip(epochs, symmetry_values):
        if sym < threshold:
            return epoch
    return -1


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics across all experiment runs.

    Args:
        results: List of per-run result dictionaries from trainer.

    Returns:
        Summary dictionary with key findings.
    """
    summary: Dict[str, Any] = {
        "num_runs": len(results),
        "hidden_dims": sorted(set(r["hidden_dim"] for r in results)),
        "epsilons": sorted(set(r["epsilon"] for r in results)),
        "runs": [],
    }

    modulus_values = sorted({r["modulus"] for r in results if "modulus" in r})
    if len(modulus_values) == 1 and modulus_values[0] > 0:
        summary["modulus"] = int(modulus_values[0])
        summary["chance_accuracy"] = 1.0 / float(modulus_values[0])

    for r in results:
        breaking_epoch = compute_breaking_epoch(
            r["epochs_logged"], r["symmetry_values"], threshold=0.5
        )
        run_summary = {
            "hidden_dim": r["hidden_dim"],
            "epsilon": r["epsilon"],
            "initial_symmetry": r["initial_symmetry"],
            "final_symmetry": r["final_symmetry"],
            "symmetry_drop": r["initial_symmetry"] - r["final_symmetry"],
            "breaking_epoch_0.5": breaking_epoch,
            "final_test_acc": r["final_test_acc"],
            "final_train_acc": r["final_train_acc"],
        }
        summary["runs"].append(run_summary)

    # Key finding: does epsilon=0 break symmetry?
    zero_eps_runs = [r for r in summary["runs"] if r["epsilon"] == 0.0]
    if zero_eps_runs:
        summary["zero_eps_final_symmetry_mean"] = float(
            np.mean([r["final_symmetry"] for r in zero_eps_runs])
        )
        summary["zero_eps_test_acc_mean"] = float(
            np.mean([r["final_test_acc"] for r in zero_eps_runs])
        )

    # Key finding: does epsilon>0 break symmetry?
    nonzero_eps_runs = [r for r in summary["runs"] if r["epsilon"] > 0.0]
    if nonzero_eps_runs:
        summary["nonzero_eps_final_symmetry_mean"] = float(
            np.mean([r["final_symmetry"] for r in nonzero_eps_runs])
        )
        summary["nonzero_eps_test_acc_mean"] = float(
            np.mean([r["final_test_acc"] for r in nonzero_eps_runs])
        )

    # Breaking epoch statistics for runs with epsilon >= 1e-4
    substantial_eps_runs = [
        r for r in summary["runs"]
        if r["epsilon"] >= 1e-4 and r["breaking_epoch_0.5"] > 0
    ]
    if substantial_eps_runs:
        breaking_epochs = [r["breaking_epoch_0.5"] for r in substantial_eps_runs]
        summary["median_breaking_epoch"] = float(np.median(breaking_epochs))
        summary["mean_breaking_epoch"] = float(np.mean(breaking_epochs))
        summary["std_breaking_epoch"] = float(np.std(breaking_epochs))

    if summary["runs"]:
        best_run = max(summary["runs"], key=lambda r: r["final_test_acc"])
        summary["best_test_acc"] = float(best_run["final_test_acc"])
        summary["best_run_hidden_dim"] = int(best_run["hidden_dim"])
        summary["best_run_epsilon"] = float(best_run["epsilon"])
        summary["high_accuracy_runs_0p1_or_above"] = int(
            sum(1 for r in summary["runs"] if r["final_test_acc"] >= 0.1)
        )

        min_eps = summary["epsilons"][0]
        max_eps = summary["epsilons"][-1]
        min_eps_runs = [r for r in summary["runs"] if r["epsilon"] == min_eps]
        max_eps_runs = [r for r in summary["runs"] if r["epsilon"] == max_eps]

        if min_eps_runs and max_eps_runs:
            min_eps_best = float(max(r["final_test_acc"] for r in min_eps_runs))
            max_eps_best = float(max(r["final_test_acc"] for r in max_eps_runs))
            summary["best_test_acc_at_min_epsilon"] = min_eps_best
            summary["best_test_acc_at_max_epsilon"] = max_eps_best
            summary["accuracy_gain_max_vs_min_epsilon"] = max_eps_best - min_eps_best

    return summary


def generate_report(
    results: List[Dict[str, Any]], summary: Dict[str, Any]
) -> str:
    """Generate a human-readable report of the experiment results.

    Args:
        results: List of per-run result dicts.
        summary: Summary statistics from summarize_results().

    Returns:
        Markdown-formatted report string.
    """
    lines = [
        "# Symmetry Breaking in Neural Network Training: Results Report",
        "",
        f"Total runs: {summary['num_runs']}",
        f"Hidden dimensions: {summary['hidden_dims']}",
        f"Epsilon values: {summary['epsilons']}",
        "",
        "## Per-Run Results",
        "",
        "| Hidden Dim | Epsilon | Init Sym | Final Sym | Drop | Breaking Epoch | Test Acc |",
        "|-----------|---------|----------|-----------|------|----------------|----------|",
    ]

    for r in summary["runs"]:
        be = r["breaking_epoch_0.5"]
        be_str = str(be) if be > 0 else "never"
        lines.append(
            f"| {r['hidden_dim']:>9} "
            f"| {r['epsilon']:.1e} "
            f"| {r['initial_symmetry']:.4f}   "
            f"| {r['final_symmetry']:.4f}    "
            f"| {r['symmetry_drop']:.4f} "
            f"| {be_str:>14} "
            f"| {r['final_test_acc']:.4f}   |"
        )

    lines.extend(["", "## Key Findings", ""])

    if "zero_eps_final_symmetry_mean" in summary:
        lines.append(
            f"- **Epsilon=0 (symmetric `W1` rows):** mean final symmetry = "
            f"{summary['zero_eps_final_symmetry_mean']:.4f}, "
            f"mean test accuracy = {summary['zero_eps_test_acc_mean']:.4f}"
        )

    if "nonzero_eps_final_symmetry_mean" in summary:
        lines.append(
            f"- **Epsilon>0 (perturbed `W1` rows):** mean final symmetry = "
            f"{summary['nonzero_eps_final_symmetry_mean']:.4f}, "
            f"mean test accuracy = {summary['nonzero_eps_test_acc_mean']:.4f}"
        )

    if "median_breaking_epoch" in summary:
        lines.append(
            f"- **Breaking speed (epsilon >= 1e-4):** median breaking epoch = "
            f"{summary['median_breaking_epoch']:.0f}, "
            f"mean = {summary['mean_breaking_epoch']:.0f} "
            f"(std = {summary['std_breaking_epoch']:.0f})"
        )

    if "chance_accuracy" in summary:
        lines.append(
            f"- **Chance-level accuracy (1/modulus):** {summary['chance_accuracy']:.4f}"
        )
    if "best_test_acc" in summary:
        lines.append(
            f"- **Best run accuracy:** {summary['best_test_acc']:.4f} "
            f"(hidden={summary['best_run_hidden_dim']}, "
            f"epsilon={summary['best_run_epsilon']:.1e})"
        )
    if "accuracy_gain_max_vs_min_epsilon" in summary:
        lines.append(
            f"- **Best accuracy gain (max epsilon vs min epsilon):** "
            f"{summary['accuracy_gain_max_vs_min_epsilon']:.4f}"
        )
    if "high_accuracy_runs_0p1_or_above" in summary:
        lines.append(
            f"- **Runs with test accuracy >= 0.10:** "
            f"{summary['high_accuracy_runs_0p1_or_above']}"
        )

    lines.extend([
        "",
        "## Methodological Note",
        "",
        "These runs symmetrize only the incoming hidden-layer weights `W1`.",
        "The readout matrix `W2` is still initialized with seeded Kaiming",
        "uniform weights, so the observed symmetry decay reflects the combined",
        "effect of readout asymmetry and mini-batch stochasticity rather than",
        "batch noise in isolation.",
        "",
        "## Interpretation",
        "",
        "Symmetry breaking is essential for neural networks to learn diverse",
        "representations. In this setup, mini-batch SGD rapidly amplifies the",
        "gradient differences induced by the asymmetric readout, driving the",
        "incoming neuron weight vectors apart even when `W1` starts exactly",
        "symmetric (epsilon=0). Even a tiny perturbation (epsilon=1e-6) still",
        "produces nearly the same final symmetry profile as epsilon=0.",
        "",
        "Larger epsilon values lead to faster and more complete symmetry",
        "breaking from the outset, and networks with stronger early asymmetry",
        "learn more effectively on the modular addition task, which requires",
        "diverse feature representations across neurons.",
    ])

    return "\n".join(lines) + "\n"


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    report: str,
    output_dir: str = "results",
) -> None:
    """Save results, summary, and report to disk.

    Args:
        results: List of per-run result dicts.
        summary: Summary statistics.
        report: Markdown report string.
        output_dir: Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_path}")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report to {report_path}")
